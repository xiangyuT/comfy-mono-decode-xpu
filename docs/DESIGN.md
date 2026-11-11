# Design notes

## Target

ComfyUI prompt enhance runs `TextGenerate` on a Qwen3.5-9B text encoder. That model has 32 layers:
24 Gated-DeltaNet and 8 full attention, hidden 4096, vocab 248320. The weights are W4A8 (G16, FP8
group scales, ConvRot 256) and generation uses MTP speculative decoding.

One decode step does two things:
1. **Verify:** up to 8 tree rows through all 32 layers, then lm_head, then accept on the GPU.
2. **Draft:** one BF16 MTP block plus lm_head per drafted depth.

At 1–8 rows every linear is a GEMV and the step is memory-bound. The stock path spends most of its
time in launch gaps and many small kernels. The goal is to reach weight-streaming bandwidth while
keeping every output bit unchanged.

## Pieces

| file | what |
|---|---|
| `csrc/mono_common.hpp` | Shared device code for the persistent kernels: grid barrier (atomic RMW polling, ≤ 32 work-groups), stream GEMV engine, RMSNorm and ConvRot rotate/quantize ports. |
| `csrc/mono_gdn.sycl` | `layer_gdn`: input norm, quantize, in-projections, deferred causal conv, gates, gated delta rule, gated RMSNorm. `layer_gdn_core` returns the core rows. |
| `csrc/mono_layer.sycl` | `layer_pre` (norm + quantize + input projections) and `layer_post` (sigmoid gate, out proj, residual, post norm, MLP, residual). |
| `csrc/mono_mlp.sycl` | `mono_mlp`: one-launch MLP block, used outside the layer path. |
| `csrc/attn_port.sycl` | `attn_core`: Kitchen `decode_gqa` + combine + `tree_merge` in one launch, for the tree verify rows and the single draft row. |
| `csrc/rope_kv.sycl` | Partial split-half RoPE on q/k, the pass-through lanes, and the K/V cache write. |
| `csrc/fast_topk.sycl` | Single-row top-k (k ≤ 8) / argmax / both, in torch XPU tie order. |
| `csrc/w4a8_decode.sycl`, `csrc/act_prep.sycl` | Fused small-M W4A8 linear (SLM codebook, DPAS at M=8) and ConvRot rotate + INT8 quantize. |

### Stream GEMV engine

- The weights are 4-bit codes plus FP8 group scales (G16) and a per-channel scale.
- Activations are rotated (ConvRot 256) and quantized to INT8 per row.
- Each work-group streams contiguous runs of 16-row tiles. Partial sums are int32 and accumulate
  exactly; split-K uses int32 atomics.
- FP8 scales are applied through an exact pair table, so the float rounding sequence matches the
  reference kernel's.
- Tile-major layout `[N/16][K/block][16][block]` gives each lane wide, coalesced loads. Row-major
  weights, as DynamicVRAM hands them over, use the same engine with 4× wider per-lane loads.

### Persistent layer kernels

- A layer is a sequence of phases separated by device-wide barriers inside one launch.
- The GDN layer's single fused kernel spilled 1–3 KB per thread at 128 GRF. It is split into a
  core-only kernel (0 spill) plus `layer_post`, which made it faster.

### MTP draft

- The draft block holds BF16 weights. Its GEMMs stay on oneDNN and run at about 96% of DRAM bandwidth.
- Its attention goes through `rope_kv` + `attn_core`.
- The logits pick uses `fast_pick`: one pass gives the argmax plus a top-8 list, and the
  following `topk(k ≤ 8)` reuses the list.

### DynamicVRAM

- Each call takes the linear's GPU weight from `comfy.ops.CastBiasWeightContext`, the same VBAR
  cast the eager forward uses. The weight stays pinned until the kernel is enqueued.
- Module-level contract checks (layout, scale range, codebook) run once, on host-side metadata.
- Resident and DynamicVRAM copies have the same per-generate owner. They are keyed by the canonical
  module weight, so a new temporary VBAR cast address reuses the copy. Layout is chosen after
  prefill from actual weight storage and the live allocator/inference budget, including VBAR
  residency outside Torch. Component copies and graph/weight caches are dropped on success,
  cancellation or failure; the global allocator pool is kept.
- Stateful dispatch is an error-recovery boundary: once KV, recurrent state or x may have been
  written, the wrapper propagates errors instead of replaying the layer. Only pure operations may
  retry the stock path after disabling their accelerated route.

## Exactness

The bar is the same token sequence as the stock image on fixed MTP trees: the sha of the generated
ids for three prompts. Every kernel is first checked op-by-op against the eager kernel it replaces,
on real and adversarial inputs (`tests/`). What it took:

- **Same reduction order.** RMSNorm, the gated norm, softmax/LSE and the GEMV epilogues reproduce the
  reference's lane-stride and shuffle tree exactly.
- **Same build.** The single icpx invocation (`build_icpx.sh`) is required. torch's per-file
  SyclExtension build with a separate `-fsycl-link` produced different device code for identical
  source (1-ulp LSE differences in a verbatim copy of Kitchen's attention).
- **Contraction.** IGC fused `a*b + c*d` into `mad` even under `#pragma clang fp contract(off)`. The
  RoPE port keeps the products separate with an opaque runtime-zero XOR. The live OmniXPU rotary
  kernel rounds each product, but Kitchen's eager rope backend uses an FMA (`addcmul_`). So `rope_kv`
  self-checks against whatever backend serves the live call (150 wide-range rows plus the first real
  call) and turns itself off if they differ.
- **Tie order.** torch XPU `topk` on one long row (sbtopk gather + segmented sort) returns ties in
  gather order: iteration, then thread, then vector load, then lane. `argmax` returns the smallest
  index, with NaN first and -0 == +0. `fast_topk` encodes both orders in 64-bit ranks. It matched
  4200 checks, including tie-heavy rows and ±0.

## Measurements (B70, fixed tree4s1m8, giant prompt)

- Stock eager: about 30 tok/s, dominated by launch gaps between many small kernels.
- Now: 95–97 tok/s resident, 94–95 tok/s DynamicVRAM.
- One draft takes 2.74 ms of GPU time:
  - lm_head (int8, 1 GB): 1.73 ms
  - BF16 MTP block GEMMs: about 0.85 ms
  - small ops: about 90 µs

  The GEMMs already run at about 96% of DRAM bandwidth, so the draft is close to its floor unless
  its weights get smaller (not bit-exact).
- Fusing RoPE + KV write gave +4% resident and +8% DynamicVRAM, larger than its kernel-time share:
  removing about 15 launches per attention call also removed host-side gaps.

## Not done / ideas

- The eager q/k RMSNorm in attention layers uses torch's layer-norm kernel on a strided view. Porting
  it exactly needs that kernel's reduction order.
- The draft's int8 embedding gather is ConvRot-rotated. Its fp32 Hadamard GEMM goes through oneDNN,
  so a fused gather would have to match oneDNN's accumulation.
- XPU graph capture of the verify step (`graph_step.py`) works with resident weights. With the layer
  kernels the launch count is already low, so it is not part of the default stack.
