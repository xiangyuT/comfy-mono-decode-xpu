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
| `csrc/attn_dpas.sycl` | `attn_dpas`: decode attention (tree verify rows and the draft row) with DPAS, split by the actual context length, then a merge with the tree rows. Not bit-identical to Kitchen. |
| `csrc/rope_kv.sycl` | Q/K RMSNorm, partial split-half RoPE, pass-through lanes and K/V cache write; also the normalization-free RoPE entry. |
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
- `layer_gdn` (the core kernel, P1–P5) keeps every operation of the replaced chain in order; what
  changed is who runs it and when its inputs are requested. Its serial phases are latency-bound:
  - P1: the a / b BF16 linear (64 DPAS units in oneDNN order) runs on idle work-groups as soon as the
    R normed rows are published through an atomic counter, in the shadow of rot_quant. The Hadamard
    matrix and the a / b weights are requested into L2 at launch: the previous launch's weight
    stream evicts them, and each of rot_quant's 16 load-mad steps then waited on DRAM.
  - P3: the causal conv runs one (channel, pair of tree slots) per thread over all work-groups, with
    ctl in SLM; 8 slots per thread were ALU-bound (exp and IEEE division per slot).
  - P5: every entry's k, q, v and gates are staged in SLM up front; the row-group partials are
    double-buffered, so each entry needs one barrier; a non-committing entry recomputes its decayed /
    updated rows into temporaries instead of snapshotting the state; the gated norm of all output
    rows runs once after the recurrence. The recurrent state, conv state and the previous step's
    buffers are requested into L2 when the work-group's qkv / z GEMV share is done.
  - Measured during a real generate (per call, B70): 132 -> 119 us (P1 10.6 -> 8, P3 10.8 -> 5,
    P5 30 -> 27); about 0.3 ms per step in the verify segment.
  - A prefetch is not free: the issuing thread's next device-scope fence (every grid barrier) waits
    for it. Issued halfway through the qkv / z GEMV (a callback in the weight loop), the cold-data
    prefetch made the launch 20 us slower.
  - Small changes to one phase move others by a few us (one register allocation and schedule for the
    whole kernel); compare builds without instrumentation.
- Decode attention (`attn_dpas`, default for the tree verify rows and the draft row;
  `MONO_DECODE_XPU_ATTN_DPAS=0` for `attn_core`):
  - A work-group handles one (kv head, context chunk). Chunks follow the actual length on the device
    (at most 48, at least 32 tokens each; work-groups past the last chunk exit). A sub-group handles 8
    query rows and half of the head dim.
  - Per 16-token tile: S = Q K^T as 16 DPAS (K^T a column-major load of the K rows), online softmax
    per row inside the sub-group, O += P V as 8 DPAS (V a row-major block load). On Xe2 an
    accumulator, an A and a B fragment all hold lane = column, element i = row i, so the scores are
    already laid out as the P operand; P goes through a 256 B SLM tile because elementwise writes
    into BF16 A/B fragments hang the device with this compiler.
  - Registers decide the speed: the running max and sum sit one row per lane and fragments are read
    through unrolled `wi_data` loops, keeping the default 128 GRF. The 256 GRF mode removes the
    remaining spills but halves occupancy and is slower.
  - A second kernel merges the active chunks' (m, l, O) in FP32 with the tree rows each row admits.
  - Accuracy: P is rounded to BF16 for P V; against an fp64 reference its error matches or is below
    the Kitchen kernel's own error on the same inputs (max about 7.8e-3, one BF16 output rounding).
  - Device time at cap 11400, L 1000 / 4000 / 8000: verify (8 rows) 45 / 82 / 134 us, draft row
    27 / 50 / 90 us (`attn_core` verify 91 / 230 / 448 us, draft 35 / 59 / 113 us; Kitchen verify
    273 / 484 / 843 us). Not helpful: cooperative K / V staging in SLM per tile (the staging stores
    cost more than the shared loads saved), S^T = K Q^T with Q^T from SLM.
- `attn_core` (with `MONO_DECODE_XPU_ATTN_DPAS=0`) keeps Kitchen's split-K structure (up to 32 splits of the cache capacity, 64-key
  tiles, per-split online softmax, then combine and tree merge), which fixes every row's rounding.
  Two scheduling choices change only which work-group runs what:
  - Tasks are numbered with the split slowest, so the live splits (those below the current length)
    are dealt to all work-groups. Numbering the split fastest with 32 splits and 32 work-groups gave
    work-group w split w only, so a short prefix ran on two or three work-groups.
  - Verify rows of one kv head (four Kitchen packs of four rows) share each staged K/V tile: every
    key is scored for all 16 rows and each value read serves the work-item's four rows. The draft's
    single row keeps one row per work-item.

### MTP draft

- The draft block holds BF16 weights. Its GEMMs stay on oneDNN and run at about 96% of DRAM bandwidth.
- Its attention goes through `rope_kv` + `attn_dpas`.
- The logits pick uses `fast_pick`: one pass gives the argmax plus a top-8 list, and the
  following `topk(k ≤ 8)` reuses the list.
- Under DynamicVRAM the head's BF16 linears keep, for the generate, the tensor comfy hands to
  `F.linear` (a copy of the cast, about 0.5 GB, when free memory allows), so later drafts skip the
  per-call cast. The first reuse per generate is compared bit for bit with comfy's forward.
- ComfyUI's `auto` tree (`qwen35.best_widths`) scores width chains by accepted tokens per
  millisecond. Off CUDA it times with host clocks: a draft counts only its launch time, and the
  verify clock includes the drafts' remaining device work only because the sampling filters wait
  for the device (`t[mask] = v` runs `nonzero()`). `autotree.py` keeps upstream's shapes, scoring
  and smoothing and feeds it XPU event timings instead (first draft → verify forward, per draft;
  verify forward → next step's first draft). With that, inside MTP generates the filters run as
  `masked_fill_` (same values, no wait). Removing the wait while keeping host clocks made `auto`
  settle on one draft per step and lowered throughput, so the two are installed together.

### DynamicVRAM

- Each call takes the linear's GPU weight from `comfy.ops.CastBiasWeightContext`, the same VBAR
  cast the eager forward uses. The weight stays pinned until the kernel is enqueued.
- On v0.45, prepared `CombinedLinear` groups own the resident codes and per-row scales; their
  members' `_v_weight` fields have been cleared. Hooks take fresh row views of the current group
  while all members are prefetched, using the upstream prefetch pins. They do not recast individual
  members or keep a group view after the forward. The tile-copy key remains the canonical module
  weight, independent of temporary VBAR addresses. Q/K/V and MLP outputs remain contiguous.
- Module-level contract checks (layout, scale range, codebook) run once, on host-side metadata.
- With tile-major copies a block's kernel arguments are all owned by the generate (tile copies,
  FP32 metadata copies, norm weights; GDN BF16 operands borrowed from a cast are copied), so the
  first call records the block's plan and later calls reuse it without recasting seven linears.
  This took a verify step's host launch time from 12.3 to 2.2 ms (H3 Image).
- Resident and DynamicVRAM copies have the same per-generate owner. They are keyed by the canonical
  module weight, so a new temporary VBAR cast address reuses the copy. Layout is chosen after
  prefill from actual weight storage and the live allocator/inference budget, including VBAR
  residency outside Torch. Component copies and graph/weight caches are dropped on success,
  cancellation or failure; the global allocator pool is kept.
- Stateful dispatch is an error-recovery boundary: once KV, recurrent state or x may have been
  written, the wrapper propagates errors instead of replaying the layer. Only pure operations may
  retry the stock path after disabling their accelerated route.
- The fused RoPE and Q/K norm + RoPE + KV-write routes compare their first call in each generate
  with the live operator chain. A numerical mismatch (for example another Torch build with a
  different RMSNorm reduction order) rewrites the KV slots from the live results, returns them and
  disables only that route. A native error in one of these sub-routes disables that route and
  propagates; the layer kernels stay enabled.

## Exactness

Kernel exactness is checked against the replaced operator chain with the same input, weights, dtype
and layout, including the v0.45 combined input-act entry. The earlier v0.43 model checks also matched
whole fixed-tree token sequences. Current workflow qualification reviews the expanded prompt and
saved image; keeping dense outputs may change downstream dispatch, so whole-generation token
identity is not its numerical boundary. The kernel checks use real operations and adversarial
inputs (`tests/`). What it took:

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

## Measurements (B70, official H3 Image prompt enhance, auto MTP, sampling on, DynamicVRAM)

- Decode: 98–101 tok/s on the public v0.1.0 code, about 115–117 tok/s with the attention scheduling,
  plan reuse, device-timed `auto` and the draft head's kept weights. `auto` now settles on about 2.4–2.6
  drafts and 2.8–2.9 accepted tokens per step.
- Per step (about 24 ms) the 32 verify layers run back to back on the device (0.13 ms of gaps in
  total); lm_head (1.7 ms per call) and the BF16 head GEMMs are at the read roof. The remaining
  distance to the bytes floor (598 GB/s for weights and KV, no idle) is about 10 ms per step:
  W4 GEMV phases at 70–80% of the roof (about 4 ms), long-context attention (about 2 ms, growing
  with the context), host/sync idle between a step's sync and its first draft (about 2.5 ms) and
  small kernels (0.65 ms). The bytes floor for this tree is about 14.6 ms per step, or about 195
  tok/s at 2.85 tokens per step.

## Not done / ideas

- Tried without gain (2026-11-12): attention register prefetch of the next tile (exact, 12% slower);
  32-row attention packs (slower at short contexts); skipping `layer_post`'s pair-table rebuilds (no
  change); L2 prefetch of the next GEMV phase by idle work-groups in `layer_post` (64–512 KB; the
  extra code changed register allocation and slowed gate/up more than it saved).
- Tried without gain in `layer_gdn` (2026-11-12): a callback in the qkv / z GEMV loop that requests P3 /
  P5 inputs halfway through (P1, P2 and P3 all slower); issuing the recurrent state loads after the
  prologue's loads (P5 faster, P2 slower by the same amount).

- **2026-11-12: Q/K norm fusion implemented.** The D256 gated Q route preserves the installed
  Torch vec4/two-SIMD32 reduction; dense K preserves Omni's eight BS32-block reduction. The BF16
  normalized intermediate stays explicit before partial RoPE and the cache write. Q/K outputs
  are contiguous. This closes the earlier independent Q/K norm item; draft projection/MLP,
  embedding, lm_head and generation scheduling remain separate work.
- The draft's int8 embedding gather is ConvRot-rotated. Its fp32 Hadamard GEMM goes through oneDNN,
  so a fused gather would have to match oneDNN's accumulation.
- XPU graph capture of the verify step (`graph_step.py`) works with resident weights. With the layer
  kernels the launch count is already low, so it is not part of the default stack.
