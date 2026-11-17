# Design notes

## Target

ComfyUI prompt enhance runs `TextGenerate` on a Qwen3.5-9B text encoder. That model has 32 layers:
24 Gated-DeltaNet and 8 full attention, hidden 4096, vocab 248320. The weights are W4A8 (G16, FP8
group scales, ConvRot 256) and generation uses MTP speculative decoding.

One decode step does two things:
1. **Verify:** up to 8 tree rows through all 32 layers, then lm_head, sampling and the accept walk.
2. **Draft:** one BF16 MTP block plus lm_head per drafted depth.

At 1–8 rows every linear is a GEMV and the step is memory-bound. The stock path spends most of its
time in launch gaps and many small kernels. The goal is to stream the weights at DRAM bandwidth with
few launches, keeping the verify path's numerics.

## Exactness

- **Verify path:** every kernel gives the same bits as the operator chain it replaces, for the same
  inputs, weights, dtype and layout. This covers norms, rotate/quantize, the W4A8 GEMVs and their
  epilogues, the Gated-DeltaNet core, RoPE/KV writes and the verify sampling's top-k. The tests in
  `tests/` check this on real operations and adversarial inputs.
- **Exception, decode attention:** `attn_dpas` splits the context differently from Kitchen's
  `decode_gqa` + `tree_merge` and uses DPAS. It is checked against an fp64 reference, with Kitchen's
  own error on the same inputs as the bound. `MONO_DECODE_XPU_ATTN_DPAS=0` selects `attn_core`, a
  bit-identical port.
- **Drafts:** the fused draft head follows the eager arithmetic in FP32 but not its reduction order.
  The experimental int4 draft lm_head may also pick different drafts. A draft only proposes tokens;
  the verify decides every output token.
- What exactness takes:
  - The same reduction order: RMSNorm, the gated norm, softmax/LSE and the GEMV epilogues reproduce
    the reference's lane-stride and shuffle trees.
  - The same build: one icpx invocation (`build_icpx.sh`). torch's per-file SyclExtension build with
    a separate device link produced different device code for identical source.
  - Explicit contraction control: IGC fuses `a*b + c*d` into `mad` even under
    `#pragma clang fp contract(off)`. The RoPE port keeps the products separate. Because the live
    rotary backend may itself fuse, `rope_kv` checks itself against the backend that serves the call
    and turns itself off if they differ.
  - The same tie order: `fast_topk` and `topk_rows` reproduce torch XPU's `topk` / `argmax` order,
    ties and ±0 included.

## Pieces

| file | what |
|---|---|
| `csrc/mono_common.hpp` | Shared device code for the persistent kernels: grid barrier, stream GEMV engine, RMSNorm and ConvRot rotate/quantize ports. |
| `csrc/mono_launch.hpp` | Host launch planning: resident work-groups, accumulator sizing, decode-attention chunk count. |
| `csrc/mono_gdn.sycl` | `layer_gdn`: input norm, quantize, in-projections, deferred causal conv, gates, gated delta rule, gated RMSNorm. |
| `csrc/mono_layer.sycl` | `layer_pre` (norm, quantize, input projections) and `layer_post` (sigmoid gate, out projection, residual, post norm, MLP, residual). |
| `csrc/mono_mlp.sycl` | `mono_mlp`: one-launch MLP block for decode rows outside the layer path. |
| `csrc/attn_dpas.sycl` | `attn_dpas`: decode attention for the tree verify rows and the draft row with DPAS. |
| `csrc/attn_port.sycl` | `attn_core`: bit-identical port of Kitchen's decode attention + tree merge. |
| `csrc/rope_kv.sycl` | Q/K RMSNorm, partial RoPE and the K/V cache write in one launch. |
| `csrc/mtp_loop.sycl` | The device MTP loop's per-step bookkeeping. |
| `csrc/draft_fused.sycl`, `csrc/draft_lmhead.sycl` | The fused draft head's small ops; `draft_pick` (draft top-n and its outputs). |
| `csrc/fast_topk.sycl`, `csrc/topk_rows.sycl` | Single-row top-k / argmax (draft and first-token logits); verify-sampling top-k. |
| `csrc/w4a8_decode.sycl`, `csrc/act_prep.sycl` | Small-M W4A8 linear and ConvRot rotate + INT8 quantize, for calls outside the layer path. |

## Verify layers

- **Gated-DeltaNet layer:** `layer_gdn` (norm through gated norm), then `layer_post`.
- **Full-attention layer:** `layer_pre` (Q / K / V projections), the fused Q/K norm + RoPE + KV
  write, `attn_dpas`, then `layer_post`.

### Stream GEMV engine

- Weights are 4-bit codes with FP8 group scales (G16) and a per-channel scale. Activations are
  rotated (ConvRot 256) and quantized to INT8 per row.
- The output is split into 16-column tiles, and each work-group streams contiguous runs of them.
  Products are INT8 DPAS with INT32 partial sums; split-K partials are added with INT32 atomics, so
  the result does not depend on the order.
- The FP8 group scales go through an exact pair table: for each FP8 scale and pair of codes it holds
  the two INT8 values the reference computes. The table depends only on the codebook. Each launch
  fills one global copy, and every GEMV phase loads it into SLM.
- Tile-major weights (`[N/16][K/32][16][16]`, made once per generate) give each sub-group one
  contiguous block per step. Row-major weights, as DynamicVRAM hands them over, use the same engine
  with wider per-lane loads.
- `layer_post` interleaves output tiles across work-groups when the tile count divides evenly; the
  bytes and the integer accumulation are unchanged.
- `layer_post` can also take the Hadamard matrix as its two distinct 16x16 tiles. Kitchen's regular
  H256 consists of ±copies of one H16 tile, and the kernel picks the sign from the block indices.
  The Python layer rebuilds the full matrix from the two tiles and compares it bit for bit once;
  any other matrix keeps the full-matrix route.

### Persistent layer kernels

- A layer runs as a sequence of phases in one launch, separated by device-wide barriers.
- Each work-group has 1024 threads, all hardware threads of one Xe core, and a launch uses at most one
  work-group per core. The barrier spins, so every work-group must be resident at once.
- SLM accumulators are sized to the launch's tile share: fewer cores (or an explicit worker count)
  give each work-group more tiles. A launch whose SLM would exceed the device limit is rejected
  before it starts.
- `grid_sync`: every thread releases its stores to the work-group; thread 0 releases to the device,
  arrives, polls and acquires once for the work-group, whose threads share the Xe core's L1. The
  fences are written per role because the compiler's lowering of all-thread device fences depended on
  unrelated code elsewhere in the kernel.
- The GDN block is two launches, `layer_gdn` (P1–P5) and `layer_post`. As one fused kernel it spilled
  registers at 128 GRF and was slower.
- `layer_gdn` keeps every operation of the replaced chain in order. It changes only which threads run
  each operation and when inputs are requested:
  - P1: input RMSNorm and rotate/quantize of the R rows. The a / b BF16 linear (in oneDNN's
    accumulation order) runs on idle work-groups once the normed rows are published.
  - P2: the qkv | z GEMV. After its share of the GEMV, a work-group requests the cold inputs of
    P3 / P5 into L2: the recurrent state, the conv state and the previous step's buffers.
  - P3: the deferred causal conv, one (channel, pair of tree slots) per thread.
  - P5: per value head, the replay of the previous step's accepted entries and the verify entries.
    Inputs are staged in SLM up front, and row-group partials are double-buffered (one barrier per
    entry). A non-committing entry recomputes into temporaries instead of snapshotting the state.
    The gated norm of all output rows runs once at the end.

### Decode attention

- `attn_dpas` (default) has two kernels.
  - Kernel 1: a work-group takes one (kv head, context chunk). The number of chunks comes from the
    cache capacity, and the actual length is split among them, at least 32 tokens each; chunks
    beyond the length exit. A sub-group handles 8 query rows and half the head dim.
  - Each 16-token tile does S = Q Kᵀ and O += P V on DPAS, with an online softmax per row.
  - Kernel 2 merges the chunks' (m, l, O) in FP32 with the tree rows each row's mask admits.
  - On Xe2 the accumulator and the A / B fragments share one layout, so the scores are already in
    P's operand layout. P still passes through a small SLM tile, because elementwise writes into BF16
    A/B fragments hang the device with this compiler.
  - The kernel stays at 128 GRF. The 256 GRF mode removes spills but halves occupancy and is slower.
- `attn_core` keeps Kitchen's split-K structure (splits of the cache capacity, per-split online
  softmax, combine, tree merge), which fixes every row's rounding. It changes only which work-group
  runs what: live splits are spread over all work-groups, and a kv head's verify rows share each
  staged K/V tile.

## Decode loop

### Device MTP loop

`mtp_loop.py` replaces the body of upstream's `_generate_mtp` for the supported configuration:
batch 1, no penalties, the Qwen3.5 caches. The per-step host state moves to the device:
- the attention position and length;
- the deferred-delta control row;
- the MTP KV counters;
- the accept walk and the next token / hidden row.

The draft, verify, lm_head and sampling calls are upstream's. The host queues step n + 1 and then
reads step n's record (committed tokens, auto-tree ranks) from a pinned ring, so a step's sync does not
idle the device. One step is queued past a stop and dropped. In this asynchronous mode an `auto` tree
change reaches the device one step later than in upstream's loop.

The loop runs only if upstream's `_generate_mtp` and the KV / tree classes it replaces match v0.45
(a source fingerprint). A failure inside it switches it off and reruns the generate with upstream's
loop.

### MTP draft

- **Fused draft head (default, inside the device loop):** keeps upstream's structure and the oneDNN
  GEMMs on BF16 weights owned for the generate (q | k | v and gate | up concatenated). It fuses the
  small ops around them:
  - the embedding row with the pre-fc norms;
  - the gate, add + norm and SiLU·mul;
  - the final norm with the lm_head's rotation and quantization.

  The ConvRot rotations run as 4-point butterflies. The lm_head is the INT8 GEMM, and `draft_pick`
  writes the top-n, argmax, auto-tree rank row and the children's verify tokens in one launch.
- **Eager head (outside the device loop):** the head's BF16 linears keep comfy's cast for the generate
  (`draft.py`; the first reuse is compared bit for bit with comfy's forward). The head can replay as an
  XPU graph (`graph_draft.py`), and the logits pick uses `fast_pick` (argmax plus a top-8 list that
  the following top-k reuses).
- **Two-stage draft lm_head (experimental, `MONO_DECODE_XPU_DRAFT_INT4=1`):**
  - An int4 copy of the int8 head (group 128, built per generate) scores the vocabulary through
    oneDNN's weight-only int4 GEMM.
  - `draft_pick` then recomputes the exact int8 logits of about 1000 candidates. Each thread keeps the
    best rows of a strided set, so adjacent token ids land on different threads.
  - The result equals the int8 draft whenever the exact top-7 lies among the candidates.

### Verify sampling top-k

Inside the device loop, upstream's verify sampling starts with `torch.topk(logits.float(), top_k)` over
8 x 248320. A `TorchFunctionMode` serves that call with `topk_rows`, which returns the same values and
indices in torch's order, ties included. Rows with NaN are not reproduced. `topk_rows` uses four
work-groups per row; each keeps the candidates above a radix threshold and ranks them by (key, gather
order), and the row's last work-group merges the four lists.

The sampling gets the BF16 logits as a stand-in for their FP32 copy (`sampling.verify_rows`). With
temperature 1 and top_k > 0 upstream only needs the shape and the top-k, so the copy is not made. The later top-p sort
of already sorted values returns them with a cached arange. Everything else in the sampling is
upstream's, and the sampled tokens equal upstream's from the same generator.

### `auto` tree timing

ComfyUI's `auto` tree (`qwen35.best_widths`) scores width chains by accepted tokens per millisecond.
Off CUDA it times with host clocks, which on XPU miss the draft's device time. `autotree.py` keeps
upstream's shapes, scoring and smoothing and feeds it XPU event timings instead. With that, inside MTP
generates, the sampling filters run as `masked_fill_` without waiting for the device. The two are
installed together: removing the wait while keeping host clocks makes `auto` pick too few drafts.

## DynamicVRAM

- Each call takes the linear's GPU weight from `comfy.ops.CastBiasWeightContext`, the same VBAR cast
  the eager forward uses. The weight stays pinned until the kernel is enqueued.
- On v0.45, prepared `CombinedLinear` groups own the resident codes and per-row scales. Hooks take
  row views of the current group while its members are prefetched, and keep no view after the forward.
- Module contract checks (layout, scale range, codebook) run once, on host-side metadata.
- Tile-major copies are owned per generate and keyed by the canonical module weight, so a new
  temporary VBAR address reuses the copy. With copies, all of a block's kernel arguments are owned by
  the generate. The first call records the block's plan, and later calls reuse it without recasting.
- The copy budget is decided after prefill from the actual weight storage, the allocator pool and the
  inference reserve. Under an observed AIMDO allocation hook, it may count the resident VBAR pages of
  inactive models on the device, which the hook reclaims as needed. Without copies the kernels read
  the row-major weights. Copies and caches are released on success, cancellation or failure.
- Stateful dispatch is an error boundary: once KV, the recurrent state or x may have been written, an
  error propagates instead of replaying the layer. Only pure operations retry the stock path, after
  disabling their own route.
- The fused RoPE routes compare their first call in each generate with the live operator chain. On a
  mismatch they rewrite the KV slots from the live result and disable only that route.

### Generation-owned weight reuse

`MONO_DECODE_XPU_WEIGHT_REUSE` defaults on and controls three operations together, without
disabling the existing session tiles or layer kernels when set to `0`:

- Skip body/MTP prefetch only for eligible BF16 decode rows when every layer has a usable tiled
  plan. Restore the original setting when plans become unavailable and on exit, including failure.
- Once prefetch is suppressed and all plans exist, clone the INT8 output head's packed data and
  quantization metadata within the live copy budget. Draft and verify share this generation-owned
  copy; release the original cast and drop the copy at generation exit. Insufficient budget or
  allocation failure keeps the ordinary cast path.
- Gather only needed INT8 embedding rows on the host when ComfyUI's current VBAR watermark
  indicates streaming and no prefetch is active. Recheck every lookup: CPU source weights alone
  do not prove non-residency. Existing non-resident draft/device-loop fallbacks stay independent.

`tiles.stats()` retains the budget, plan count, prefetch skips, embedding-wrapper row calls and
head-cache status for the current or last generation, without a user-facing profiler wrapper.

### Windows whole-feature observations (2026-11-17)

User-supplied MiniMax H3 image-edit logs: Arc Pro B70 (31907 MB VRAM), Windows, Python 3.13.14,
Torch 2.14.1+xpu, ComfyUI 0.45.0, DynamicVRAM, Kitchen 0.2.41, AIMDO 0.6.0 and
`omni_xpu_kernel 0.2.0b3+torch214.bmg`. Source/native build: `dae77d7`, before consolidation/rebase.
Both groups kept MONO and LAYER enabled. A disabled all three former weight-reuse switches; B
enabled them. One warmup and three measurements per server; reported token progress-bar `it/s`:

| run | A: all off, limit 128 (tok/s) | B: all on, limit 8196 (tok/s) |
|---|---:|---:|
| warmup (excluded) | 1.92 | 117.70 |
| 1 | 1.91 | 109.46 |
| 2 | 2.12 | 121.88 |
| 3 | 2.12 | 117.79 |
| measured median | 2.12 | 117.79 |

A reached 128 tokens each run; B's measured runs reached 2667, 4971 and 4602. Each B run reported
32 reused tile plans and a 1,018,112,000-byte owned INT8 head. Host-row counts were not recorded.
These are whole-feature observations, not a matched-length speedup or per-component attribution.
The tester separately reports short B at 20-30 tok/s and long A around 2 tok/s; those trials are
not in the supplied log table. Setup cost, auto MTP and generated text can affect reported rates.

After consolidation (`bc1bdb2`, before rebase), the tester reported all five XPU-dependent error
recovery cases passing (6.92 s, no skips), image-edit decode at 117.40 tok/s (4602/8196), and T2I
at 140.73 tok/s (4642/8192). These snippets establish decode execution, not final-image quality.
The five cases complement 82 locally passed runtime cases from that revision, not a single
87-case target run. The rebased source includes upstream #8 native changes and needs a matching
rebuilt extension and fresh target validation; earlier rates are not measurements of that build.

For the rebased source `3498d71ca816bbe8ca5ad325a064fc306dc4be0b`, the Windows BMG extension
was rebuilt and its 24 callable exports checked in an independent portable Python process.
The artifact SHA-256 is `DF5D114B674CBF925EEE4CB3F30877D3D526780C3236F2E2272A2C91037C4A29`.
The B70 tester subsequently reported **185 passed in 10.19 s**, without failures or skips, and
these decode observations:

| workflow | tokens / limit | logged tok/s |
|---|---:|---:|
| T2I | 2904 / 8192 | 137.48 |
| image edit | 4602 / 8196 | 118.22 |

These single-run rates are not a matched performance comparison or proof of complete workflows.
The platform probe reports an unresolved ISA FAIL and multiple visible XPUs despite the passing
suite. Full workflow logs, repeated-run completion, prompt/image inspection and recovery evidence
remain pending. See [B70 acceptance](../tests/README.md#b70-rebased-baseline-acceptance) for the
required evidence before marking this baseline fully retested; do not substitute earlier results.

### Removed prefill-trim experiment

With weight reuse enabled, the tester supplied one Windows run per setting/workflow before rebase:

| workflow | trim | generated tokens | logged tok/s | workflow seconds |
|---|---:|---:|---:|---:|
| T2I | 0 | 8192 | 140.46 | 87.96 |
| T2I | 1 | 6576 | 137.53 | 76.67 |
| image edit | 0 | 4602 | 117.05 | 109.25 |
| image edit | 1 | 4602 | 117.11 | 108.78 |

T2I trim reduced Torch reserved memory by 116 MiB, with allocated and reported free memory
unchanged. Its shorter total time is confounded by fewer generated tokens and downstream attention
length changing from 4097 to 533. Image edit's allocated/reserved/free readings were unchanged;
its 0.05% rate difference and 0.47 s total-time difference do not establish a benefit in one trial.
The T2I trim-off run reached its token limit; prompt completeness was not established by the log.

There is no demonstrated need for this extra synchronization/cache flush on these workflows.
`PREFILL_TRIM`, its boundary hook and logging are removed rather than made automatic. This is not
a claim of no benefit under every memory-pressure condition.

## Not in the default stack

- `graph_step.py`: XPU graph capture of the whole verify step (`MONO_DECODE_XPU_GRAPH=1`), resident
  weights only. With the layer kernels the launch count is already low.
- `attn_core` (`MONO_DECODE_XPU_ATTN_DPAS=0`), the bit-identical decode attention.
