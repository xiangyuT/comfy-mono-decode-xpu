# Environment variables

The variables are read when ComfyUI starts, when the custom node loads. `1` means on, `0` means off.

## Pieces (read by the node's `__init__.py`)

| variable | when unset | effect |
|---|---|---|
| `MONO_DECODE_XPU` | on | Master switch. `0` disables `W4`, `PREP`, `MONO` and `LAYER` unless a piece is explicitly enabled. |
| `MONO_DECODE_XPU_LAYER` | same as `MONO_DECODE_XPU` | Decode transformer blocks through the persistent layer kernels. Also installs decode attention, `rope_kv`, RMSNorm scale caching, the logits picks, DynamicVRAM session tiles, the device MTP loop, `auto` tree timing and the draft head pieces below. |
| `MONO_DECODE_XPU_MONO` | same as `MONO_DECODE_XPU` | Single-launch MLP block (`mono_mlp`) for decode rows that do not take the layer path. |
| `MONO_DECODE_XPU_W4` | same as `MONO_DECODE_XPU` | Fused small-M W4A8 linear for packed linears outside the layer kernels. |
| `MONO_DECODE_XPU_PREP` | same as `MONO_DECODE_XPU` | ConvRot rotate + row INT8 quantize in one launch (`rot_quant`). |
| `MONO_DECODE_XPU_GRAPH` | off | Experimental XPU graph over the verify step. Resident weights only, and not needed with `LAYER`. |

A piece set explicitly overrides the master switch, e.g. `MONO_DECODE_XPU=1 MONO_DECODE_XPU_W4=0`.

## Layer path tuning (A/B and debugging)

These default to `1` (except where noted). `0` returns that part only to the stock code. The verify numerics stay the same, except `ATTN_DPAS=0` (bit-identical attention). `AUTOTREE` and the draft switches change what is drafted, which can change the sampled text; the verify still decides every token.

| variable | part |
|---|---|
| `MONO_DECODE_XPU_GDN1` | Whole Gated-DeltaNet block in persistent launches. `0` runs `layer_pre` + Kitchen GDN calls + `layer_post`. |
| `MONO_DECODE_XPU_GDN_SPLIT` | GDN core-only kernel plus `layer_post`. `0` runs one fused launch, which spills registers and is slower. |
| `MONO_DECODE_XPU_ATTN` | One-launch tree / draft attention port for `FixedKVCache.attend`. |
| `MONO_DECODE_XPU_ATTN_DPAS` | Decode attention (tree verify rows and the draft row) with DPAS and a context split (`attn_dpas`). Not bit-identical to Kitchen (fp64-checked; see DESIGN); `0` uses the bit-identical port. |
| `MONO_DECODE_XPU_ROPE` | `rope_kv` (fused partial RoPE + KV write), with the MTP draft attention through the same path. |
| `MONO_DECODE_XPU_NORM_CACHE` | Cache `RMSNorm(add=True)` weight + 1 instead of computing it every call. |
| `MONO_DECODE_XPU_LOGITS` | Exact single-launch top-k / argmax on draft and first-token logits. |
| `MONO_DECODE_XPU_SAMPLE_TOPK` | Inside the device MTP loop's verify sampling, `torch.topk` over the verify rows runs as `topk_rows` (same values and indices in torch's order). `check` runs both and counts mismatches in `.sampling.stats()`. |
| `MONO_DECODE_XPU_PICK` | Argmax and top-8 in one pass, reused by the topk that follows. |
| `MONO_DECODE_XPU_TILED` | Tile-major W4 copies. `0` makes the kernels read row-major weights even when resident. |
| `MONO_DECODE_XPU_SESSION_TILES` | Per-generate tile-major copies in resident and DynamicVRAM modes. |
| `MONO_DECODE_XPU_TILE_MARGIN_GB` | Default `3`. Free VRAM kept beyond the actual model-copy requirement and inference reserve. |
| `MONO_DECODE_XPU_DRAFT` | Keep the eager MTP draft head's BF16 weights for the generate instead of recasting them on every draft (DynamicVRAM: one copy, about 0.5 GB, when free memory allows). |
| `MONO_DECODE_XPU_DEVICE_MTP` | Device MTP loop: the per-step bookkeeping runs on the device and the host reads each step one step late. `sync` keeps one sync per step; `shadow` also runs upstream's host walk and compares (slow); `0` runs upstream's loop. Unsupported configurations (penalties, batch > 1, other caches, a ComfyUI whose MTP loop differs from v0.45) run upstream's loop. |
| `MONO_DECODE_XPU_DRAFT_FUSED` | Inside the device MTP loop, the fused draft head (`draft_head.py`). Drafts follow the eager arithmetic in FP32 but not its reduction order; the verify decides every token. `0` runs the eager head. |
| `MONO_DECODE_XPU_GRAPH_DRAFT` | Replay the eager draft head as an XPU graph once its weights are owned (`DRAFT`), after a few eager drafts per generate. Applies outside the device loop (see `DEVICE_MTP_GRAPH`). |
| `MONO_DECODE_XPU_AUTOTREE` | Time ComfyUI's `auto` MTP tree with XPU device events (upstream uses host clocks off CUDA) and let the sampling filters run without waiting for the device. Changes which tree `auto` picks, so `auto` text differs; fixed trees are unaffected. |

## Other

| variable | effect |
|---|---|
| `MONO_DECODE_XPU_XPU_DEVICE` | AOT device for the build (default `bmg`); `install.py --target` sets it. |
| `MONO_DECODE_XPU_HOST_ROWS` | Default `1`. Enable the session INT8 host-row optimization only when ComfyUI's current VBAR watermark indicates streaming and no prefetch is active. Rechecked at each lookup; CPU source weights alone do not qualify. `0` disables this optimization, not the existing non-resident fallbacks in the draft head and device MTP loop. |
| `MONO_DECODE_XPU_STATS` | Path: the W4 linear patch writes its call counters there at exit. |
| `MONO_DECODE_XPU_GRAPH_WARMUP` | Eager steps before the experimental graph captures (default `3`). |
| `MONO_DECODE_XPU_DEVICE_MTP_GRAPH` | `1` replays the draft head graph inside the asynchronous device loop too (default off: the loop already hides the host time, no gain measured). |
| `MONO_DECODE_XPU_GRAPH_DRAFT_WARMUP` | Eager drafts per generate before the draft head is captured (default `4`). |
| `MONO_DECODE_XPU_GRAPH_DRAFT_CHECK` | Validation: rerun every replayed draft eagerly and count bitwise mismatches in `.graph_draft.stats()`. Slow. |
| `MONO_DECODE_XPU_DRAFT_INT4` | Experimental, default `0`. `1`: inside the device MTP loop, a draft's lm_head scores the vocabulary with an int4 copy of the int8 head (about 0.55 GB more per generate, when free memory allows) and recomputes the exact int8 logits of about 1000 candidates. Only drafts change. `check`: both run, the int8 one drives the loop, and mismatches are counted in `.mtp_loop.stats()`. |

## Checking what ran

Each module exposes `stats()`: `comfy_mono_decode_xpu.layer.stats()`, `.logits.stats()`,
`.mono.stats()`, `.patch.stats()`, `.autotree.stats()` (the device-timed draft / verify cost that
`auto` uses), `.draft.stats()`, `.graph_draft.stats()` and `.mtp_loop.stats()`. They report call counts and the fallback reasons.
`layer.stats()` counts layer calls (`layer`, `gdn1` for whole GDN blocks), attention and RoPE routes
(`attn`, `attn_dpas`, `rope`, `norm_rope`), the fallback reasons and the tile-copy state
(`weight_layout`, live and copied bytes, `session_active`, `disabled`). The `weights` fallback counts
the BF16 MTP block, which does not take the layer path.
