# Environment variables

The variables are read when ComfyUI starts, when the custom node loads. `1` means on, `0` means off.

## Pieces (read by the node's `__init__.py`)

| variable | when unset | effect |
|---|---|---|
| `MONO_DECODE_XPU` | on | Master switch. `0` disables `W4`, `PREP`, `MONO` and `LAYER` unless a piece is explicitly enabled. |
| `MONO_DECODE_XPU_LAYER` | same as `MONO_DECODE_XPU` | Decode transformer blocks through the persistent layer kernels. Also installs the attention port, `rope_kv`, the MTP draft attention path, RMSNorm scale caching, the fast logits picks and DynamicVRAM session tiles. |
| `MONO_DECODE_XPU_MONO` | same as `MONO_DECODE_XPU` | Single-launch MLP block (`mono_mlp`) for decode rows that do not take the layer path. |
| `MONO_DECODE_XPU_W4` | same as `MONO_DECODE_XPU` | Fused small-M W4A8 linear for packed linears outside the layer kernels. |
| `MONO_DECODE_XPU_PREP` | same as `MONO_DECODE_XPU` | ConvRot rotate + row INT8 quantize in one launch (`rot_quant`). |
| `MONO_DECODE_XPU_GRAPH` | off | Experimental XPU graph over the verify step. Resident weights only, and not needed with `LAYER`. |

A piece set explicitly overrides the master switch, e.g. `MONO_DECODE_XPU=1 MONO_DECODE_XPU_W4=0`.

## Layer path tuning (A/B and debugging)

These all default to `1`. Set `0` to fall back to the stock code for that part only; tokens stay the same.

| variable | part |
|---|---|
| `MONO_DECODE_XPU_GDN1` | Whole Gated-DeltaNet block in persistent launches. `0` runs `layer_pre` + Kitchen GDN calls + `layer_post`. |
| `MONO_DECODE_XPU_GDN_SPLIT` | GDN core-only kernel plus `layer_post`. `0` runs one fused launch, which spills registers and is slower. |
| `MONO_DECODE_XPU_ATTN` | One-launch tree / draft attention port for `FixedKVCache.attend`. |
| `MONO_DECODE_XPU_ATTN_DPAS` | Tree-verify attention with DPAS and a context split (`attn_dpas`). Not bit-identical to Kitchen (fp64-checked; see DESIGN); `0` uses the bit-identical port for tree rows too. |
| `MONO_DECODE_XPU_ROPE` | `rope_kv` (fused partial RoPE + KV write), with the MTP draft attention through the same path. |
| `MONO_DECODE_XPU_NORM_CACHE` | Cache `RMSNorm(add=True)` weight + 1 instead of computing it every call. |
| `MONO_DECODE_XPU_LOGITS` | Exact single-launch top-k / argmax on draft and first-token logits. |
| `MONO_DECODE_XPU_PICK` | Argmax and top-8 in one pass, reused by the topk that follows. |
| `MONO_DECODE_XPU_TILED` | Tile-major W4 copies. `0` makes the kernels read row-major weights even when resident. |
| `MONO_DECODE_XPU_SESSION_TILES` | Per-generate tile-major copies in resident and DynamicVRAM modes. |
| `MONO_DECODE_XPU_TILE_MARGIN_GB` | Default `3`. Free VRAM kept beyond the actual model-copy requirement and inference reserve. |
| `MONO_DECODE_XPU_DRAFT` | Keep the MTP draft head's BF16 weights for the generate (DynamicVRAM: one copy, about 0.5 GB, when free memory allows) instead of recasting them on every draft. The first reuse is compared bit for bit with comfy's forward. |
| `MONO_DECODE_XPU_AUTOTREE` | Time ComfyUI's auto MTP tree on the XPU device (upstream uses host clocks off CUDA) and, inside MTP generates, let the sampling filters run without waiting for the device. Unlike the rest of this table, this changes which tree `auto` picks, so `auto` text differs; fixed modes are unaffected apart from the removed wait. |

## Other

| variable | effect |
|---|---|
| `MONO_DECODE_XPU_XPU_DEVICE` | AOT device for the build (default `bmg`); `install.py --target` sets it. |
| `MONO_DECODE_XPU_STATS` | Path: the W4 linear patch writes its call counters there at exit. |
| `MONO_DECODE_XPU_GRAPH_WARMUP` | Eager steps before the experimental graph captures (default `3`). |

## Checking what ran

Each module exposes `stats()`: `comfy_mono_decode_xpu.layer.stats()`, `.logits.stats()`,
`.mono.stats()`, `.patch.stats()`, `.autotree.stats()` (the device-timed draft / verify cost that
`auto` uses) and `.draft.stats()`. They report call counts and the fallback reasons. Typical
fixed-tree output:
`{"layer": 12512, "gdn1": 9384, "attn": 4692, "rope": 4692, "fallback": {"weights": 1564}}`.
Here `weights` counts the BF16 MTP block, which runs eagerly apart from its attention.
