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
| `MONO_DECODE_XPU_ATTN_DPAS` | Decode attention (tree verify rows and the draft row) with DPAS and a context split (`attn_dpas`). Not bit-identical to Kitchen (fp64-checked; see DESIGN); `0` uses the bit-identical port. |
| `MONO_DECODE_XPU_ROPE` | `rope_kv` (fused partial RoPE + KV write), with the MTP draft attention through the same path. |
| `MONO_DECODE_XPU_NORM_CACHE` | Cache `RMSNorm(add=True)` weight + 1 instead of computing it every call. |
| `MONO_DECODE_XPU_LOGITS` | Exact single-launch top-k / argmax on draft and first-token logits. |
| `MONO_DECODE_XPU_SAMPLE_TOPK` | Inside the device MTP loop's verify sampling, `torch.topk(logits, top_k)` (8 x 248k FP32, k <= 64) runs as `topk_rows`: the same values and indices in torch's order, ties included. `check` runs both and counts mismatches in `.sampling.stats()`. |
| `MONO_DECODE_XPU_PICK` | Argmax and top-8 in one pass, reused by the topk that follows. |
| `MONO_DECODE_XPU_TILED` | Tile-major W4 copies. `0` makes the kernels read row-major weights even when resident. |
| `MONO_DECODE_XPU_SESSION_TILES` | Per-generate tile-major copies in resident and DynamicVRAM modes. |
| `MONO_DECODE_XPU_TILE_MARGIN_GB` | Default `3`. Free VRAM kept beyond the actual model-copy requirement and inference reserve. |
| `MONO_DECODE_XPU_DRAFT` | Keep the MTP draft head's BF16 weights for the generate (DynamicVRAM: one copy, about 0.5 GB, when free memory allows) instead of recasting them on every draft. The first reuse is compared bit for bit with comfy's forward. |
| `MONO_DECODE_XPU_DEVICE_MTP` | Qwen3.5 MTP decode with the per-step bookkeeping (accept walk, cache counters, deferred-delta ctl, KV slot moves, next token / hidden row) on the device and the host reading each step's tokens one step late, so a step's sync no longer idles the device. `sync` keeps one sync per step; `shadow` also recomputes upstream's host walk and compares (slow); `0` runs upstream's loop. Unsupported shapes (penalties, batch > 1, other caches) always run upstream's loop, and so does a ComfyUI whose `_generate_mtp` or KV / tree code differs from v0.45 (`.mtp_loop.stats()` fallback `upstream_changed`); a failure inside the loop switches it off and reruns the generate with upstream's loop. With a fixed tree, or in `sync` / `shadow` mode, the tokens are upstream's on the same values; asynchronously, an `auto` tree change takes effect one step later than in upstream's loop (each step is read one step late). |
| `MONO_DECODE_XPU_DRAFT_FUSED` | Inside the device MTP loop, the draft head runs with fused small kernels (`draft_head.py`): one launch each for the embedding lookup + pre-fc norms + cat, the gate, add + norm (twice), silu·mul and the final norm + the lm_head's rotation and quantization; q/k/v and gate/up as one GEMM each on owned BF16 weights; the top-n pick and its outputs in one launch. Drafts follow the eager arithmetic in FP32 but not its reduction order (only drafts change; the verify decides every token). `0` runs the eager head. |
| `MONO_DECODE_XPU_GRAPH_DRAFT` | Replay the MTP draft head as an XPU graph once its weights are owned (`DRAFT`), after a few eager drafts per generate: the same kernels on the same inputs, without the eager launch path on the host. Off when `DRAFT` is off. |
| `MONO_DECODE_XPU_AUTOTREE` | Time ComfyUI's auto MTP tree on the XPU device (upstream uses host clocks off CUDA) and, inside MTP generates, let the sampling filters run without waiting for the device. Unlike the rest of this table, this changes which tree `auto` picks, so `auto` text differs; fixed modes are unaffected apart from the removed wait. |

## Other

| variable | effect |
|---|---|
| `MONO_DECODE_XPU_XPU_DEVICE` | AOT device for the build (default `bmg`); `install.py --target` sets it. |
| `MONO_DECODE_XPU_STATS` | Path: the W4 linear patch writes its call counters there at exit. |
| `MONO_DECODE_XPU_GRAPH_WARMUP` | Eager steps before the experimental graph captures (default `3`). |
| `MONO_DECODE_XPU_DEVICE_MTP_GRAPH` | `1` replays the draft head graph inside the asynchronous device loop too (default off: the loop already hides the host time, no gain measured). |
| `MONO_DECODE_XPU_GRAPH_DRAFT_WARMUP` | Eager drafts per generate before the draft head is captured (default `4`). |
| `MONO_DECODE_XPU_GRAPH_DRAFT_CHECK` | Validation: rerun every replayed draft eagerly and count bitwise mismatches in `.graph_draft.stats()`. Slow. |
| `MONO_DECODE_XPU_DRAFT_INT4` | Experimental, default `0`. `1`: inside the device MTP loop, a draft's lm_head scores the vocabulary with an int4 copy of the int8 head (oneDNN weight-only GEMM, about 0.55 GB extra per generate when free memory allows) and recomputes the exact int8 logits of about 1000 candidates (`draft_pick`), so its top-7 / argmax equal the int8 draft's whenever the exact top-7 lies among the candidates. Only drafts change; the verify, its sampling and so the output distribution do not. `check`: both paths run, the int8 one drives the loop and mismatches are counted in `.mtp_loop.stats()` (`draft_int4_mismatch`). |

## Checking what ran

Each module exposes `stats()`: `comfy_mono_decode_xpu.layer.stats()`, `.logits.stats()`,
`.mono.stats()`, `.patch.stats()`, `.autotree.stats()` (the device-timed draft / verify cost that
`auto` uses), `.draft.stats()`, `.graph_draft.stats()` and `.mtp_loop.stats()`. They report call counts and the fallback reasons. Typical
fixed-tree output:
`{"layer": 12512, "gdn1": 9384, "attn": 4692, "rope": 4692, "fallback": {"weights": 1564}}`.
Here `weights` counts the BF16 MTP block, which runs eagerly apart from its attention.
