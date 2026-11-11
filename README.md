# comfy-mono-decode-xpu

Intel XPU (Xe2 / BMG) decode kernels for ComfyUI prompt enhance (`TextGenerate`) with
Qwen3.5-9B W4A8 text encoders. Kernels hook the model-level decode step, so official
templates benefit without new nodes. Every routed op is bit-identical to the eager path.

Status (dev/mono-layer-20261111): every decode verify layer runs in persistent launches with
device-wide grid barriers, bit-identical to the eager kernels:
- Gated-DeltaNet layers: the whole block (norms, ConvRot quant, W4A8 projections, Kitchen deferred
  conv / gates / delta ports, MLP, residuals) in ONE launch (`mono_gdn.sycl`).
- Full-attention layers: `layer_pre` + q/k norm/RoPE (eager) + attention port + `layer_post`.
- Attention (tree verify and the single MTP draft row) is a one-launch port of Kitchen's
  decode_gqa + combine + tree_merge (`attn_port.sycl`).

## Pieces
- `csrc/w4a8_decode.sycl` — fused small-M W4A8 linear (SLM codebook, DPAS for M=8).
- `csrc/act_prep.sycl` — fused ConvRot rotation + row INT8 quantization.
- `csrc/mono_mlp.sycl` — single-launch MLP block; `csrc/mono_common.hpp` — shared engine
  (tile-major W4 stream GEMV with an exact FP8-scale pair table), RMSNorm port, rot_quant.
- `csrc/mono_layer.sycl` (pre/post), `csrc/mono_gdn.sycl` (GDN layer), `csrc/attn_port.sycl`.
- `comfy_mono_decode_xpu/{patch,mono,graph_step}.py` — env-gated routing.
- `comfyui_node/` — custom node; link into `ComfyUI/custom_nodes/zz_cmdx`.
  `CMDX_ENABLE=1` (W4 linear), `CMDX_PREP=1` (rotate+quant), `CMDX_MONO=1` (MLP block),
  `CMDX_LAYER=1` (whole decode layers + attention), `CMDX_GRAPH=1` (XPU graph over the verify step).
- `tests/` — exactness / micro benchmarks (paths assume the dev container mounts).

## Build
```
./build_icpx.sh        # one icpx invocation (compile + device link), like omni_xpu_kernel
```
Do not use `setup.py build_ext` for the exact kernels: torch's SyclExtension compiles per file and
device-links with `-fsycl-link`, which produced different device code for identical source.

## Limits
Resident weights only (`--gpu-only --disable-dynamic-vram`); DynamicVRAM/aimdo later.
B70, resident weights: fixed tree (tree4s1m8) decode 79/91/87 tok/s with identical token
sequences (image eager 28/30/30); official `auto` MTP 86/89/95 tok/s (image eager 30/27/30).
The persistent kernels read a tile-major copy of each W4 weight (~5 GB extra VRAM).
