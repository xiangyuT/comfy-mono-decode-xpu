# comfy-mono-decode-xpu

Intel XPU (Xe2 / BMG) decode kernels for ComfyUI prompt enhance (`TextGenerate`) with
Qwen3.5-9B W4A8 text encoders. Kernels hook the model-level decode step, so official
templates benefit without new nodes. Every routed op is bit-identical to the eager path.

Status: mono v1 — the decode MLP block (ConvRot rotate+quant → gate/up W4A8 → SiLU·mul →
rotate+quant → down W4A8) runs as one persistent launch with device-wide grid barriers.

## Pieces
- `csrc/w4a8_decode.sycl` — fused small-M W4A8 linear (SLM codebook, DPAS for M=8).
- `csrc/act_prep.sycl` — fused ConvRot rotation + row INT8 quantization.
- `csrc/mono_mlp.sycl` — single-launch MLP block.
- `comfy_mono_decode_xpu/{patch,mono,graph_step}.py` — env-gated routing.
- `comfyui_node/` — custom node; link into `ComfyUI/custom_nodes/zz_cmdx`.
  `CMDX_ENABLE=1` (W4 linear), `CMDX_PREP=1` (rotate+quant), `CMDX_MONO=1` (MLP block),
  `CMDX_GRAPH=1` (XPU graph over the verify step).
- `tests/` — exactness / micro benchmarks (paths assume the dev container mounts).

## Build
```
TORCH_XPU_ARCH_LIST=bmg python setup.py build_ext --inplace   # rm -rf build/ when flags change
```

## Limits
Resident weights only (`--gpu-only --disable-dynamic-vram`); DynamicVRAM/aimdo later.
B70 fixed-tree (tree4s1m8) decode: image eager ~29 tok/s → W4+prep+mono ~62–70 tok/s,
identical token sequences.
