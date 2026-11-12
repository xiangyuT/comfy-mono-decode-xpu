# Unit tests and model validation

Run in the tested ComfyUI XPU environment from the repository root:

```bash
python install.py
PYTHONPATH=/path/to/ComfyUI python -m pytest -q tests
```

The two CombinedLinear tests import `comfy.ops` and are skipped when ComfyUI is not on `PYTHONPATH`.

The unit tests use random/synthetic tensors and native operations; no checkpoint, model download,
ComfyUI server or development script is needed. They require the built extension,
Torch XPU, comfy-kitchen and omni_xpu_kernel from the tested stack. Kernel cases use the XPU visible
to the process. Eligibility and cache lifetime checks use CPU inputs. Error recovery checks use
small XPU tensors with controlled failures; they do not load a model.

Coverage includes the existing numerical comparisons for top-k, RoPE/KV, ConvRot,
W4A8 linear and both MLP layouts; eligibility checks; and three focused runtime
regressions (no replay after KV/x writes, and inference-compatible cache lifetime).
On ComfyUI v0.45, two additional cases exercise real combined storage and its residency window,
then compare the MLP, normalization and projections with the actual combined input-act entry for
1/3/8 rows. They skip on older ComfyUI versions without that interface.

## Model checks through the ComfyUI UI

Use existing model files and load the official template through the ComfyUI frontend. Keep its
models, resolution, steps, CFG, sampler and prompt enhancement. For v0.45 H3 Image, use
`image_minimax_h3_image_t2i`: its PE is Qwen3.5-9B W4A8, its separate conditioning encoder is
Qwen3VL-32B INT8 ConvRot, and its defaults include thinking, auto MTP, 4MP and six sampling steps.

1. Queue through the frontend Run button. Confirm actual layer/GDN/attention/RoPE execution and
   no disabled paths. Check the expanded prompt is complete, coherent and consistent with the
   requested subjects, scene and text. An empty caption or unfinished thinking is a failed prompt
   check even if SaveImage produced a file. Inspect the final image as well as its media format.
2. Change only the text-generation sampling seed to force another expansion. Confirm the text
   and conditioning nodes re-execute, stay on XPU, and complete the diffusion/VAE handoff.
3. Use the memory mode being qualified. The default is DynamicVRAM; ordinary memory management
   uses `--disable-dynamic-vram`. ComfyUI's optional `--async-offload 2 --force-non-blocking` path
   exercises prepared CombinedLinear weights on XPU. Record it separately from the default lane
   and verify the `mono.stats()["combined"]` counter when groups are actually prepared.

Kernel numerical checks hold inputs, weights, dtype and layout fixed. Whole generated-token
equality and pixel equality are not the v0.45 workflow acceptance criteria. The earlier fixed-tree
MOSS comparison in the README retains its original v0.43 measurement and equality checks.

The component's `layer.stats()` reports disabled paths, layout/copy budget, cumulative copied bytes,
current live tile bytes and active-session state. A completed generation should have no live tiles
or active session. Accumulated copied bytes count earlier generations too and are not a leak gauge.

Kernel exceptions from a stateful dispatch must fail the current request; they must not silently
retry a partially updated layer. Pure-operation failures may disable only that path and use the
stock operation. A user cancellation must not permanently disable a path.

Model-running scripts and checkpoint fixtures are intentionally outside this repository. UI
lifecycle checks establish functionality; they do not by themselves establish a speedup, image
quality acceptance or clean-image validation.
