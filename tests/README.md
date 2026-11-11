# Unit tests and model validation

Run in the tested ComfyUI XPU environment from the repository root:

```bash
python install.py
python -m pytest -q tests
```

The unit tests use random/synthetic tensors and native operations; no checkpoint, model download,
ComfyUI server, developer mount or `/artifacts` script is needed. They require the built extension,
Torch XPU, comfy-kitchen and omni_xpu_kernel from the tested stack. Kernel cases use the XPU visible
to the process. Eligibility and cache lifetime checks use CPU inputs. Error recovery checks use
small XPU tensors with controlled failures; they do not load a model.

Coverage includes the existing numerical comparisons for top-k, RoPE/KV, ConvRot,
W4A8 linear and both MLP layouts; eligibility checks; and three focused runtime
regressions (no replay after KV/x writes, and inference-compatible cache lifetime).

## Model checks through the ComfyUI UI

Use the existing model files and load the official `image_moss_image_t2i` template from ComfyUI's
template browser. Keep its models, resolution, steps, CFG and sampler settings. The README showcase
fixes KSampler seed to 42 with its post-generation seed control set to fixed. Prompt enhancement
must remain enabled.

1. Start the tested ComfyUI stack with mono off. In the UI set the TextGenerate MTP selector inside
   the subgraph to `tree4s1m8`, queue the workflow, and retain the enhanced text shown by PreviewAny
   and the saved image.
2. Restart with `MONO_DECODE_XPU=1`, load the same graph and seed, and queue it from the UI. Confirm
   the install logs, enhanced text equality and successful image output. Images have their own
   numerical variation and are not required to be pixel-identical.
3. Queue another forced expansion by changing only the TextGenerate sampling seed, as in the
   README's warm-run scheme. Confirm TextGenerate and conditioning execute again and the workflow
   completes the text-encoder to diffusion/VAE handoff.
4. Repeat the candidate lifecycle in default DynamicVRAM and resident mode
   (`--gpu-only --disable-dynamic-vram`). Also run the official `auto` MTP setting. Auto may choose a
   different speculative tree and consume a seeded sampling stream differently; use the fixed tree
   for equal-output comparisons.

The component's `layer.stats()` reports disabled paths, layout/copy budget, cumulative copied bytes,
current live tile bytes and active-session state. A completed generation should have no live tiles
or active session. Accumulated copied bytes count earlier generations too and are not a leak gauge.

Kernel exceptions from a stateful dispatch must fail the current request; they must not silently
retry a partially updated layer. Pure-operation failures may disable only that path and use the
stock operation. A user cancellation must not permanently disable a path.

Model-running scripts and checkpoint fixtures are intentionally outside this repository. UI
lifecycle checks establish functionality; they do not by themselves establish a speedup, image
quality acceptance or clean-image validation.
