# Unit tests and model validation

Run in the tested ComfyUI XPU environment from the repository root:

```bash
python install.py
PYTHONPATH=/path/to/ComfyUI python -m pytest -q tests
```

Tests that need ComfyUI modules (CombinedLinear, the MTP loop install, the logits and sampling hooks)
skip when ComfyUI is not on `PYTHONPATH`. State-write replay checks supply their own ComfyUI stubs
and do not skip for a missing `comfy.ops` installation.

The unit tests use random/synthetic tensors and native operations; no checkpoint, model download,
ComfyUI server or development script is needed. They require the built extension,
Torch XPU, comfy-kitchen and omni_xpu_kernel from the tested stack. Kernel cases use the XPU visible
to the process. Eligibility and cache lifetime checks use CPU inputs. Error recovery checks use
small XPU tensors with controlled failures; they do not load a model.

Coverage includes the existing numerical comparisons for top-k, RoPE/KV, ConvRot,
W4A8 linear and both MLP layouts; eligibility checks; and three focused runtime
regressions (no replay after KV/x writes, and inference-compatible cache lifetime).
The MLP comparison also uses 16 workers in both layouts for 1/3/8 rows to exercise accumulator
growth on every device. A one-worker request checks that an excessive total SLM footprint fails
before launch; invalid worker counts still clamp to the resident-core limit.
The compact post Hadamard case compares with the full-matrix route on the same
operands for 1/3/8 rows, including the wider intermediate projection; another case
checks that only Kitchen's regular H256 takes the compact route.
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

## B70 rebased-baseline acceptance

Report unit-test success, workflow decode observations and complete target acceptance separately.
Mark a new baseline **B70 retest passed** only when every applicable gate below has evidence.
Scope that result to the tested source/native pair, software stack and memory mode; it does not
certify other platforms, memory modes or performance gains.

| gate | required evidence |
|---|---|
| Source and native identity | Record the full source commit, local changes, `.pyd` SHA-256, Python ABI, Torch build and imported extension path. Stop ComfyUI before replacing files and keep the matching source/binary backup outside `custom_nodes`. An installer exit code or successful import alone is not a numerical test. |
| Platform and device | Save `platform_probe.py --comfyui <root> --json <report>` output, driver and dependency versions, effective environment variables and server arguments. With multiple XPUs, select B70 before process startup and verify the visible device in both tests and ComfyUI. Explain every warning and resolve every FAIL. |
| Full numerical and runtime suite | Run all of `tests` with ComfyUI's Python and import path, retaining the pytest summary and JUnit XML (`--junitxml=<file>`). Require no failures; account for every skip. Missing XPU, extension or required dependencies must not skip core coverage. Include the state-write and RoPE recovery cases, not only the native arithmetic cases. |
| First and repeated workflows | Run both the official H3 T2I and the image-edit workflow with unchanged models, inputs, sampling, thinking, MTP mode, generation limits and image settings. Record the first run and at least three subsequent runs per workflow, changing only the prompt-enhance seed to force generation and conditioning to execute again. |
| Completion and output inspection | Keep full server logs through conditioning, sampling, VAE and saving, plus each enhanced prompt and image. Record human inspection of prompt completeness/coherence and image correctness. Empty captions, unfinished thinking or corrupt images fail this check even if a file was saved. Flag token-limit hits separately; a progress-bar percentage is not a completion verdict. |
| Runtime paths and recovery | Confirm expected layer/GDN/attention/RoPE execution and explain fallbacks without silent disabling. Read counters in the actual server process. After each generation, `tiles.stats()` should report `active=False` and `live_bytes=0`; cumulative copied bytes need not reset. Cancel one generation, then verify a subsequent request completes with the expected paths still enabled. |

If the probe reports an ISA FAIL despite native tests passing, retain the raw
`has_subgroup_2d_block_io` and `has_subgroup_matrix_multiply_accumulate` values and device list.
The current probe treats both an absent property (`None`) and `False` as a failure; these are
different observations. Do not assume the probe is correct, declare the GPU unsupported, or
override the failure by device name alone. Resolve and document the reporting discrepancy before
closing the platform gate; numerical-test success remains valid evidence in its own scope.

State-write, native-error and RoPE-mismatch tests intentionally exercise warning/exception paths.
The installer tests also use a fake `torch test+xpu` device and an empty `0 ops` module. Pytest's
collection of the root entry point can emit a relative-import warning outside a package context.
These messages are not failures when their assertions pass, but do not excuse the same errors
in a normal ComfyUI workflow. Verify the real extension separately before running the suite.

Attach the platform JSON, source/binary identity, pytest summary/XML, effective configuration,
full server logs, workflow definitions, per-run token counts/rates/total times, and prompt/image
inspection results. Use a per-run table with workflow, seed, first/repeat, tokens/limit, tok/s,
total seconds, completion, prompt check and image check. Record cancellation/recovery separately.
Keep private input/media outside the public repository unless publication is authorized.

Performance acceptance is separate: use fresh servers for A/B/A with matched inputs, limits and
seeds, warmups and repeated measurements. For weight-reuse A/B, keep MONO and LAYER on and vary
only WEIGHT_REUSE. Report fixed-tree and adaptive-tree results separately and check baseline drift.
Different token counts or outputs do not justify dividing workflow times into a speedup claim.

### Evidence received on 2026-11-17

Source `3498d71ca816bbe8ca5ad325a064fc306dc4be0b`, Windows B70, Python 3.13.14,
Torch 2.14.1+xpu and ComfyUI 0.45.0: the tester reported **185 passed in 10.19 s**, with no
failures or skips. T2I logged 2904/8192 tokens at 137.48 tok/s; image edit logged 4602/8196 at
118.22 tok/s. These establish suite success and observed decode execution, not full acceptance.
The reported probe still has an ISA FAIL and a multi-XPU warning. Raw capability/device details,
verified device isolation, repeated workflow completion, output inspection and cancellation/recovery
evidence remain outstanding. **Complete B70 retest acceptance is pending.**
