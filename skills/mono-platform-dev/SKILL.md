---
name: mono-platform-dev
description: Use when developing, porting, bringing up or tuning comfy-mono-decode-xpu on an Intel GPU platform other than the validated one (card, architecture, driver, OS or ComfyUI / Torch stack), or when it underperforms there.
license: Apache-2.0
metadata:
  author: xiangyuT <xiangyu.tian@intel.com>
  version: "0.1.0"
  tags: [intel-xpu, sycl, comfyui, decode, porting, benchmarking]
---

# Mono decode on another platform

Plain Markdown and scripts; any coding agent, or a person, can follow it. Run commands from the
repository root. `references/` and `scripts/` paths are relative to this file; `docs/`, `tests/` and
`install.py` are at the repository root. Read `docs/DESIGN.md` (what each kernel does, the
exactness boundary), `docs/ENV.md` (switches) and `tests/README.md` first.

## Principles

- **Exactness is the default contract.** Verify-path kernels give the same bits as the operator
  chain they replace on this platform, for the same inputs, weights, dtype and layout; the tests
  check exactly that. The documented exceptions (decode attention, held to an fp64-referenced
  error bound, and the draft path, judged by acceptance) carry their own criteria. A new exception
  is a maintainer decision with a stated acceptance criterion, a DESIGN entry and a switch back to
  the exact path; it is never a local trade for speed.
- **Establish, don't assume.** Every assumption the code makes about the device and the stack is
  checked on the target, and every claim rests on a measurement in the real workflow there.
- **Small, gated changes.** Platform-specific behaviour sits behind a device or capability gate.
  Validated platforms stay unchanged unless they are measured again.
- **One piece at a time.** The node's pieces are independent. Each must be correct and no slower
  than stock before the next.

## Prerequisites

- The ComfyUI environment the user runs: Torch with XPU, the oneAPI `icpx` compiler, Kitchen and
  ComfyUI-OmniXPU.
- The official workflow exported in API format (frontend: Export (API)), with its models present.

## Instructions

1. **Probe.**
   - Run `python skills/mono-platform-dev/scripts/platform_probe.py --comfyui <ComfyUI>` with
     ComfyUI's Python.
   - Answer the remaining questions in `references/platform.md`.
   - A FAIL means the kernels must not run here (no XPU, or a launch that could hang): keep the node
     off with `MONO_DECODE_XPU=0` and do not run the tests until the code is ported.
   - The probe does not judge capabilities such as the instruction set; step 2 establishes them.
2. **Build and test.**
   - `python install.py --force [--target <aot>]`, then
     `PYTHONPATH=<ComfyUI> python -m pytest -q tests`, in ComfyUI's own environment.
   - A failure is a correctness problem on this platform; fix it before measuring anything.
3. **Confirm what runs.**
   - Run the official workflow.
   - Check the node's log lines and every module's `stats()`: which paths ran, which fell back and
     why, and the weight layout of each generate.
   - Nothing may be silently disabled.
4. **Measure.**
   - Baseline with `scripts/bench_workflow.py`: stock against default, a fresh server each
     (`--help` lists the options).
   - Measure the roof with `scripts/read_bandwidth.py`.
   - Locate the time with the observer's device segments and profile (`MONO_TUNE_EVENTS`,
     `MONO_TUNE_PROFILE`, then `scripts/profile_summary.py`).
   - Follow `references/methodology.md`.
5. **Change**, choosing by what steps 3–4 showed:
   - configuration (memory, experimental switches, MTP mode);
   - then host launch policy;
   - then kernels or a port. See `references/development.md`.
6. **Validate and report.**
   - Acceptance as in `references/methodology.md`.
   - Report the platform (probe output, bandwidth), stack and commit, configurations, results
     for the first and later images, fallbacks, what was tried without effect, and which platforms
     a change was measured on.

## Examples

```bash
S=skills/mono-platform-dev/scripts
python $S/platform_probe.py --comfyui ~/ComfyUI
python $S/read_bandwidth.py
python $S/bench_workflow.py --comfyui ~/ComfyUI --workflow template-api.json --out bench/base \
    --config stock:MONO_DECODE_XPU=0 --config default: --config stock2:MONO_DECODE_XPU=0
cat bench/base/summary.md
```

## Available Scripts

| script | purpose | arguments |
|---|---|---|
| `scripts/platform_probe.py` | Device properties and stack versions, checked against the kernels' assumptions. | `--comfyui DIR`, `--node DIR`, `--device N`, `--json OUT` |
| `scripts/read_bandwidth.py` | Sustained device read bandwidth: the roof. | `--gib`, `--iters`, `--device` |
| `scripts/bench_workflow.py` | Replays an API-format workflow on fresh servers per configuration and summarizes decode per run. It loads the observer node `scripts/nodes/mono_tune_observer`, which writes per-generate records: tokens, steps, drafts, timings, stats and optional device segments and profile. | `--comfyui`, `--workflow`, `--out`, `--config NAME:VAR=V,...` (repeat), `--warmup`, `--runs`, `--pe-only`, `--mtp`, `--launcher`, `--server-args`, `--extra-model-paths`, `--run-timeout`; `--help` for all |
| `scripts/profile_summary.py` | Per-step device time by kernel family, and idle time, from the observer's profile. | `PROFILE.json`, `--gdn-per-step`, `--skip` |

## Troubleshooting

| symptom | where to look |
|---|---|
| The server exits before it is ready | `<out>/<config>/server.log`: the launcher, model paths or port. |
| "no observer records" in the summary | The observer did not load: a custom-node whitelist must include `mono_tune_observer`. Generates without MTP are not recorded. |
| A run ends in `timeout` | Something on the device never completed. Read the server log, then bring the pieces up one at a time (`references/development.md`). |
| The probe reports FAIL | The kernels must not run here: keep `MONO_DECODE_XPU=0` and see `references/development.md`, Porting. |

## Limitations

- Validated on one platform, the Arc Pro B70 (Xe2); everything else is to be established.
- The observer and the benchmark target ComfyUI's Qwen3.5 `generate` interface, and
  `profile_summary.py` assumes Qwen3.5-9B's layer mix. A different ComfyUI version or model needs
  them checked first.
