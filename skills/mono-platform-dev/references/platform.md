# What to establish about a platform

The kernels were written for and validated on Xe2 (Arc Pro B70). On any other platform, answer each
question below by measurement or by reading the code, before relying on a fast path.
The skill's `scripts/platform_probe.py` checks co-residency before anything runs and reports the
device properties; the tests and measurements establish the rest.

## Device

- **Co-residency.** The persistent kernels synchronize with a spinning grid-wide barrier, so every
  work-group of a launch must be resident at once. How does the launch planner
  (`csrc/mono_launch.hpp`) count resident work-groups, and does that match this device's topology?
  If it does not, a launch can hang instead of failing.
- **Work-group shape.** Can one work-group hold all hardware threads of a core, with the sub-group
  size the kernels require?
- **ISA.** Which matrix instructions (DPAS shapes) and block-load features exist? The kernels assume
  Xe2's. Establish this by building and running the tests on the device; reported device properties
  are not evidence.
- **SLM.** The persistent kernels use most of a core's shared local memory, and more when fewer
  cores share the work. Where is the limit on this device?
- **Memory capacity.** Per-generate weight copies need free memory beside the workflow's models.
  What is actually free at decode time, for the first image and for later ones?
- **Bandwidth.** Decode is memory-bound. What read bandwidth does this device sustain?
- **Host and runtime.** Per-step host work, CPU speed, the Level Zero adapter version and graph
  support all reach the decode loop.

## Stack

- Does the stock workflow on this platform (`MONO_DECODE_XPU=0`) run the text encoder through the
  same modules and quant formats the hooks expect? ComfyUI, Kitchen and Torch can choose different
  formats, kernels or memory paths per device, OS or version.
- Do the operators being replaced have the same reduction order here? The exactness oracle is the
  live operator chain on this platform, not results from another device.
- Do the Torch, oneAPI compiler and SYCL runtime versions match each other?
- Is the ComfyUI version one the node's version-checked hooks recognize (see `mtp_loop.py`)?
