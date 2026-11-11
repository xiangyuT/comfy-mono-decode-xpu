# SPDX-License-Identifier: Apache-2.0
"""Build the extension in place: comfy_mono_decode_xpu/_C*.so.

ComfyUI-Manager runs this after cloning the node, with ComfyUI's Python, in this directory. It can also
be run by hand (``python install.py``) or through ``install.sh``.

  python install.py [--force] [--target DEVICE]

- Skips the build when the extension is newer than every source (``--force`` rebuilds).
- Needs the oneAPI ``icpx`` compiler. If it is not on PATH, ``ONEAPI_ROOT/setvars.sh`` (default
  /opt/intel/oneapi) is sourced for the build.
- Never fails the node install: without a compiler or XPU it prints what to do and exits 0. ComfyUI
  then starts normally and decode runs the stock path.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import sysconfig

HERE = os.path.dirname(os.path.abspath(__file__))
SO = os.path.join(HERE, "comfy_mono_decode_xpu", "_C" + sysconfig.get_config_var("EXT_SUFFIX"))


def log(msg):
    print(f"[mono_decode_xpu] {msg}", flush=True)


def up_to_date():
    if not os.path.exists(SO):
        return False
    srcs = glob.glob(os.path.join(HERE, "csrc", "*")) + [os.path.join(HERE, "build_icpx.sh")]
    return os.path.getmtime(SO) >= max(os.path.getmtime(p) for p in srcs)


def build_command():
    script = os.path.join(HERE, "build_icpx.sh")
    if shutil.which("icpx"):
        return ["bash", script]
    setvars = os.path.join(os.environ.get("ONEAPI_ROOT", "/opt/intel/oneapi"), "setvars.sh")
    if os.path.exists(setvars):
        return ["bash", "-c", f'source "{setvars}" >/dev/null 2>&1; exec bash "{script}"']
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="rebuild even if the extension is up to date")
    ap.add_argument("--target", help="AOT device for icpx (default bmg)")
    args = ap.parse_args()

    try:
        import torch
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            log("torch has no usable XPU device: skipping the build (the node stays inactive).")
            return 0
        log(f"torch {torch.__version__} | {torch.xpu.get_device_name(0)}")
    except Exception as e:  # noqa: BLE001
        log(f"torch not importable ({e}): skipping the build.")
        return 0

    if up_to_date() and not args.force:
        log(f"extension up to date: {os.path.relpath(SO, HERE)}")
    else:
        cmd = build_command()
        if cmd is None:
            log("icpx not found and no oneAPI setvars.sh: install the oneAPI DPC++ compiler, then rerun "
                "`python install.py` here. Until then decode runs the stock path.")
            return 0
        env = dict(os.environ, PYTHON=sys.executable)
        if args.target:
            env["MONO_DECODE_XPU_XPU_DEVICE"] = args.target
        log(f"building for {env.get('MONO_DECODE_XPU_XPU_DEVICE', 'bmg')} (single icpx invocation, a few minutes)")
        if subprocess.call(cmd, cwd=HERE, env=env) != 0:
            log("build failed (see the compiler output above). Decode runs the stock path.")
            return 0

    sys.path.insert(0, HERE)
    try:
        from comfy_mono_decode_xpu import _C
        ops = [n for n in dir(_C) if not n.startswith("_")]
        log(f"extension ok ({len(ops)} ops). Start ComfyUI with MONO_DECODE_XPU=1.")
    except Exception as e:  # noqa: BLE001
        log(f"built extension failed to import: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
