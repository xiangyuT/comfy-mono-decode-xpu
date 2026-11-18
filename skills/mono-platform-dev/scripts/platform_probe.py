#!/usr/bin/env python3
"""Report the XPU and software stack, and check the assumptions the decode kernels make about the device.

usage: python platform_probe.py [--comfyui DIR] [--node DIR] [--device N] [--json OUT]

Run it with the Python that runs ComfyUI. It reads device properties only; it launches no kernel
and loads no model. A FAIL means the kernels must not run here: there is no XPU, or a launch could
hang, which the tests cannot safely discover. Exit code 0 means no FAIL. Capabilities whose absence
fails cleanly (instruction set, launch limits) are established by building the extension and
running the tests, not from reported properties.
"""
import argparse
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

# The checkout that contains this script (skills/<skill>/scripts/), else the script's directory.
REPO = next((p for p in Path(__file__).resolve().parents if (p / "comfy_mono_decode_xpu").is_dir()),
            Path(__file__).resolve().parent)

# The validated reference device. Anything else is unvalidated, not unsupported.
KNOWN = {
    0xE223: ("Arc Pro B70", "BMG-G31", "bmg", "validated reference device"),
}


def version(dist):
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def git_rev(path):
    try:
        out = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(path), "describe", "--always", "--dirty"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def comfyui_version(root):
    if root is None:
        return None
    f = Path(root) / "comfyui_version.py"
    if f.exists():
        for line in f.read_text().splitlines():
            if line.startswith("__version__"):
                return line.split("=", 1)[1].strip().strip("\"'")
    return git_rev(root)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--comfyui", help="ComfyUI root (for its version)")
    ap.add_argument("--node", default=str(REPO), help="comfy-mono-decode-xpu checkout (default: this repository)")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--json", help="also write the report here")
    a = ap.parse_args()
    node = Path(a.node)

    import torch

    rep = {"python": sys.version.split()[0], "os": platform.platform(), "torch": torch.__version__,
           "packages": {d: version(d) for d in ("comfy-kitchen", "comfy-aimdo", "omni_xpu_kernel",
                                                 "intel-sycl-rt", "onednn")},
           "comfyui": comfyui_version(a.comfyui), "node": {"path": str(node), "rev": git_rev(node)}}
    built = sorted(p.name for p in (node / "comfy_mono_decode_xpu").glob("_C*"))
    rep["node"]["extension"] = built or None
    checks = []

    def check(level, name, detail):
        checks.append({"level": level, "check": name, "detail": detail})

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        check("FAIL", "xpu", "torch.xpu is not available: the node stays inactive")
        return finish(rep, checks, a.json)

    p = torch.xpu.get_device_properties(a.device)
    keys = ("name", "device_id", "driver_version", "total_memory", "is_integrated_gpu", "gpu_eu_count",
            "gpu_subslice_count", "max_compute_units", "local_mem_size", "max_work_group_size",
            "sub_group_sizes", "last_level_cache_size")
    dev = {k: getattr(p, k, None) for k in keys}
    if isinstance(dev["device_id"], int):
        dev["device_id_hex"] = f"0x{dev['device_id']:04X}"
    dev["xpu_count"] = torch.xpu.device_count()
    rep["device"] = dev
    known = KNOWN.get(dev["device_id"])
    rep["known"] = dict(zip(("product", "die", "aot_target", "status"), known)) if known else None

    # 1. Persistent kernels spin on a grid barrier, so every work-group must be resident at once.
    #    This mirrors resident_wgs() in csrc/mono_launch.hpp (max_compute_units / 8 work-groups);
    #    update both together.
    cus, cores = dev["max_compute_units"], dev["gpu_subslice_count"]
    if cus and cores:
        planned = cus // 8
        if planned > cores:
            check("FAIL", "co-residency", f"planner would launch {planned} work-groups on {cores} Xe cores "
                  f"({cus // cores} EUs per core): more than can be resident, persistent kernels can deadlock")
        elif planned < cores:
            check("WARN", "co-residency", f"planner would launch {planned} work-groups on {cores} Xe cores: "
                  "safe, but some cores stay idle")
        else:
            check("OK", "co-residency", f"{cores} Xe cores, one 1024-thread work-group each")
    else:
        check("WARN", "co-residency", f"not established (max_compute_units {cus}, gpu_subslice_count "
              f"{cores}): compare the device topology with csrc/mono_launch.hpp before running the kernels")
    # 2. One work-group is all 1024 hardware threads of an Xe core, in SIMD16 sub-groups. A device
    #    without them rejects the launch with an error, which the tests report.
    if (dev["max_work_group_size"] or 0) < 1024 or 16 not in (dev["sub_group_sizes"] or []):
        check("WARN", "work-group", f"needs 1024 work-items and sub-group 16; device reports "
              f"{dev['max_work_group_size']} / {dev['sub_group_sizes']}: expect the launches to be rejected")
    # 3. Persistent work-groups use more than 64 KiB of SLM; fewer cores give each one a larger share.
    slm = dev["local_mem_size"] or 0
    check("OK" if slm >= 128 * 1024 else "WARN", "slm",
          f"{slm // 1024} KiB per work-group (the launch is rejected before start if a share exceeds it)")
    # 4. Memory: the per-generate tile-major copy of the post-block weights (~3.9 GB for Qwen3.5-9B
    #    W4A8) only happens when it fits beside the models plus MONO_DECODE_XPU_TILE_MARGIN_GB (3).
    gb = (dev["total_memory"] or 0) / 2**30
    if gb >= 24:
        check("OK", "memory", f"{gb:.1f} GiB")
    else:
        check("WARN", "memory", f"{gb:.1f} GiB: the tile-major copy may not fit beside the workflow's "
              "models; expect row-major weights and check stats.layer.weight_layout per generate")
    if dev["is_integrated_gpu"]:
        check("WARN", "integrated", "shared system memory: bandwidth and DynamicVRAM behaviour are unvalidated")
    if dev["xpu_count"] > 1:
        check("WARN", "multi-xpu", "several XPUs visible: pin the one under test (ZE_AFFINITY_MASK) "
              "and confirm nothing else runs on it")
    if known is None:
        check("WARN", "device", f"{dev.get('device_id_hex')} is not the validated device: establish "
              "every assumption (references/platform.md) before trusting a fast path")
    # 5. Build state.
    if not built:
        check("WARN", "extension", "comfy_mono_decode_xpu/_C* not built yet: build it with `python install.py`")
    check("OK" if rep["packages"]["comfy-kitchen"] else "WARN", "kitchen",
          f"comfy-kitchen {rep['packages']['comfy-kitchen']}")
    return finish(rep, checks, a.json)


def finish(rep, checks, path):
    rep["checks"] = checks
    if path:
        Path(path).write_text(json.dumps(rep, indent=1, default=str))
    dev = rep.get("device", {})
    print(f"device  : {dev.get('name')} {dev.get('device_id_hex', '')} driver {dev.get('driver_version')}")
    if rep.get("known"):
        print(f"known   : {rep['known']}")
    print(f"stack   : python {rep['python']} torch {rep['torch']} ComfyUI {rep['comfyui']} | "
          + " ".join(f"{k} {v}" for k, v in rep["packages"].items() if v))
    print(f"node    : {rep['node']['rev']} extension {rep['node']['extension']}")
    for c in checks:
        print(f"{c['level']:5s} {c['check']:13s} {c['detail']}")
    return 1 if any(c["level"] == "FAIL" for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
