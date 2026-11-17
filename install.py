# SPDX-License-Identifier: Apache-2.0
"""Build the native extension in place (.pyd on Windows, .so on Linux).

ComfyUI-Manager runs this after cloning the node, with ComfyUI's Python, in this directory. It can also
be run by hand (``python install.py``) or through ``install.sh``.

    python install.py [--force] [--target DEVICE] [--python-dev DIRECTORY]

- Skips the build when the extension is newer than every source (``--force`` rebuilds).
- Needs the oneAPI ``icpx`` compiler. If it is not on PATH, ``ONEAPI_ROOT/setvars.sh`` (default
  /opt/intel/oneapi) is sourced for the build.
- Windows uses ``icx.exe`` and MSVC 2022, initializing ``ONEAPI_ROOT/setvars.bat`` when needed.
    Embedded Python can use matching development files via ``--python-dev`` or
    ``MONO_DECODE_XPU_PYTHON_DEV`` (the full Python installation directory).
- Never fails the node install: without a compiler or XPU it prints what to do and exits 0. ComfyUI
  then starts normally and decode runs the stock path.
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SO = os.path.join(HERE, "comfy_mono_decode_xpu", "_C" + sysconfig.get_config_var("EXT_SUFFIX"))
SOURCES = (
    "csrc/bindings.cpp", "csrc/w4a8_decode.sycl", "csrc/act_prep.sycl", "csrc/mono_mlp.sycl",
    "csrc/mono_layer.sycl", "csrc/mono_gdn.sycl", "csrc/attn_port.sycl", "csrc/attn_dpas.sycl",
    "csrc/fast_topk.sycl", "csrc/rope_kv.sycl", "csrc/mtp_loop.sycl", "csrc/draft_lmhead.sycl",
    "csrc/topk_rows.sycl", "csrc/draft_fused.sycl",
)


def log(msg):
    print(f"[mono_decode_xpu] {msg}", flush=True)


def up_to_date():
    if not os.path.exists(SO):
        return False
    srcs = glob.glob(os.path.join(HERE, "csrc", "*")) + [
        os.path.join(HERE, "build_icpx.sh"), os.path.join(HERE, "install.py")]
    return os.path.getmtime(SO) >= max(os.path.getmtime(p) for p in srcs)


def build_command():
    script = os.path.join(HERE, "build_icpx.sh")
    if shutil.which("icpx"):
        return ["bash", script]
    setvars = os.path.join(os.environ.get("ONEAPI_ROOT", "/opt/intel/oneapi"), "setvars.sh")
    if os.path.exists(setvars):
        return ["bash", "-c", f'source "{setvars}" >/dev/null 2>&1; exec bash "{script}"']
    return None


def windows_compiler(env):
    compiler = shutil.which("icx.exe", path=env.get("PATH", ""))
    if compiler:
        return compiler
    root = env.get("ONEAPI_ROOT", os.path.join(env.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
                                               "Intel", "oneAPI"))
    compiler = os.path.join(root, "compiler", "latest", "bin", "icx.exe")
    return compiler if os.path.isfile(compiler) else None


def windows_environment():
    env = {key.upper(): value for key, value in os.environ.items()}
    if (windows_compiler(env) and shutil.which("cl.exe", path=env.get("PATH", ""))
            and env.get("LIB")):
        return env
    program_files = env.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
    root = env.get("ONEAPI_ROOT", os.path.join(program_files, "Intel", "oneAPI"))
    setvars = os.path.join(root, "setvars.bat")
    if not os.path.isfile(setvars):
        raise RuntimeError("oneAPI setvars.bat not found; install Intel oneAPI DPC++/C++ or set ONEAPI_ROOT")
    visual_studio = env.get("VS2022INSTALLDIR") or env.get("VSINSTALLDIR")
    if not visual_studio:
        vswhere = os.path.join(program_files, "Microsoft Visual Studio", "Installer", "vswhere.exe")
        if os.path.isfile(vswhere):
            visual_studio = subprocess.check_output([
                vswhere, "-latest", "-products", "*", "-version", "[17.0,18.0)",
                "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property", "installationPath"], text=True).strip()
    if not visual_studio:
        raise RuntimeError("MSVC 2022 not found; install Build Tools with Desktop development with C++ "
                           "and Windows SDK, or set VS2022INSTALLDIR")
    env["VS2022INSTALLDIR"] = visual_studio
    command = ('"' + env.get("COMSPEC", "cmd.exe") + '" /d /u /s /c "call "' + setvars
               + '" >nul 2>nul & set"')
    configured = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-16-le", check=True)
    env = {key.upper(): value for line in configured.stdout.splitlines()
           for key, separator, value in [line.partition("=")] if separator and key}
    if not (windows_compiler(env) and shutil.which("cl.exe", path=env.get("PATH", "")) and env.get("LIB")):
        raise RuntimeError("oneAPI/MSVC initialization incomplete; retry from an x64 Native Tools "
                           "Command Prompt for VS 2022 after calling oneAPI setvars.bat")
    return env


def windows_python_development(root=None):
    if root:
        root = os.path.abspath(root)
    include = os.path.join(root, "include") if root else sysconfig.get_paths()["include"]
    abi = ("t" if sysconfig.get_config_var("Py_GIL_DISABLED") else "")
    abi += "_d" if sysconfig.get_config_var("Py_DEBUG") else ""
    library = os.path.join(root or sys.base_prefix, "libs",
                           f"python{sys.version_info.major}{sys.version_info.minor}{abi}.lib")
    for required in (os.path.join(include, "Python.h"), os.path.join(include, "patchlevel.h"), library):
        if not os.path.isfile(required):
            raise RuntimeError(f"Python development file missing: {required}. Use --python-dev DIRECTORY "
                               "or MONO_DECODE_XPU_PYTHON_DEV for a matching full Python installation; "
                               "keep running install.py with ComfyUI's Python")
    with open(os.path.join(include, "patchlevel.h"), encoding="utf-8") as header:
        versions = dict(re.findall(r"^#define\s+PY_(MAJOR|MINOR)_VERSION\s+(\d+)", header.read(), re.M))
    if (versions.get("MAJOR"), versions.get("MINOR")) != tuple(map(str, sys.version_info[:2])):
        raise RuntimeError("Python development headers do not match the running Python major/minor version")
    return include, library


def build_windows(torch, env, python_dev=None):
    compiler = windows_compiler(env)
    if not compiler:
        raise RuntimeError("icx.exe not found; install Intel oneAPI DPC++/C++")
    include, python_library = windows_python_development(python_dev)
    torch_root = os.path.dirname(torch.__file__)
    sources = [os.path.join(HERE, source) for source in SOURCES]
    libraries = [os.path.join(torch_root, "lib", name + ".lib") for name in
                 ("torch", "torch_python", "torch_cpu", "torch_xpu", "c10", "c10_xpu")]
    libraries.append(python_library)
    for required in sources + libraries:
        if not os.path.isfile(required):
            raise RuntimeError(f"build input missing: {required}")
    target = env.get("MONO_DECODE_XPU_XPU_DEVICE", "bmg")
    with tempfile.TemporaryDirectory(prefix=".mono-build-", dir=HERE) as work:
        output = os.path.join(work, os.path.basename(SO))
        command = [compiler, "--driver-mode=g++", "-fsycl", "-std=c++20", "-fms-runtime-lib=dll",
                   "-O3", "-DNDEBUG", "-DNOMINMAX", "-shared", "-fsycl-targets=spir64_gen",
                   "-Xsycl-target-backend", f"-device {target}",
                   "-I" + os.path.join(torch_root, "include"),
                   "-I" + os.path.join(torch_root, "include", "torch", "csrc", "api", "include"),
                   "-I" + include, "-I" + os.path.join(HERE, "csrc"),
                   "-DTORCH_EXTENSION_NAME=_C", "-DTORCH_API_INCLUDE_EXTENSION_H", "-x", "c++",
                   *sources, "-x", "none", "-o", output, *libraries]
        if subprocess.call(command, cwd=work, env=env) != 0:
            return False
        os.replace(output, SO)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="rebuild even if the extension is up to date")
    ap.add_argument("--target", help="AOT device (default bmg)")
    ap.add_argument("--python-dev", default=os.environ.get("MONO_DECODE_XPU_PYTHON_DEV"),
                    help="Windows full Python directory containing matching include/ and libs/ files")
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
        try:
            windows = sys.platform == "win32"
            env = windows_environment() if windows else dict(os.environ)
            env["PYTHON"] = sys.executable
            if args.target:
                env["MONO_DECODE_XPU_XPU_DEVICE"] = args.target
            log(f"building for {env.get('MONO_DECODE_XPU_XPU_DEVICE', 'bmg')} (single compiler invocation)")
            if windows:
                success = build_windows(torch, env, args.python_dev)
            else:
                cmd = build_command()
                if cmd is None:
                    raise RuntimeError("icpx not found and no oneAPI setvars.sh; install the oneAPI "
                                       "DPC++ compiler, then rerun `python install.py`")
                success = subprocess.call(cmd, cwd=HERE, env=env) == 0
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            hint = " Stop ComfyUI before rebuilding." if windows else ""
            log(f"build unavailable ({error}).{hint} Decode runs the stock path.")
            return 0
        if not success:
            log("build failed (see the compiler output above). Decode runs the stock path.")
            return 0

    sys.path.insert(0, HERE)
    try:
        from comfy_mono_decode_xpu import _C
        ops = [n for n in dir(_C) if not n.startswith("_")]
        log(f"extension ok ({len(ops)} ops). Restart ComfyUI; it is enabled by default (MONO_DECODE_XPU=0 disables it).")
    except Exception as e:  # noqa: BLE001
        log(f"built extension failed to import: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
