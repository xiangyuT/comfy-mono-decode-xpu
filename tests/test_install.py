import importlib.util
import os
from pathlib import Path
import types

import pytest


@pytest.fixture
def installer():
    path = Path(__file__).resolve().parents[1] / "install.py"
    spec = importlib.util.spec_from_file_location("mono_installer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_sources(installer):
    root = Path(installer.HERE)
    assert len(installer.SOURCES) == len(set(installer.SOURCES))
    assert set(installer.SOURCES) == {"csrc/bindings.cpp"} | {
        path.relative_to(root).as_posix() for path in (root / "csrc").glob("*.sycl")}
    assert all((root / source).is_file() for source in installer.SOURCES)
    assert "install.SOURCES" in (root / "build_icpx.sh").read_text()


def test_installer_change_triggers_rebuild(installer, monkeypatch, tmp_path):
    (tmp_path / "csrc").mkdir()
    inputs = [tmp_path / name for name in ("csrc/kernel.sycl", "build_icpx.sh", "install.py")]
    output = tmp_path / "_C.pyd"
    for path in inputs + [output]:
        path.touch()
        os.utime(path, (100, 100))
    monkeypatch.setattr(installer, "HERE", str(tmp_path))
    monkeypatch.setattr(installer, "SO", str(output))
    assert installer.up_to_date()
    os.utime(tmp_path / "install.py", (200, 200))
    assert not installer.up_to_date()


def test_linux_build_command(installer, monkeypatch):
    monkeypatch.setattr(installer.shutil, "which", lambda name: "/tools/icpx")
    assert installer.build_command() == ["bash", os.path.join(installer.HERE, "build_icpx.sh")]


@pytest.mark.parametrize("available", [True, False])
def test_linux_setup_fallback(installer, monkeypatch, tmp_path, available):
    monkeypatch.setattr(installer.shutil, "which", lambda name: None)
    monkeypatch.setenv("ONEAPI_ROOT", str(tmp_path))
    if available:
        (tmp_path / "setvars.sh").touch()
    command = installer.build_command()
    if available:
        assert command[:2] == ["bash", "-c"]
        assert str(tmp_path / "setvars.sh") in command[2]
    else:
        assert command is None


@pytest.fixture
def python_dev(installer, tmp_path):
    root = tmp_path / "full python"
    (root / "include").mkdir(parents=True)
    (root / "libs").mkdir()
    (root / "include/Python.h").touch()
    major, minor = installer.sys.version_info[:2]
    (root / "include/patchlevel.h").write_text(
        f"#define PY_MAJOR_VERSION {major}\n#define PY_MINOR_VERSION {minor}\n")
    (root / f"libs/python{major}{minor}.lib").touch()
    return root


def test_windows_python_development(installer, python_dev):
    include, library = installer.windows_python_development(str(python_dev))
    assert Path(include) == python_dev / "include"
    assert Path(library).is_file()


@pytest.mark.parametrize("missing", ["header", "library"])
def test_missing_python_development(installer, python_dev, missing):
    if missing == "header":
        (python_dev / "include/Python.h").unlink()
    else:
        next((python_dev / "libs").glob("*.lib")).unlink()
    with pytest.raises(RuntimeError, match="--python-dev"):
        installer.windows_python_development(str(python_dev))


def test_wrong_python_headers(installer, python_dev):
    (python_dev / "include/patchlevel.h").write_text("#define PY_MAJOR_VERSION 2\n#define PY_MINOR_VERSION 7\n")
    with pytest.raises(RuntimeError, match="do not match"):
        installer.windows_python_development(str(python_dev))


def test_initialized_windows_environment(installer, monkeypatch):
    monkeypatch.setenv("LIB", "msvc-libraries")
    monkeypatch.setattr(installer, "windows_compiler", lambda env: "icx.exe")
    monkeypatch.setattr(installer.shutil, "which", lambda *args, **kwargs: "cl.exe")
    monkeypatch.setattr(installer.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not run setup"))
    assert installer.windows_environment()["LIB"] == "msvc-libraries"


def test_windows_setup_discovers_msvc(installer, monkeypatch, tmp_path):
    monkeypatch.delenv("VS2022INSTALLDIR", raising=False)
    monkeypatch.delenv("VSINSTALLDIR", raising=False)
    monkeypatch.delenv("LIB", raising=False)
    monkeypatch.setenv("ONEAPI_ROOT", str(tmp_path / "oneAPI"))
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path / "program files"))
    setup = tmp_path / "oneAPI/setvars.bat"
    setup.parent.mkdir()
    setup.touch()
    vswhere = tmp_path / "program files/Microsoft Visual Studio/Installer/vswhere.exe"
    vswhere.parent.mkdir(parents=True)
    vswhere.touch()
    monkeypatch.setattr(installer, "windows_compiler", lambda env: "icx.exe" if env.get("LIB") else None)
    monkeypatch.setattr(installer.shutil, "which", lambda name, path: "cl.exe" if path == "configured" else None)

    def discover(command, **kwargs):
        assert command[0] == str(vswhere)
        assert "Microsoft.VisualStudio.Component.VC.Tools.x86.x64" in command
        assert "[17.0,18.0)" in command
        return "C:\\VS Build Tools\n"

    def configure(command, **kwargs):
        assert isinstance(command, str)
        assert f'call "{setup}"' in command
        assert " /u " in command
        assert kwargs["encoding"] == "utf-16-le"
        assert kwargs["env"]["VS2022INSTALLDIR"] == "C:\\VS Build Tools"
        return types.SimpleNamespace(stdout="Path=configured\nLIB=libraries\nOTHER=left=right\n=C:=ignored\n")

    monkeypatch.setattr(installer.subprocess, "check_output", discover)
    monkeypatch.setattr(installer.subprocess, "run", configure)
    env = installer.windows_environment()
    assert env["PATH"] == "configured" and "Path" not in env
    assert env["OTHER"] == "left=right"
    assert "" not in env


@pytest.mark.skipif(os.name != "nt", reason="requires Windows cmd.exe")
def test_windows_environment_preserves_unicode(installer, monkeypatch, tmp_path):
    monkeypatch.delenv("LIB", raising=False)
    monkeypatch.setenv("ONEAPI_ROOT", str(tmp_path))
    monkeypatch.setenv("VS2022INSTALLDIR", "C:\\VS Build Tools")
    value = "C:\\Build\\caf\u00e9\\\u4e2d\u6587"
    monkeypatch.setenv("MONO_INSTALL_TEST_PATH", value)
    (tmp_path / "setvars.bat").write_text('@echo off\nset "LIB=test-libraries"\n', encoding="ascii")
    monkeypatch.setattr(installer, "windows_compiler", lambda env: "icx.exe" if env.get("LIB") else None)
    monkeypatch.setattr(installer.shutil, "which", lambda *args, **kwargs: "cl.exe")
    env = installer.windows_environment()
    assert env["LIB"] == "test-libraries"
    assert env["MONO_INSTALL_TEST_PATH"] == value


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("success", [True, False])
def test_windows_single_invocation_preserves_output_on_failure(installer, monkeypatch, tmp_path, python_dev, success, relative):
    root = tmp_path / "source checkout"
    for source in installer.SOURCES:
        path = root / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    torch_root = tmp_path / "torch runtime"
    (torch_root / "lib").mkdir(parents=True)
    for name in ("torch", "torch_python", "torch_cpu", "torch_xpu", "c10", "c10_xpu"):
        (torch_root / "lib" / (name + ".lib")).touch()
    output = root / "_C.pyd"
    output.write_bytes(b"old extension")
    monkeypatch.setattr(installer, "HERE", str(root))
    monkeypatch.setattr(installer, "SO", str(output))
    monkeypatch.setattr(installer, "windows_compiler", lambda env: "C:\\oneAPI compiler\\icx.exe")
    calls = []

    def compile_native(command, cwd, env):
        calls.append(command)
        assert command[0] == "C:\\oneAPI compiler\\icx.exe"
        assert "-I" + str(python_dev / "include") in command
        assert str(next((python_dev / "libs").glob("*.lib"))) in command
        for flag in ("--driver-mode=g++", "-std=c++20", "-fms-runtime-lib=dll", "-device test-target"):
            assert flag in command
        sources = command[command.index("c++") + 1:command.index("none") - 1]
        assert [Path(source) for source in sources] == [root / source for source in installer.SOURCES]
        assert env["MONO_DECODE_XPU_XPU_DEVICE"] == "test-target"
        built = Path(command[command.index("-o") + 1])
        assert built.parent == Path(cwd) and built != output
        built.write_bytes(b"new extension")
        return 0 if success else 1

    monkeypatch.setattr(installer.subprocess, "call", compile_native)
    torch = types.SimpleNamespace(__file__=str(torch_root / "__init__.py"))
    python_root = os.path.relpath(python_dev) if relative else str(python_dev)
    assert installer.build_windows(torch, {"MONO_DECODE_XPU_XPU_DEVICE": "test-target"}, python_root) == success
    assert len(calls) == 1
    assert output.read_bytes() == (b"new extension" if success else b"old extension")
    assert not list(root.glob(".mono-build-*"))


@pytest.mark.parametrize("failure", [None, "missing", "compiler"])
@pytest.mark.parametrize("development_source", ["cli", "environment"])
def test_windows_main_routes_options_and_keeps_install_nonfatal(installer, monkeypatch, failure, capsys,
                                                                development_source):
    torch = types.SimpleNamespace(__version__="test+xpu", xpu=types.SimpleNamespace(
        is_available=lambda: True, get_device_name=lambda index: "test GPU"))
    monkeypatch.setitem(installer.sys.modules, "torch", torch)
    monkeypatch.setitem(installer.sys.modules, "comfy_mono_decode_xpu",
                        types.SimpleNamespace(_C=types.SimpleNamespace(operation=lambda: None)))
    monkeypatch.setattr(installer.sys, "platform", "win32")
    arguments = ["install.py", "--force", "--target", "test-gpu"]
    monkeypatch.setenv("MONO_DECODE_XPU_PYTHON_DEV", "dev-root" if development_source == "environment" else "unused")
    if development_source == "cli":
        arguments.extend(["--python-dev", "dev-root"])
    monkeypatch.setattr(installer.sys, "argv", arguments)
    monkeypatch.setattr(installer, "up_to_date", lambda: True)
    calls = []

    def environment():
        if failure == "missing":
            raise RuntimeError("MSVC missing")
        return {}

    def build(torch_arg, env, python_dev):
        calls.append(1)
        assert torch_arg is torch and env["PYTHON"] == installer.sys.executable
        assert env["MONO_DECODE_XPU_XPU_DEVICE"] == "test-gpu" and python_dev == "dev-root"
        return failure != "compiler"

    monkeypatch.setattr(installer, "windows_environment", environment)
    monkeypatch.setattr(installer, "build_windows", build)
    assert installer.main() == 0
    assert len(calls) == int(failure != "missing")
    output = capsys.readouterr().out
    assert ("extension ok" in output) == (failure is None)
    assert ("stock path" in output) == (failure is not None)


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_stop_before_rebuild_hint_is_windows_only(installer, monkeypatch, platform, capsys):
    monkeypatch.setitem(installer.sys.modules, "torch", types.SimpleNamespace(
        __version__="test+xpu", xpu=types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda index: "GPU")))
    monkeypatch.setattr(installer.sys, "platform", platform)
    monkeypatch.setattr(installer.sys, "argv", ["install.py", "--force"])
    monkeypatch.setattr(installer, "up_to_date", lambda: False)

    def unavailable():
        raise RuntimeError("compiler unavailable")

    monkeypatch.setattr(installer, "windows_environment", unavailable)
    monkeypatch.setattr(installer, "build_command", unavailable)
    assert installer.main() == 0
    output = capsys.readouterr().out
    assert "build unavailable" in output and "stock path" in output
    assert ("Stop ComfyUI before rebuilding" in output) == (platform == "win32")


def test_up_to_date_windows_does_not_need_compiler(installer, monkeypatch):
    monkeypatch.setitem(installer.sys.modules, "torch", types.SimpleNamespace(
        __version__="test+xpu", xpu=types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda index: "GPU")))
    monkeypatch.setitem(installer.sys.modules, "comfy_mono_decode_xpu", types.SimpleNamespace(_C=types.SimpleNamespace()))
    monkeypatch.setattr(installer.sys, "argv", ["install.py"])
    monkeypatch.setattr(installer, "up_to_date", lambda: True)
    monkeypatch.setattr(installer, "windows_environment", lambda: pytest.fail("must not initialize compiler"))
    assert installer.main() == 0