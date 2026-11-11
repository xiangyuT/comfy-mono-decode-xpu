"""ComfyUI custom node entry point (this repo cloned into ComfyUI/custom_nodes/).

Installs the decode hooks at startup and adds no nodes. With MONO_DECODE_XPU unset nothing happens.
MONO_DECODE_XPU=1 turns on the recommended stack (W4, PREP, MONO, LAYER). Each piece can be set on its
own with MONO_DECODE_XPU_<PIECE>=1/0, which also overrides the master switch.
"""
import glob
import logging
import os
import sys

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_MASTER = os.environ.get("MONO_DECODE_XPU") == "1"
_DEFAULTS = {"W4": _MASTER, "PREP": _MASTER, "MONO": _MASTER, "LAYER": _MASTER, "GRAPH": False}


def _on(piece):
    v = os.environ.get(f"MONO_DECODE_XPU_{piece}")
    return _DEFAULTS[piece] if v is None else v == "1"


def _install():
    if not any(_on(p) for p in _DEFAULTS):
        return
    repo = os.environ.get("MONO_DECODE_XPU_REPO", os.path.dirname(os.path.realpath(__file__)))
    if not glob.glob(os.path.join(repo, "comfy_mono_decode_xpu", "_C*.so")):
        logging.warning("[mono_decode_xpu] extension not built: run `python install.py` in %s. "
                        "Decode runs the stock path.", repo)
        return
    sys.path.insert(0, repo)
    try:
        from comfy_mono_decode_xpu import _C  # noqa: F401
    except ImportError as e:
        logging.warning("[mono_decode_xpu] extension failed to load (%s): rebuild with "
                        "`python install.py --force`. Decode runs the stock path.", e)
        return
    try:
        if _on("W4"):
            from comfy_mono_decode_xpu import patch
            logging.info("[mono_decode_xpu] w4a8 linear installed=%s", patch.install())
        if _on("PREP"):
            import comfy_kitchen  # noqa: F401  (loads the Kitchen XPU backend modules first)
            from comfy_mono_decode_xpu import patch
            logging.info("[mono_decode_xpu] rotate+quant installed=%s", patch.install_prep())
        if _on("MONO"):
            from comfy_mono_decode_xpu import mono
            logging.info("[mono_decode_xpu] mono_mlp installed=%s", mono.install())
        if _on("LAYER"):
            from comfy_mono_decode_xpu import layer
            logging.info("[mono_decode_xpu] layer installed=%s", layer.install())
        if _on("GRAPH"):
            from comfy_mono_decode_xpu import graph_step
            logging.info("[mono_decode_xpu] graph_step installed=%s", graph_step.install())
    except Exception:  # never take ComfyUI down: the eager path stays in place
        logging.exception("[mono_decode_xpu] install failed; decode runs the stock path")


_install()
