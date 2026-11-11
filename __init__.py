# SPDX-License-Identifier: Apache-2.0
"""ComfyUI custom node entry point (this repo cloned into ComfyUI/custom_nodes/).

Installs the decode hooks at startup and adds no nodes. With MONO_DECODE_XPU unset nothing happens.
MONO_DECODE_XPU=1 turns on the recommended stack (W4, PREP, MONO, LAYER). Each piece can be set on its
own with MONO_DECODE_XPU_<PIECE>=1/0, which also overrides the master switch. Each piece is installed
on its own: one that fails to install is logged and the others still load.
"""
import glob
import logging
import os

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_MASTER = os.environ.get("MONO_DECODE_XPU") == "1"
_DEFAULTS = {"W4": _MASTER, "PREP": _MASTER, "MONO": _MASTER, "LAYER": _MASTER, "GRAPH": False}


def _on(piece):
    v = os.environ.get(f"MONO_DECODE_XPU_{piece}")
    return _DEFAULTS[piece] if v is None else v == "1"


def _install_w4():
    from .comfy_mono_decode_xpu import patch
    return patch.install()


def _install_prep():
    import comfy_kitchen  # noqa: F401  (loads the Kitchen XPU backend modules first)
    from .comfy_mono_decode_xpu import patch
    return patch.install_prep()


def _install_mono():
    from .comfy_mono_decode_xpu import mono
    return mono.install()


def _install_layer():
    from .comfy_mono_decode_xpu import layer
    return layer.install()


def _install_graph():
    from .comfy_mono_decode_xpu import graph_step
    return graph_step.install()


_PIECES = (("W4", "w4a8 linear", _install_w4), ("PREP", "rotate+quant", _install_prep),
           ("MONO", "mono_mlp", _install_mono), ("LAYER", "layer", _install_layer),
           ("GRAPH", "graph_step", _install_graph))


def _install():
    if not any(_on(p) for p in _DEFAULTS):
        return
    here = os.path.dirname(os.path.realpath(__file__))
    if not glob.glob(os.path.join(here, "comfy_mono_decode_xpu", "_C*.so")):
        logging.warning("[mono_decode_xpu] extension not built: run `python install.py` in %s. "
                        "Decode runs the stock path.", here)
        return
    try:
        from .comfy_mono_decode_xpu import _C  # noqa: F401
    except ImportError as e:
        logging.warning("[mono_decode_xpu] extension failed to load (%s): rebuild with "
                        "`python install.py --force`. Decode runs the stock path.", e)
        return
    for piece, label, fn in _PIECES:
        if not _on(piece):
            continue
        try:
            logging.info("[mono_decode_xpu] %s installed=%s", label, fn())
        except Exception:  # never take ComfyUI down: that piece keeps the stock path
            logging.exception("[mono_decode_xpu] %s failed to install; it stays on the stock path", label)


_install()
