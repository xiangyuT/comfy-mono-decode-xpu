# SPDX-License-Identifier: Apache-2.0
"""Route Qwen3.5 decode MLP blocks (rows <= 8) to the single-launch mono_mlp kernel.

``MLP.forward(x, norm)`` computes ``down(silu(gate(norm(x))) * up(norm(x)))``. For decode rows
whose three projections are packed W4A8 (G16, FP8 group scales, ConvRot group 256), the norm
keeps its original implementation and everything after it runs in one persistent kernel that is
bit-identical to the eager kernels. Any other call (prefill, other quantization, CPU, offloaded
weights) runs the original forward.

Also holds the W4A8 eligibility checks shared with layer.py.
"""
from __future__ import annotations

import threading

import torch
import torch.nn.functional as F

from . import guard, tiles

CLAIM = "__comfy_mono_decode_xpu_mlp__"
_stats = {"mono": 0, "fallback": {}}
_lock = threading.Lock()
_consts = {}


def stats():
    with _lock:
        return {"mono": _stats["mono"], "fallback": dict(_stats["fallback"])}


def _fallback(why):
    with _lock:
        _stats["fallback"][why] = _stats["fallback"].get(why, 0) + 1


def _constants(device):
    c = _consts.get(device)
    if c is None:
        from comfy_kitchen.tensor.int8_utils import _build_hadamard
        h = _build_hadamard(256, device=device, dtype=torch.bfloat16).contiguous()
        allbf = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).to(device)
        lut = F.silu(allbf).contiguous()  # torch's own SiLU for every BF16 input
        c = _consts[device] = (h, lut)
    return c


def comfyui_has_combined_linear():
    """ComfyUI >= 0.45 fuses q/k/v and gate/up into CombinedLinear; the hooks are not adapted to it yet."""
    try:
        import comfy.ops
        return hasattr(comfy.ops, "CombinedLinear")
    except Exception:
        return False


def _is_w4a8_params(p):
    try:
        from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout
        return isinstance(p, AsymW4A8Int8Layout.Params)
    except Exception:
        return type(p).__qualname__ == "AsymW4A8Int8Layout.Params"


def _layout_ok(w):
    """True only for the packed W4A8 layout the kernels implement; anything else (other quantization
    formats, unexpected fields) is False, never an exception."""
    try:
        p = getattr(w, "_params", None)
        q = getattr(w, "_qdata", None)
        if p is None or q is None or not _is_w4a8_params(p):
            return False
        scale, cb, sc = getattr(p, "scale", None), getattr(p, "codebook", None), getattr(p, "s_channel", None)
        if scale is None or cb is None or sc is None:
            return False
        return not (getattr(p, "mma_packed", False) or getattr(p, "transposed", False)
                    or getattr(p, "group_size", None) != 16 or getattr(p, "convrot_groupsize", None) != 256
                    or getattr(p, "correction", None) is not None
                    or q.dtype != torch.int8 or q.dim() != 2 or not q.is_contiguous()
                    or scale.dtype != torch.float8_e4m3fn or not scale.is_contiguous()
                    or q.shape[0] == 0 or q.shape[1] == 0
                    or q.shape[0] % 16 or (2 * q.shape[1]) % 256
                    or scale.numel() != q.shape[0] * (2 * q.shape[1] // 16)
                    or cb.numel() != 16 or sc.numel() != q.shape[0])
    except Exception:
        return False


def _static_ok(lin):
    """Module-level contract, checked once on the module's own (possibly host-side) weight: packed W4A8,
    no bias or weight functions, and FP8 scale bytes inside the pair table (0x00-0x7F, non-negative)."""
    w = getattr(lin, "weight", None)
    key = id(getattr(w, "_qdata", None))
    cached = getattr(lin, "_mono_decode_xpu_static", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    try:
        ok = (_layout_ok(w) and getattr(lin, "bias", None) is None and not getattr(lin, "weight_function", None)
              and not getattr(lin, "bias_function", None) and int(w._params.scale.view(torch.uint8).max()) < 128)
    except Exception:
        ok = False
    lin._mono_decode_xpu_static = (key, ok)
    return ok


def _w4(lin, w):
    """Row-major kernel arguments for the device weight `w` of `lin` (resident or cast for this call),
    or None when the weight is not on the device or not the supported layout."""
    dev = getattr(w, "device", None)
    if w is None or dev is None or dev.type != "xpu" or not _static_ok(lin) or not _layout_ok(w):
        return None
    return tiles.row_major(w)


def install():
    from comfy.text_encoders import llama
    from . import _C

    if comfyui_has_combined_linear():
        import logging
        logging.warning("[mono_decode_xpu] this ComfyUI fuses projections (CombinedLinear, v0.45+); the MLP hook "
                        "is not adapted to it yet and stays off")
        return False
    MLP = llama.MLP
    original = MLP.forward
    if getattr(original, CLAIM, False):
        return False

    def forward(self, x, norm=None):
        if (guard.off("mono") or norm is None or getattr(self, "merged_mlp", False) or x.device.type != "xpu"
                or x.dtype != torch.bfloat16 or x.shape[-1] != 4096 or not 1 <= x.numel() // x.shape[-1] <= 8):
            return original(self, x, norm=norm)
        if getattr(self, "_mono_decode_xpu_skip", False):
            return original(self, x, norm=norm)
        lins = (self.gate_proj, self.up_proj, self.down_proj)
        parts = [_w4(l, getattr(l, "weight", None)) for l in lins]
        if any(p is None for p in parts):
            self._mono_decode_xpu_skip = True
            _fallback("weights")
            return original(self, x, norm=norm)
        key = tuple(p[0].data_ptr() for p in parts)
        if getattr(self, "_mono_decode_xpu_cb_ok", (None,))[0] != key:  # one shared codebook (checked once per weights)
            self._mono_decode_xpu_cb_ok = (key, torch.equal(parts[0][3], parts[1][3]) and torch.equal(parts[0][3], parts[2][3]))
        if not self._mono_decode_xpu_cb_ok[1]:
            _fallback("codebook")
            return original(self, x, norm=norm)
        t = [tiles.tiled(l, l.weight) for l in lins]
        use_tiles = all(p is not None for p in t)
        if use_tiles:
            parts = t
        try:
            h, lut = _constants(x.device)
            xn = norm(x)
            out = _C.mono_mlp(xn.reshape(-1, x.shape[-1]).contiguous(), h, lut, *parts[0], *parts[1], *parts[2],
                              0, 4, use_tiles)
        except Exception as e:  # nothing was written: the original forward recomputes from x
            guard.failed("mono", e)
            return original(self, x, norm=norm)
        with _lock:
            _stats["mono"] += 1
        return out.reshape(*x.shape[:-1], out.shape[-1])

    setattr(forward, CLAIM, True)
    forward.__wrapped__ = original
    MLP.forward = forward
    return True
