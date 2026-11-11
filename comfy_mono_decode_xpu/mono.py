# SPDX-License-Identifier: Apache-2.0
"""Route Qwen3.5 decode MLP blocks (rows <= 8) to the single-launch mono_decode_xpu.mono_mlp kernel.

``MLP.forward(x, norm)`` computes ``down(silu(gate(norm(x))) * up(norm(x)))``. For decode rows
whose three projections are packed W4A8 (G16, FP8 group scales, ConvRot group 256), the norm
keeps its original implementation and everything after it runs in one persistent kernel that is
bit-identical to the eager kernels. Any other call (prefill, other quantization, CPU, offloaded
weights) runs the original forward.
"""
from __future__ import annotations

import threading

import torch
import torch.nn.functional as F

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


def _layout_ok(w):
    p = getattr(w, "_params", None)
    q = getattr(w, "_qdata", None)
    if p is None or q is None:
        return False
    return not (getattr(p, "mma_packed", False) or getattr(p, "transposed", False) or p.group_size != 16
                or p.convrot_groupsize != 256 or p.correction is not None or p.codebook is None
                or q.dtype != torch.int8 or q.dim() != 2 or not q.is_contiguous()
                or p.scale.dtype != torch.float8_e4m3fn or not p.scale.is_contiguous()
                or q.shape[0] % 16 or (2 * q.shape[1]) % 256)


def _static_ok(lin):
    """Module-level contract, checked once on the module's own (possibly host-side) weight: no bias or
    weight functions, and FP8 scale bytes inside the pair table (0x00-0x7F, non-negative)."""
    cached = getattr(lin, "_mono_decode_xpu_static", None)
    w = getattr(lin, "weight", None)
    key = id(getattr(w, "_qdata", None))
    if cached is not None and cached[0] == key:
        return cached[1]
    ok = (_layout_ok(w) and getattr(lin, "bias", None) is None and not getattr(lin, "weight_function", None)
          and not getattr(lin, "bias_function", None) and int(w._params.scale.view(torch.uint8).max()) < 128)
    lin._mono_decode_xpu_static = (key, ok)
    return ok


def _w4(lin, w=None):
    """Kernel arguments for a packed W4A8 linear. Resident weights (w None): a cached tile-major copy.
    A DynamicVRAM weight cast for this call (w given): the checkpoint's row-major tensors, no copy."""
    if w is not None:
        if not _static_ok(lin) or w.device.type != "xpu" or not _layout_ok(w):
            return None
        p = w._params
        return [w._qdata, p.scale, p.s_channel.float().contiguous(), p.codebook.float().contiguous()]
    w = getattr(lin, "weight", None)
    if w is None or w.device.type != "xpu" or not _static_ok(lin):
        return None
    q, p = w._qdata, w._params
    cache = getattr(lin, "_mono_decode_xpu_w4", None)
    if cache is None or cache[0] is not q:
        parts = [tile_major(q, 16), tile_major(p.scale.view(torch.uint8), 16).view(torch.float8_e4m3fn),
                 p.s_channel.float().contiguous(), p.codebook.float().contiguous()]
        cache = (q, parts)
        lin._mono_decode_xpu_w4 = cache
    return cache[1]


def tile_major(t, block):
    """[N, B] bytes -> [N/16][B/block][16][block] (the persistent kernels' W4 layout), kept as [N, B]."""
    n, b = t.shape
    return t.view(n // 16, 16, b // block, block).permute(0, 2, 1, 3).contiguous().view(n, b)


def install():
    from comfy.text_encoders import llama
    from . import _C

    MLP = llama.MLP
    original = MLP.forward
    if getattr(original, CLAIM, False):
        return False

    def forward(self, x, norm=None):
        if (norm is None or self.merged_mlp or x.device.type != "xpu" or x.dtype != torch.bfloat16
                or x.shape[-1] != 4096 or not 1 <= x.numel() // x.shape[-1] <= 8):
            return original(self, x, norm=norm)
        if getattr(self, "_mono_decode_xpu_skip", False):
            return original(self, x, norm=norm)
        parts = [_w4(self.gate_proj), _w4(self.up_proj), _w4(self.down_proj)]
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
        h, lut = _constants(x.device)
        xn = norm(x)
        out = _C.mono_mlp(xn.reshape(-1, x.shape[-1]).contiguous(), h, lut, *parts[0], *parts[1], *parts[2], 0, 4)
        with _lock:
            _stats["mono"] += 1
        return out.reshape(*x.shape[:-1], out.shape[-1])

    setattr(forward, CLAIM, True)
    forward.__wrapped__ = original
    MLP.forward = forward
    return True
