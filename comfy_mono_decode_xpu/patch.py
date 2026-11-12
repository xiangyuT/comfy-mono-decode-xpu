# SPDX-License-Identifier: Apache-2.0
"""Route small-M packed W4A8 linears to the fused mono_decode_xpu decode kernel.

The wrapped entry is ``omni_xpu_kernel.int8.w4a8_linear_prequantized``, which the XPU
Kitchen backend looks up on every call. A call that matches the kernel contract (1-8 rows,
K a multiple of 2048, G16 or G%32, FP32/FP8 group scales, no bias, BF16/FP32 output,
aligned contiguous operands) runs one fused kernel whose result is bit-identical to the
original mm_w4a8 -> fused_scaleback -> cast sequence. Every other call runs the original.
"""
from __future__ import annotations

import atexit
import json
import os
import threading

import torch

from . import guard

CLAIM = "__comfy_mono_decode_xpu__"
_stats = {"fused": 0, "fallback": 0, "reasons": {}}
_lock = threading.Lock()


def _reason(r):
    with _lock:
        _stats["fallback"] += 1
        _stats["reasons"][r] = _stats["reasons"].get(r, 0) + 1


def _contract(x, x_scale, packed, s_rel, s_channel, codebook, bias, group_size, out_dtype):
    if bias is not None:
        return "bias"
    if out_dtype not in (torch.bfloat16, torch.float32):
        return "out_dtype"
    if x.device.type != "xpu" or x.dtype != torch.int8 or not x.is_contiguous():
        return "x"
    k = x.shape[-1]
    rows = x.numel() // k if k else 0
    if not 1 <= rows <= 8:
        return "rows"
    if k % 2048 or not (group_size == 16 or (group_size % 32 == 0 and k % group_size == 0)):
        return "shape"
    if packed.dtype != torch.int8 or packed.dim() != 2 or not packed.is_contiguous() or packed.shape[1] * 2 != k:
        return "packed"
    if packed.data_ptr() % 16 or x.data_ptr() % 32:
        return "align"
    if s_rel.dtype not in (torch.float32, torch.float8_e4m3fn) or not s_rel.is_contiguous():
        return "s_rel"
    if s_channel.numel() != packed.shape[0]:
        return "s_channel"
    if codebook is not None and (codebook.dtype != torch.float32 or codebook.numel() != 16):
        return "codebook"
    return None


def install():
    from omni_xpu_kernel import int8 as oint8

    original = oint8.w4a8_linear_prequantized
    if getattr(original, CLAIM, False):
        return False
    from . import _C

    clear = [getattr(oint8, n, None) for n in ("_clear_krea2_activation_cache", "_clear_bmg_qkv_activation_cache")]

    def w4a8_linear_prequantized(x_int8, x_scale, packed, relative_scale, channel_scale,
                                 codebook=None, bias=None, group_size=16, out_dtype=torch.bfloat16):
        if guard.off("w4") or torch.compiler.is_compiling():
            return original(x_int8, x_scale, packed, relative_scale, channel_scale, codebook, bias, group_size, out_dtype)
        why = _contract(x_int8, x_scale, packed, relative_scale, channel_scale, codebook, bias, group_size, out_dtype)
        if why is None and (x_scale.device != x_int8.device or relative_scale.device != x_int8.device
                            or channel_scale.device != x_int8.device or packed.device != x_int8.device
                            or (codebook is not None and codebook.device != x_int8.device)
                            or x_scale.numel() != x_int8.numel() // x_int8.shape[-1]):
            why = "operands"
        if why is not None:
            _reason(why)
            return original(x_int8, x_scale, packed, relative_scale, channel_scale, codebook, bias, group_size, out_dtype)
        for fn in clear:  # the original entry clears these activation caches on every call
            if fn is not None:
                fn()
        k = x_int8.shape[-1]
        try:
            x2 = x_int8.reshape(-1, k)
            xs = x_scale.reshape(-1).to(torch.float32).contiguous()
            sc = channel_scale.reshape(-1).to(torch.float32).contiguous()
            out = _C.w4a8_decode_linear(x2, xs, packed, relative_scale, sc, codebook, None, group_size,
                                        0 if out_dtype == torch.float32 else 2, 0, False)
        except Exception as e:  # pure function of its inputs: the original recomputes it
            guard.failed("w4", e)
            return original(x_int8, x_scale, packed, relative_scale, channel_scale, codebook, bias, group_size, out_dtype)
        with _lock:
            _stats["fused"] += 1
        return out.reshape(*x_int8.shape[:-1], packed.shape[0])

    setattr(w4a8_linear_prequantized, CLAIM, True)
    w4a8_linear_prequantized.__wrapped__ = original
    oint8.w4a8_linear_prequantized = w4a8_linear_prequantized
    output = os.environ.get("MONO_DECODE_XPU_STATS")
    if output:
        def export_stats():
            with open(output, "w") as stream:
                json.dump(stats(), stream)
        atexit.register(export_stats)
    return True


def install_prep():
    """Route decode-row ConvRot rotation + row quantization to mono_decode_xpu.rot_quant (bit-identical)."""
    import sys as _sys
    from . import _C
    targets = [m for n, m in list(_sys.modules.items())
               if n.endswith("backends.xpu.w4a8_int8") and hasattr(m, "quantize_and_rotate_rowwise")]
    done = 0
    for mod in targets:
        original = mod.quantize_and_rotate_rowwise
        if getattr(original, CLAIM, False):
            continue

        def quantize_and_rotate_rowwise(x, h, group_size, stochastic_rounding=0, _original=original):
            k = x.shape[-1] if x.dim() else 0
            ok = (not guard.off("prep") and group_size == 256 and not stochastic_rounding and x.device.type == "xpu"
                  and x.dtype == torch.bfloat16 and h.dtype == torch.bfloat16 and h.numel() == 65536
                  and k and k % 2048 == 0 and k <= 12288 and 1 <= x.numel() // k <= 8)
            if not ok:
                _reason("prep")
                return _original(x, h, group_size, stochastic_rounding)
            try:
                out = _C.rot_quant(x.contiguous(), h.contiguous(), 3)
            except Exception as e:
                guard.failed("prep", e)
                return _original(x, h, group_size, stochastic_rounding)
            with _lock:
                _stats["prep"] = _stats.get("prep", 0) + 1
            return out

        setattr(quantize_and_rotate_rowwise, CLAIM, True)
        quantize_and_rotate_rowwise.__wrapped__ = original
        mod.quantize_and_rotate_rowwise = quantize_and_rotate_rowwise
        done += 1
    return done


def stats():
    with _lock:
        return json.loads(json.dumps(_stats))
