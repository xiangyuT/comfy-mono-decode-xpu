"""Exact fast top-k / argmax on single-row BF16 logits (the MTP draft and first-token picks).

BaseLlama.logits returns a zero-copy subclass view; its topk (k <= 8, largest, sorted, last dim) and
argmax (last dim) run mono_decode_xpu.fast_topk, which reproduces torch XPU's results including the order of
tied values (see csrc/fast_topk.sycl). Every other op returns a plain tensor.
"""
from __future__ import annotations

import threading

import torch

CLAIM = "__comfy_mono_decode_xpu_logits__"
_stats = {"topk": 0, "argmax": 0}
_lock = threading.Lock()
_pick_cache = __import__("os").environ.get("MONO_DECODE_XPU_PICK", "1") == "1"


def stats():
    with _lock:
        return dict(_stats)


def _row_ok(t):
    return (t.device.type == "xpu" and t.dtype == torch.bfloat16 and t.is_contiguous()
            and t.dim() >= 1 and t.numel() == t.shape[-1] and t.shape[-1] >= 32768)


def _cache(t):
    with torch._C.DisableTorchFunctionSubclass():  # plain attribute access, no handler re-entry
        return t.__dict__


class FastLogits(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        from . import _C
        if func in (torch.Tensor.topk, torch.topk) and args:
            x = args[0]
            k = args[1] if len(args) > 1 else kwargs.get("k")
            dim = args[2] if len(args) > 2 else kwargs.get("dim", -1)
            largest = args[3] if len(args) > 3 else kwargs.get("largest", True)
            sort = args[4] if len(args) > 4 else kwargs.get("sorted", True)
            if (isinstance(k, int) and 1 <= k <= 8 and largest and sort and dim in (-1, x.dim() - 1)
                    and _row_ok(x) and "out" not in kwargs):
                plain = x.as_subclass(torch.Tensor)
                cached = _cache(x).get("_mono_decode_xpu_pick")
                if cached is not None and cached[0] == plain.data_ptr():
                    v, i = cached[1][:, :k], cached[2][:, :k]  # top-8 from the argmax pass: same order, prefix
                    hit = "topk_cached"
                else:
                    v, i = _C.fast_topk(plain, k, False)
                    hit = "topk"
                shape = list(plain.shape[:-1]) + [k]
                with _lock:
                    _stats[hit] = _stats.get(hit, 0) + 1
                return torch.return_types.topk((v.reshape(shape), i.reshape(shape)))
        if func in (torch.Tensor.argmax, torch.argmax) and args:
            x = args[0]
            dim = args[1] if len(args) > 1 else kwargs.get("dim")
            keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
            if dim is not None and dim in (-1, x.dim() - 1) and _row_ok(x) and "out" not in kwargs:
                plain = x.as_subclass(torch.Tensor)
                if _pick_cache:  # one pass: argmax plus the top-8 a following topk (k <= 8) reuses
                    v8, i8, i = _C.fast_pick(plain, 8)
                    _cache(x)["_mono_decode_xpu_pick"] = (plain.data_ptr(), v8, i8)
                else:
                    _, i = _C.fast_topk(plain, 1, True)
                shape = list(plain.shape[:-1]) + ([1] if keepdim else [])
                with _lock:
                    _stats["argmax"] += 1
                return i.view(shape)
        name = getattr(func, "__name__", "")
        if (name.endswith("_") and not name.startswith("__")) or name in ("__setitem__",) or name.startswith("__i"):
            for a in args:  # an in-place op on a picked row: drop its cached top-8
                if isinstance(a, FastLogits):
                    _cache(a).pop("_mono_decode_xpu_pick", None)
        with torch._C.DisableTorchFunctionSubclass():
            out = func(*args, **kwargs)
        if func is torch.Tensor.__getitem__ and isinstance(out, torch.Tensor):
            return out.as_subclass(cls)  # keep the view eligible for the pick that follows
        return out


def install():
    from comfy.text_encoders import llama
    cls = llama.BaseGenerate  # Qwen35(BaseLlama, BaseGenerate) resolves logits here
    original = cls.logits
    if getattr(original, CLAIM, False):
        return False

    def logits(self, x):
        out = original(self, x)
        if isinstance(out, torch.Tensor) and type(out) is torch.Tensor and out.device.type == "xpu" \
                and out.dtype == torch.bfloat16 and out.shape[-1] >= 32768:
            return out.as_subclass(FastLogits)
        return out

    setattr(logits, CLAIM, True)
    logits.__wrapped__ = original
    cls.logits = logits
    return True
