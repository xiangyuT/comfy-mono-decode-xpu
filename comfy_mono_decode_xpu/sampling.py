# SPDX-License-Identifier: Apache-2.0
"""Exact fast top-k for the verify step's sampling filter (MONO_DECODE_XPU_SAMPLE_TOPK).

ComfyUI's processed_probs starts the sampling filters with torch.topk(logits, top_k) over the verify
rows (8 x 248320 FP32): about 270 us per step in torch XPU's sbtopk gather. Inside the device MTP loop's
verify sampling, this mode serves that call with topk_rows (csrc/topk_rows.sycl), which returns the same
values and indices in the same order, ties included (torch's gather order). Every other op of the
sampling (temperature, min-p, top-p, softmax, multinomial and its generator) is upstream's.

The rows handed to the sampling are the BF16 logits standing for their FP32 copy (verify_rows): topk_rows
reads them directly, shape queries pass through, the dtype reads FP32, and any other op on them runs on
the FP32 copy, made then (with temperature 1 and top_k > 0, upstream only takes the shape and the top-k,
so the 24 us copy is not made at all).

top-p then sorts the top-k values (torch.sort(logits, descending=True)). They are already sorted, ties in
their order, and torch XPU's sort is stable (300 of 300 checks, tie-heavy rows included: values and
arange), so for the values topk_rows returned in this mode (a min-p mask only lowers their tail) the
sort returns a copy and the cached arange instead of a 10 us segmented radix sort.

"1" (default) on, "0" off, "check": both run, torch's result is used and mismatches are counted in stats().
"""
from __future__ import annotations

import contextlib
import hashlib
import inspect
import os
import threading

import torch

from . import guard

MODE = os.environ.get("MONO_DECODE_XPU_SAMPLE_TOPK", "1")
FUSED_MODE = os.environ.get("MONO_DECODE_XPU_SAMPLE_FUSED", "1")
_lock = threading.Lock()
_stats = {"calls": 0, "checked": 0, "check_mismatch": 0, "skipped": 0}


def fused_compatible(model):
    """Only replace the known sampling policy, preserving instance/class overrides."""
    from comfy.text_encoders import llama
    sample = getattr(model.sample_token, "__func__", None)
    processed = getattr(model.processed_probs, "__func__", None)
    if sample is not llama.BaseGenerate.sample_token or processed is not llama.BaseGenerate.processed_probs:
        return False
    # autotree wraps processed_probs only to remove masked-assignment waits.
    if getattr(processed, "__comfy_mono_decode_xpu_autotree__", False):
        processed = processed.__wrapped__
    try:
        source = inspect.getsource(processed) + inspect.getsource(sample)
    except (OSError, TypeError):
        return False
    return hashlib.sha256(source.encode()).hexdigest() == "d23a1250bec4d5ff1f14b240debbbd60475d3e0948caa1d7347115e116a28f65"


def fused_sample(C, logits, policy, generator):
    """Native filtering and pick, with torch's original exponential RNG stream.

    Returns None before doing work for an unsupported contract. Once the RNG
    has advanced, errors propagate; sampling must never replay with a new draw.
    """
    if (FUSED_MODE != "1" or MODE in ("0", "check") or guard.off("sample_fused") or guard.off("sample_topk")
            or not hasattr(C, "sampling_pick")
            or not isinstance(logits, torch.Tensor) or type(logits) is not torch.Tensor
            or logits.device.type != "xpu" or logits.dtype not in (torch.bfloat16, torch.float32)
            or logits.dim() != 2 or not logits.is_contiguous() or not 1 <= logits.shape[0] <= 16
            or logits.shape[1] < 32768):
        return None
    k = policy["top_k"]
    temperature, top_p, min_p = policy["temperature"], policy["top_p"], policy["min_p"]
    if (not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 64
            or temperature != 1.0 or not 0. <= top_p <= 1. or not 0. <= min_p <= 1.):
        return None
    v, indices = C.topk_rows(logits, k)
    # Same shape, strides, dtype and exponential_ call as torch.multinomial's
    # one-sample Gumbel-max path. Only deterministic work moves around the draw.
    noise = torch.empty_like(v).exponential_(1., generator=generator)
    picked, valid = C.sampling_pick(v, indices, noise, top_p, min_p)
    torch._assert_async(valid.all(), "probability tensor contains inf, nan, negative values or no positive value")
    with _lock:
        _stats["fused"] = _stats.get("fused", 0) + 1
    return picked


def stats():
    with _lock:
        return dict(_stats)


def _args(args, kwargs, lazy=False):
    """(x, k) for torch.topk(x, k, dim=-1, largest=True, sorted=True) on rows topk_rows serves, else None
    (lazy: x is the BF16 stand-in for FP32 rows)."""
    if len(args) < 2 or kwargs.get("out") is not None:
        return None
    x, k = args[0], args[1]
    dim = args[2] if len(args) > 2 else kwargs.get("dim", -1)
    largest = args[3] if len(args) > 3 else kwargs.get("largest", True)
    sorted_ = args[4] if len(args) > 4 else kwargs.get("sorted", True)
    if (not isinstance(x, torch.Tensor) or type(x) is not torch.Tensor or not isinstance(k, int) or isinstance(k, bool)
            or dim not in (-1, 1) or largest is not True or sorted_ is not True):
        return None
    if (x.device.type != "xpu" or x.dtype != (torch.bfloat16 if lazy else torch.float32) or x.dim() != 2 or not x.is_contiguous()
            or not 1 <= x.shape[0] <= 16 or x.shape[1] < 32768 or not 1 <= k <= 64):
        return None
    return x, k


_aranges = {}


def _arange(R, k, device):
    key = (R, k, device)
    if key not in _aranges:
        _aranges[key] = torch.arange(k, device=device).expand(R, k).contiguous()  # read only (scatter index)
    return _aranges[key]


_SHAPE = {torch.Tensor.size, torch.Tensor.dim, torch.Tensor.numel, torch.Tensor.shape.__get__,
          torch.Tensor.ndim.__get__, torch.Tensor.device.__get__, torch.Tensor.is_contiguous}


class _TopkRows(torch.overrides.TorchFunctionMode):
    def __init__(self, C, lazy=None):
        super().__init__()
        self.C = C
        self.lazy = lazy  # a BF16 [R, N] tensor standing for lazy.float()
        self.dense = None
        self.sorted = []  # the top-k values returned here (sorted descending)

    def _swap(self, a):
        if a is self.lazy:
            if self.dense is None:
                self.dense = self.lazy.float()  # (this mode is not active inside __torch_function__)
            return self.dense
        if isinstance(a, (list, tuple)):
            return type(a)(self._swap(x) for x in a)
        return a

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        lazy = self.lazy is not None and args and args[0] is self.lazy
        if lazy and func in _SHAPE:
            return func(*args, **kwargs)
        if lazy and func is torch.Tensor.dtype.__get__:
            return torch.float32
        if lazy and (func is torch.topk or func is torch.Tensor.topk):
            xk = _args((self.lazy,) + tuple(args[1:]), kwargs, lazy=True)
            if xk is not None and not guard.off("sample_topk"):
                try:
                    v, i = self.C.topk_rows(*xk)
                except Exception as e:  # an argument check: torch computes it on the FP32 copy
                    guard.failed("sample_topk", e)
                else:
                    if MODE != "check":
                        with _lock:
                            _stats["calls"] += 1
                            _stats["bf16_rows"] = _stats.get("bf16_rows", 0) + 1
                        self.sorted.append(v)
                        return torch.return_types.topk((v, i))
        if (func is torch.sort or func is torch.Tensor.sort) and args and any(args[0] is v for v in self.sorted):
            dim = args[1] if len(args) > 1 else kwargs.get("dim", -1)
            desc = args[2] if len(args) > 2 else kwargs.get("descending", False)
            if dim in (-1, 1) and desc is True and set(kwargs) <= {"dim", "descending", "stable"} and len(args) <= 3:
                x = args[0]
                with _lock:
                    _stats["sorted"] = _stats.get("sorted", 0) + 1
                return torch.return_types.sort((x.clone(), _arange(x.shape[0], x.shape[1], x.device)))
        if self.lazy is not None:
            args = self._swap(tuple(args))
            kwargs = {k: self._swap(v) for k, v in kwargs.items()}
        if func is torch.topk or func is torch.Tensor.topk:
            xk = _args(args, kwargs)
            if xk is not None and not guard.off("sample_topk"):
                try:
                    v, i = self.C.topk_rows(*xk)
                except Exception as e:  # an argument check: torch computes it
                    guard.failed("sample_topk", e)
                else:
                    if MODE != "check":
                        with _lock:
                            _stats["calls"] += 1
                        self.sorted.append(v)
                        return torch.return_types.topk((v, i))
                    ref = func(*args, **kwargs)
                    bad = not (torch.equal(ref.indices, i) and torch.equal(ref.values.view(torch.int32), v.view(torch.int32)))
                    with _lock:
                        _stats["checked"] += 1
                        _stats["check_mismatch"] += int(bad)
                    return ref
            with _lock:
                _stats["skipped"] += 1
        return func(*args, **kwargs)


def topk_mode(C):
    """The verify sampling's context: topk_rows serves its eligible torch.topk calls."""
    if MODE == "0" or guard.off("sample_topk") or not hasattr(C, "topk_rows"):
        return contextlib.nullcontext()
    return _TopkRows(C)


def verify_rows(C, lg0):
    """(context, rows) for the verify sampling of the BF16 logits lg0 [R, N]: rows stands for lg0.float()
    (the BF16 tensor itself inside the context when it is served lazily, else the FP32 copy)."""
    if (MODE in ("0", "check") or guard.off("sample_topk") or not hasattr(C, "topk_rows") or lg0.dtype != torch.bfloat16
            or lg0.dim() != 2 or not lg0.is_contiguous() or lg0.device.type != "xpu"):
        return topk_mode(C), lg0.float()
    return _TopkRows(C, lazy=lg0), lg0
