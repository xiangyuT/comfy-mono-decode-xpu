# SPDX-License-Identifier: Apache-2.0
"""Exact fast top-k for the verify step's sampling filter (MONO_DECODE_XPU_SAMPLE_TOPK).

ComfyUI's processed_probs starts the sampling filters with torch.topk(logits, top_k) over the verify
rows (8 x 248320 FP32): about 270 us per step in torch XPU's sbtopk gather. Inside the device MTP loop's
verify sampling, this mode serves that call with topk_rows (csrc/topk_rows.sycl), which returns the same
values and indices in the same order, ties included (torch's gather order). Every other op of the
sampling (temperature, min-p, top-p, softmax, multinomial and its generator) is upstream's.

"1" (default) on, "0" off, "check": both run, torch's result is used and mismatches are counted in stats().
"""
from __future__ import annotations

import contextlib
import os
import threading

import torch

from . import guard

MODE = os.environ.get("MONO_DECODE_XPU_SAMPLE_TOPK", "1")
_lock = threading.Lock()
_stats = {"calls": 0, "checked": 0, "check_mismatch": 0, "skipped": 0}


def stats():
    with _lock:
        return dict(_stats)


def _args(args, kwargs):
    """(x, k) for torch.topk(x, k, dim=-1, largest=True, sorted=True) on rows topk_rows serves, else None."""
    if len(args) < 2 or kwargs.get("out") is not None:
        return None
    x, k = args[0], args[1]
    dim = args[2] if len(args) > 2 else kwargs.get("dim", -1)
    largest = args[3] if len(args) > 3 else kwargs.get("largest", True)
    sorted_ = args[4] if len(args) > 4 else kwargs.get("sorted", True)
    if (not isinstance(x, torch.Tensor) or type(x) is not torch.Tensor or not isinstance(k, int) or isinstance(k, bool)
            or dim not in (-1, 1) or largest is not True or sorted_ is not True):
        return None
    if (x.device.type != "xpu" or x.dtype != torch.float32 or x.dim() != 2 or not x.is_contiguous()
            or not 1 <= x.shape[0] <= 16 or x.shape[1] < 32768 or not 1 <= k <= 64):
        return None
    return x, k


class _TopkRows(torch.overrides.TorchFunctionMode):
    def __init__(self, C):
        super().__init__()
        self.C = C

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
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
