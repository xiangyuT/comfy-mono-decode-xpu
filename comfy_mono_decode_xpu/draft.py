# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 MTP draft head: generation-owned BF16 weights instead of a DynamicVRAM cast per call.

Under DynamicVRAM every BF16 linear of the MTP head (fc, q/k/v/o, gate/up/down) recasts its weight
through comfy.ops.CastBiasWeightContext on each draft, which is most of the draft's host time. The
draft that follows a step's host sync starts on an idle device, so that host time is device idle.
On a generate's first use, each such linear keeps a copy of its cast weight (about 0.5 GB in total,
only when free memory allows) and later drafts call F.linear on it: the same oneDNN GEMM on the same
values as comfy's forward_comfy_cast_weights. Copies are released with the generate (tiles.py).
Resident weights (no cast), quantized weights and weight/bias functions keep comfy's forward.
"""
from __future__ import annotations

import os
import threading

import torch
import torch.nn.functional as F

from . import guard, tiles

CLAIM = "__comfy_mono_decode_xpu_draft__"
_lock = threading.Lock()
_stats = {"owned_calls": 0, "copied_bytes": 0, "skipped_budget": 0}
_margin = int(float(os.environ.get("MONO_DECODE_XPU_TILE_MARGIN_GB", "3")) * (1 << 30))


def stats():
    with _lock:
        return dict(_stats)


def _linears(mtp):
    """BF16 linears whose comfy forward is run_every_op + CastBiasWeightContext + F.linear: comfy
    linear ops without a quantized layout, AWQ pre-scale or weight/bias functions."""
    from comfy_kitchen.tensor import QuantizedTensor
    out = []
    for m in mtp.modules():
        w = getattr(m, "weight", None)
        if (hasattr(m, "forward_comfy_cast_weights") and isinstance(w, torch.Tensor) and w.dim() == 2
                and not isinstance(w, QuantizedTensor) and getattr(m, "layout_type", None) is None
                and getattr(m, "pre_quant_scale", None) is None and not getattr(m, "weight_function", None)
                and not getattr(m, "bias_function", None)):
            out.append(m)
    return out


def install():
    from comfy.text_encoders import qwen35 as q35
    import comfy.ops
    original_gen = getattr(getattr(q35, "Qwen35", None), "_generate_mtp", None)
    if original_gen is None or getattr(original_gen, CLAIM, False):
        return False

    def make_forward(m, comfy_forward):
        def owned_linear(x, owned):
            w = owned.get("w")
            if w is not None and w.device == x.device and w.dtype == x.dtype:
                if not owned.get("checked"):  # first reuse in this generate: same bits as comfy's forward
                    out = F.linear(x, w, owned["b"])
                    ref = comfy_forward(x)
                    owned["checked"] = True
                    if out.shape != ref.shape or not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                        guard.failed("draft", RuntimeError("owned-weight draft linear differs from comfy's forward"))
                        return ref
                    return out
                comfy.ops.run_every_op()
                with _lock:
                    _stats["owned_calls"] += 1
                return F.linear(x, w, owned["b"])
            if not owned.get("budget_ok"):
                return comfy_forward(x)
            comfy.ops.run_every_op()
            # comfy's forward_comfy_cast_weights for a BF16 linear, keeping what it hands to F.linear
            with comfy.ops.CastBiasWeightContext(m, x, offloadable=True, compute_dtype=x.dtype) as (weight, bias):
                out = F.linear(x, weight, bias)
                resident = not hasattr(m, "_v") and weight is m.weight  # no cast: reference, not a copy
                owned["w"] = weight if resident else weight.clone()
                owned["b"] = bias if (resident or bias is None) else bias.clone()
                if not resident:
                    with _lock:
                        _stats["copied_bytes"] += weight.numel() * weight.element_size()
            return out

        def forward(x, *args, **kwargs):
            if (args or kwargs or guard.off("draft") or not tiles.session_active() or not isinstance(x, torch.Tensor)
                    or getattr(m, "weight_function", None) or getattr(m, "bias_function", None)):
                return comfy_forward(x, *args, **kwargs)
            owned = tiles.cached(("draft", id(m)), lambda: {})
            try:
                return owned_linear(x, owned)
            except Exception as e:  # a pure function of x and the weights: comfy's forward recomputes it
                guard.failed("draft", e)
                owned.clear()
                return comfy_forward(x)
        return forward

    def _generate_mtp(self, embeds, *args, **kwargs):
        mtp = getattr(self, "mtp", None)
        device = getattr(embeds, "device", None)
        if guard.off("draft") or mtp is None or device is None or device.type != "xpu":
            return original_gen(self, embeds, *args, **kwargs)
        try:
            mods = _linears(mtp)
            need = sum(m.weight.numel() * 2 for m in mods)  # BF16 copies
            budget_ok = bool(mods) and torch.xpu.mem_get_info(device)[0] > need + _margin
        except Exception as e:
            guard.failed("draft", e)
            return original_gen(self, embeds, *args, **kwargs)
        if not budget_ok:
            with _lock:
                _stats["skipped_budget"] += 1
            return original_gen(self, embeds, *args, **kwargs)
        saved = []
        for m in mods:
            saved.append((m, vars(m).get("forward")))
            m.forward = make_forward(m, m.forward)
            tiles.cached(("draft", id(m)), lambda: {})["budget_ok"] = True
        try:
            return original_gen(self, embeds, *args, **kwargs)
        finally:
            for m, previous in saved:
                if previous is None:
                    vars(m).pop("forward", None)
                else:
                    m.forward = previous

    setattr(_generate_mtp, CLAIM, True)
    _generate_mtp.__wrapped__ = original_gen
    q35.Qwen35._generate_mtp = _generate_mtp
    return True
