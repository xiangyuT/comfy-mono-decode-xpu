# SPDX-License-Identifier: Apache-2.0
"""Experimental two-stage draft lm_head for the device MTP loop (MONO_DECODE_XPU_DRAFT_INT4).

A draft's lm_head reads the whole int8 head (248320 x 4096, about 1 GB, 1.7 ms at the bandwidth roof)
to pick 7 tokens. Two stages read about half of that:
  1. score every row with an int4 copy of the int8 weights (grouped symmetric, the int8 row scale
     folded into the group scales) through oneDNN's weight-only int4 GEMM (torch XPU)
  2. draft_pick (csrc/draft_lmhead.sycl): keep 1024 candidates by that score (32 per work-group), recompute their exact
     int8 logits the way the decode path does (same rotated, quantized activation), take the top-n with
     torch topk's tie order (and the argmax), and write the draft's outputs
The whole vocabulary is scored; the result equals the int8 draft whenever the exact top-n lies within
the candidates (measured on 8636 real drafts: always within the int4 top-32). Only the draft changes:
the verify and its sampling are untouched, so output tokens stay the verify's.

MONO_DECODE_XPU_DRAFT_INT4: 0 (default) off, 1 on, check: both paths, the int8 result drives the loop
and the two-stage result is compared with it (mismatches counted in mtp_loop.stats()). The int4 copy
(about 0.55 GB) is made per generate when free memory allows, and released with it.
"""
from __future__ import annotations

import contextlib
import os

import torch

from . import guard, tiles

MODE = os.environ.get("MONO_DECODE_XPU_DRAFT_INT4", "0")
GROUP = 128


def enabled():
    return MODE in ("1", "check") and not guard.off("draft_int4")


def _plain(w):
    """(int8 [N, K], float32 [N]) of an int8 convrot lm_head weight, or None."""
    try:
        from comfy_kitchen.tensor import QuantizedTensor
    except ImportError:
        return None
    if not isinstance(w, QuantizedTensor) or getattr(w, "_layout_cls", None) != "TensorWiseINT8Layout":
        return None
    p = w._params
    q, s = w._qdata, getattr(p, "scale", None)
    if (not getattr(p, "convrot", False) or getattr(p, "convrot_groupsize", 0) != 256 or getattr(p, "transposed", False)
            or not isinstance(s, torch.Tensor) or q.dtype != torch.int8 or q.dim() != 2 or not q.is_contiguous()
            or s.numel() != q.shape[0] or s.dtype != torch.float32 or q.shape[1] % GROUP or q.shape[1] % 256):
        return None
    return q, s.reshape(-1).contiguous()


def _weight(head, x):
    """The head's weight on the device for one call (comfy's cast context under DynamicVRAM)."""
    if not head.comfy_cast_weights:
        return contextlib.nullcontext((head.weight, None))
    import comfy.ops
    return comfy.ops.CastBiasWeightContext(head, x, offloadable=True)


class DraftLMHead:
    """Per generate: the int4 copy of the head and the two-stage pick."""

    def __init__(self, C, head, native, q, s):
        self.C, self.head, self.native = C, head, native
        N, K = q.shape
        self.N, self.K = N, K
        g = K // GROUP
        packed = torch.empty((N, K // 2), dtype=torch.uint8, device=q.device)
        self.scales = torch.empty((g, N), dtype=torch.bfloat16, device=q.device)
        self.zeros = torch.full((g, N), 8, dtype=torch.int8, device=q.device)
        step = 16384
        for n0 in range(0, N, step):
            w = q[n0:n0 + step].float().view(-1, g, GROUP)
            sg = w.abs().amax(-1, keepdim=True).clamp_min(1.0) / 7.0
            u = ((w / sg).round().clamp(-8, 7) + 8).to(torch.uint8).view(-1, K)
            packed[n0:n0 + step] = u[:, 1::2] << 4 | u[:, 0::2]  # low nibble: even k
            self.scales[:, n0:n0 + step] = (sg.view(-1, g) * s[n0:n0 + step].view(-1, 1)).t().to(torch.bfloat16)
        self.b4 = torch.ops.aten._convert_weight_to_int4pack(packed, 8)
        del packed
        self.calls = torch.zeros(1, dtype=torch.long, device=q.device)
        self.mismatch = torch.zeros(1, dtype=torch.long, device=q.device)
        self.checked = 0

    @classmethod
    def create(cls, C, head, device):
        """The two-stage head for this generate, or None (not eligible, not enough memory, or a failure)."""
        if not enabled() or not hasattr(C, "draft_pick") or device.type != "xpu":
            return None
        if (not isinstance(getattr(head, "in_features", None), int)  # a linear head (not a tied embedding)
                or getattr(head, "weight_function", None) or getattr(head, "bias_function", None)
                or getattr(head, "bias", None) is not None
                or not hasattr(torch.ops.aten, "_weight_int4pack_mm_with_scales_and_zeros")):
            return None
        try:
            from omni_xpu_kernel import int8 as oi
            native = oi._get_native()
            if native is None or not hasattr(native, "rotate_convrot") or not hasattr(oi, "quantize_int8_rowwise"):
                return None
            x = torch.zeros((1, 1, head.in_features), dtype=torch.bfloat16, device=device)
            with _weight(head, x) as (w, _b):
                plain = _plain(w)
                if plain is None:
                    return None
                q, s = plain
                need = q.numel() + q.numel() // GROUP * 3  # int4 copy (and its packing source) + scales/zeros
                if torch.xpu.mem_get_info(device)[0] < need + tiles._margin:
                    return None
                return cls(C, head, (oi, native), q, s)
        except Exception as e:
            guard.failed("draft_int4", e)
            return None

    def _approx(self, n1):
        oi, native = self.native
        x = n1.reshape(1, self.K)
        xr = oi._native_rotate_convrot(native, x, 256)  # the decode path's rotation and quantization
        qx, sx = oi.quantize_int8_rowwise(xr)
        approx = torch.ops.aten._weight_int4pack_mm_with_scales_and_zeros(xr, self.b4, GROUP, self.scales, self.zeros)
        return approx, qx, sx

    def pick(self, n1, nout, argmax_first, top, ranks_row, token, vtok, children):
        """The draft's outputs from its normed hidden n1: top[:nout], ranks_row[:nout], token (the argmax
        if argmax_first, else top[0]) and vtok[children] (token, then top[1..]). False: nothing was
        written (the head's weight is not the int8 layout this call, or the path is off); the caller
        runs the int8 pick."""
        if guard.off("draft_int4"):
            return False
        try:
            with _weight(self.head, n1) as (w, _b):
                plain = _plain(w)
                if plain is None or plain[0].shape != (self.N, self.K):
                    return False
                q, s = plain
                approx, qx, sx = self._approx(n1)
                empty = torch.empty(0, dtype=torch.long, device=n1.device)
                self.C.draft_pick(approx.view(-1), qx.view(-1), sx.view(-1), q, s, nout, argmax_first,
                                  empty if top is None else top.view(-1),
                                  empty if ranks_row is None else ranks_row.view(-1),
                                  token.view(-1), vtok.view(-1), list(children), self.calls)
            return True
        except Exception as e:  # argument checks fail before anything is written: the int8 pick runs
            guard.failed("draft_int4", e)
            return False

    def check(self, n1, nout, argmax_first, top, token, vtok, children):
        """check mode: the two-stage result for the draft just computed by the int8 path, compared on the device."""
        t2 = torch.empty(nout, dtype=torch.long, device=n1.device)
        k2 = torch.empty(1, dtype=torch.long, device=n1.device)
        v2 = vtok.clone()
        if not self.pick(n1, nout, argmax_first, t2, None, k2, v2, children):
            return
        bad = (k2 != token.view(-1)).any() | (v2.view(-1) != vtok.view(-1)).any()
        if top is not None and nout > 1:
            bad = bad | (t2 != top.view(-1)[:nout]).any()
        self.mismatch += bad.long()
        self.checked += 1

    def counters(self):
        return int(self.calls.item()), int(self.mismatch.item()), self.checked
