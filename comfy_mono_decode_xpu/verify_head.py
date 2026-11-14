# SPDX-License-Identifier: Apache-2.0
"""Per-generate final RMSNorm + ConvRot/quantize for the verify lm_head.

The normalized hidden is still returned to the model and MTP accept. Only the
head's activation preparation is shared; its INT8 GEMM stays on oneDNN.
"""
from __future__ import annotations

import contextlib
import os

import torch

from . import draft_lmhead, layer

MODE = os.environ.get("MONO_DECODE_XPU_VERIFY_HEAD", "1")
EMBED_MODE = os.environ.get("MONO_DECODE_XPU_EMBED", "1")
_stats = {"created": 0, "calls": 0, "checked": 0}


def stats():
    return dict(_stats)


class EmbeddingRows:
    """Fuse gather/cast/scale, retaining the original FP32 inverse-rotation GEMM."""
    def __init__(self, C, embed, device):
        from comfy_kitchen.tensor.int8_utils import _build_hadamard
        self.C, self.embed = C, embed
        self.h = _build_hadamard(256, device=device, dtype=torch.float32)
        self.checked = False

    @classmethod
    def create(cls, C, embed, device):
        from torch.nn.modules import module
        if (EMBED_MODE != "1" or device.type != "xpu" or not hasattr(C, "embedding_gather")
                or any((module._global_forward_hooks, module._global_forward_pre_hooks))
                or embed._forward_hooks or embed._forward_pre_hooks or "forward" in vars(embed)
                or getattr(embed.forward, "__module__", None) != "comfy.ops"
                or getattr(embed, "weight_function", None) or getattr(embed, "weight_lowvram_function", None)
                or draft_lmhead._plain(embed.weight) is None):
            return None
        return cls(C, embed, device)

    def forward(self, ids):
        import comfy.ops
        import comfy_kitchen as ck
        from comfy_kitchen.tensor.int8_utils import _rotate_weight
        if ck.registry.get_backend_override() is not None or not ck.registry.is_available("xpu"):
            return None
        e = self.embed
        context = (comfy.ops.CastBiasWeightContext(e, device=ids.device, dtype=e.weight.dtype, offloadable=True)
                   if getattr(e, "comfy_cast_weights", False) else contextlib.nullcontext((e.weight, None)))
        with context as (w, _bias):
            parts = draft_lmhead._plain(w)
            if parts is None:
                raise ValueError("decode embedding quantization changed during generation")
            x, valid = self.C.embedding_gather(ids.contiguous(), *parts)
            torch._assert_async(valid, "decode embedding token outside vocabulary")
            # Keep exactly the original matrix, transpose, dtype, shape and
            # oneDNN accumulation; a butterfly would change verify rounding.
            out = _rotate_weight(x.reshape(-1, x.shape[-1]), self.h, 256).reshape(x.shape).bfloat16()
            if not self.checked:
                ref = e(ids).bfloat16()
                if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                    raise ArithmeticError("decode embedding differs from the live lookup")
                self.checked = True
        _stats["embedding"] = _stats.get("embedding", 0) + 1
        return out


class VerifyHead:
    def __init__(self, C, model, device):
        self.C, self.norm, self.head = C, model.model.norm, model.model.lm_head
        self.original = self.norm.forward
        self.w = layer._norm_weight(self.norm, device)
        self.h = layer._constants(device)[0]
        self.pending = None
        self.checked = set()
        self.head_checked = set()

    @classmethod
    def create(cls, C, model, device):
        from comfy.text_encoders import llama
        from torch.nn.modules import module
        norm, head = getattr(model.model, "norm", None), getattr(model.model, "lm_head", None)
        if (MODE == "0" or not hasattr(C, "verify_norm_rot_quant") or device.type != "xpu"
                or type(norm) is not llama.RMSNorm or norm.weight.dtype != torch.bfloat16
                or norm.weight.numel() != 4096 or getattr(head, "in_features", None) != 4096
                or getattr(head, "_full_precision_mm_config", False)
                or getattr(head, "pre_quant_scale", None) is not None
                or getattr(head, "input_scale", None) is not None
                or any((module._global_forward_hooks, module._global_forward_pre_hooks))):
            return None
        for m in (norm, head):
            if (m._forward_hooks or m._forward_pre_hooks or "forward" in vars(m)
                    or getattr(m, "weight_function", None) or getattr(m, "bias_function", None)
                    or getattr(m, "bias", None) is not None):
                return None
        # Upstream verify_logits calls F.linear directly inside the cast
        # context; it does not call head.forward. Scaler's forward wrapper,
        # factory full_precision_mm and force-cast flags do not change that
        # operation. Preserve the cast below and the file's explicit policy.
        if (getattr(norm.forward, "__module__", None) != "comfy.text_encoders.llama"
                or draft_lmhead._plain(head.weight) is None):
            return None
        if not head.can_use_quantized_matmul([]):
            return None
        _stats["created"] += 1
        return cls(C, model, device)

    @contextlib.contextmanager
    def capture(self):
        """Only the verify forward uses this instance's final norm wrapper."""
        if "forward" in vars(self.norm):
            raise RuntimeError("verify norm forward changed during generation")
        self.norm.forward = self.forward
        try:
            yield
        finally:
            del self.norm.forward

    def forward(self, x):
        if x.dtype != torch.bfloat16 or not x.is_contiguous() or x.numel() // 4096 > 8:
            raise ValueError("verify norm input changed outside the prepared decode contract")
        out, q, s = self.C.verify_norm_rot_quant(x, self.w, self.norm.eps, self.h)
        rows = q.shape[0]
        if rows not in self.checked:
            from omni_xpu_kernel import int8 as oi
            ref = self.original(x)
            rq, rs = oi.quantize_int8_rowwise(oi.rotate_convrot(ref.contiguous(), 256))
            if not (torch.equal(out.view(torch.int16), ref.view(torch.int16))
                    and torch.equal(q, rq.reshape_as(q))
                    and torch.equal(s.view(torch.int32), rs.reshape_as(s).view(torch.int32))):
                raise ArithmeticError("verify final norm/quantization differs from the live operator chain")
            self.checked.add(rows)
            _stats["checked"] += 1
        self.pending = (out, q, s)
        return out

    def logits(self, x):
        import comfy_kitchen as ck
        from omni_xpu_kernel import int8 as oi
        if self.pending is None or self.pending[0] is not x:
            raise RuntimeError("verify lm_head did not consume its prepared normalized hidden")
        _, qx, sx = self.pending
        self.pending = None
        with draft_lmhead._weight(self.head, x) as (w, bias):
            if bias is not None:
                raise ValueError("verify lm_head acquired a bias")
            parts = draft_lmhead._plain(w)
            if parts is None:
                raise ValueError("verify lm_head quantization changed during generation")
            q, s = parts
            # Honor Kitchen's explicit backend selection and disable/priority
            # settings. The prepared input is only for its native XPU route.
            backend = ck.registry.get_capable_backend("int8_linear", dict(
                x=x, weight=q, weight_scale=s, bias=None, out_dtype=x.dtype,
                convrot=True, convrot_groupsize=256, input_act=None, input_act_weight=None,
                input_act_eps=0., residual=None, residual_scale=None))
            if backend != "xpu":
                return torch.nn.functional.linear(x, w)
            out = oi._get_native().int8_linear_prequantized(qx, sx, q, s, None, 2)
            if qx.shape[0] not in self.head_checked:
                ref = torch.nn.functional.linear(x, w).reshape_as(out)
                if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                    raise ArithmeticError("verify prepared lm_head differs from the live F.linear")
                self.head_checked.add(qx.shape[0])
        _stats["calls"] += 1
        return out.reshape(*x.shape[:-1], out.shape[-1])
