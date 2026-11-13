# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 MTP draft head with fused small kernels, inside the device MTP loop (MONO_DECODE_XPU_DRAFT_FUSED).

The eager head (embedding lookup, MTPHead.forward, lm_head, top-k and the copies after it) is about 45
launches per draft, of which about 35 are small (0.17 ms of kernels and gaps against 2.6 ms of GEMMs).
This head keeps upstream's structure and GEMMs (oneDNN, on BF16 weights owned for the generate) and
replaces the rest (csrc/draft_fused.sycl):

    embedding row (int8 ConvRot: gather, scale, un-rotate) + both pre-fc RMSNorms + cat   one launch
    fc; input RMSNorm; q|k|v (one GEMM on concatenated weights)                           oneDNN + one launch
    fused Q/K norm + RoPE + KV write, DPAS attention                                       as before (layer.py)
    attn * sigmoid(gate)                                                                   one launch
    o_proj; x += o, RMSNorm                                                                oneDNN + one launch
    gate|up (one GEMM); silu(gate) * up; down                                              oneDNN + one launch
    x += down, final RMSNorm, the lm_head's ConvRot rotation and INT8 quantization          one launch
    lm_head (prequantized INT8 GEMM, or the experimental two-stage int4 one), draft_pick   two launches

draft_pick (csrc/draft_lmhead.sycl) takes the exact INT8 logits as its scores and writes the draft's
top-n (torch topk's tie order), argmax, auto-tree rank row and the children's verify tokens.

Only drafts change: their arithmetic is the eager head's in FP32 but not its reduction order, so a
draft token can differ where two candidates are within rounding. The verify decides every output
token. 0 disables; drafts then run the eager head. Needs the int8 ConvRot lm_head and BF16 linears.
"""
from __future__ import annotations

import contextlib
import os

import torch
import torch.nn.functional as F

from . import draft_lmhead, guard, tiles

MODE = os.environ.get("MONO_DECODE_XPU_DRAFT_FUSED", "1")
_stats = {"created": 0, "rejected": {}}


def stats():
    return dict(_stats, rejected=dict(_stats["rejected"]))


def enabled():
    return MODE == "1" and not guard.off("draft_fused")


def _reject(why):
    _stats["rejected"][why] = _stats["rejected"].get(why, 0) + 1
    return None


def _observer(m):
    """An instance-level forward on m that only observes (timing / counting wrappers marked
    _mono_observer), or none: the fused head does not call forward."""
    f = vars(m).get("forward")
    return f is None or getattr(f, "_mono_observer", False)


def _ctx(m, x):
    if not getattr(m, "comfy_cast_weights", False):
        return contextlib.nullcontext((m.weight, getattr(m, "bias", None)))
    import comfy.ops
    return comfy.ops.CastBiasWeightContext(m, x, offloadable=True)


def _owned_weight(m, x):
    """The BF16 weight comfy hands to F.linear, copied for the generate (resident: the tensor itself)."""
    import comfy.ops
    with comfy.ops.CastBiasWeightContext(m, x, offloadable=True, compute_dtype=x.dtype) as (w, b):
        if b is not None:
            raise ValueError("draft linear with bias")
        resident = not hasattr(m, "_v") and w is m.weight
        return w if resident else w.clone()


class DraftHead:
    """Per generate: the owned weights, scratch rows and the fused forward."""

    def __init__(self, C, model, device, lm2):
        from comfy_kitchen.tensor.int8_utils import _build_hadamard
        from . import draft, layer
        mtp = model.mtp
        blk = mtp.layers[0]
        sa, mlp = blk.self_attn, blk.mlp
        self.C, self.model, self.mtp, self.sa, self.layer, self.lm2 = C, model, mtp, sa, layer, lm2
        H = model.model.config.hidden_size
        self.H = H
        x1 = torch.zeros((1, 1, H), dtype=torch.bfloat16, device=device)
        x2 = torch.zeros((1, 1, 2 * H), dtype=torch.bfloat16, device=device)
        xi = torch.zeros((1, 1, mlp.down_proj.in_features), dtype=torch.bfloat16, device=device)
        self.fc = _owned_weight(mtp.fc, x2)
        self.qkv = torch.cat([_owned_weight(m, x1) for m in (sa.q_proj, sa.k_proj, sa.v_proj)])
        self.o = _owned_weight(sa.o_proj, xi[..., :sa.o_proj.in_features])
        self.gate_up = torch.cat([_owned_weight(m, x1) for m in (mlp.gate_proj, mlp.up_proj)])
        self.down = _owned_weight(mlp.down_proj, xi)
        self.split = (sa.num_heads * sa.head_dim * 2, sa.num_kv_heads * sa.head_dim, sa.num_kv_heads * sa.head_dim)
        nw = layer._norm_weight
        self.n_e, self.n_h = nw(mtp.pre_fc_norm_embedding, device), nw(mtp.pre_fc_norm_hidden, device)
        self.n_in, self.n_post, self.n_out = nw(blk.input_layernorm, device), nw(blk.post_attention_layernorm, device), nw(mtp.norm, device)
        self.eps = (mtp.pre_fc_norm_embedding.eps, mtp.pre_fc_norm_hidden.eps, blk.input_layernorm.eps,
                    blk.post_attention_layernorm.eps, mtp.norm.eps)
        self.hmat = _build_hadamard(256, device=device, dtype=torch.float32).contiguous()
        self.fc_in = torch.empty((1, 1, 2 * H), dtype=torch.bfloat16, device=device)
        self.h = torch.empty((1, 1, H), dtype=torch.bfloat16, device=device)
        self.xr = torch.empty(H, dtype=torch.bfloat16, device=device)
        self.qx = torch.empty(H, dtype=torch.int8, device=device)
        self.sx = torch.empty(1, dtype=torch.float32, device=device)
        self.empty = torch.empty(0, device=device)
        self.calls = torch.zeros(1, dtype=torch.long, device=device)
        self.embed = model.model.embed_tokens
        self.head = model.model.lm_head

    @classmethod
    def create(cls, C, model, device, lm2=None):
        """The fused head for this generate, or None (not eligible, not enough memory, or a failure)."""
        if not enabled() or device.type != "xpu" or not all(hasattr(C, f) for f in (
                "draft_embed_norm_cat", "draft_add_rmsnorm", "draft_sigmoid_mul", "draft_silu_mul",
                "draft_norm_rot_quant", "draft_pick")):
            return _reject("unavailable")
        try:
            from comfy.text_encoders import llama
            from comfy.text_encoders import qwen35 as q35
            from torch.nn.modules import module
            from . import draft
            mtp = getattr(model, "mtp", None)
            if type(mtp) is not q35.MTPHead or len(mtp.layers) != 1 or not hasattr(model.model, "lm_head"):
                return _reject("structure")
            blk = mtp.layers[0]
            sa, mlp = getattr(blk, "self_attn", None), blk.mlp
            if (type(sa) is not q35.GatedAttention or getattr(mlp, "merged_mlp", False)
                    or getattr(mlp, "activation", None) is not F.silu or sa.num_heads * sa.head_dim != model.model.config.hidden_size
                    or any((module._global_forward_hooks, module._global_forward_pre_hooks))):
                return _reject("structure")
            for m in (mtp, blk, sa, mlp):
                if m._forward_hooks or m._forward_pre_hooks or not _observer(m):
                    return _reject("hooks")
            lins = draft._linears(mtp)
            if {id(m) for m in lins} != {id(m) for m in (mtp.fc, sa.q_proj, sa.k_proj, sa.v_proj, sa.o_proj,
                                                          mlp.gate_proj, mlp.up_proj, mlp.down_proj)}:
                return _reject("linears")  # quantized, patched or biased draft linears: the eager head
            for n in (mtp.pre_fc_norm_embedding, mtp.pre_fc_norm_hidden, blk.input_layernorm, blk.post_attention_layernorm, mtp.norm):
                if type(n) is not llama.RMSNorm or n.weight.dtype != torch.bfloat16:
                    return _reject("norms")
            head = model.model.lm_head
            if head.weight_function or head.bias_function or getattr(head, "bias", None) is not None:
                return _reject("lm_head")
            x = torch.zeros((1, 1, head.in_features), dtype=torch.bfloat16, device=device)
            with _ctx(head, x) as (w, _b):
                if draft_lmhead._plain(w) is None:
                    return _reject("lm_head")
            need = sum(m.weight.numel() * 2 for m in lins)
            if torch.xpu.mem_get_info(device)[0] < need + tiles._margin:
                return _reject("memory")
            head_ = cls(C, model, device, lm2)
            _stats["created"] += 1
            return head_
        except Exception as e:
            guard.failed("draft_fused", e)
            return _reject("error")

    def _embed(self, token, hidden):
        """fc input: cat(RMSNorm(embedding(token)), RMSNorm(hidden)) in self.fc_in."""
        import comfy.ops
        e = self.embed
        C = self.C
        eps_e, eps_h = self.eps[0], self.eps[1]
        host = (hasattr(e, "_v") and e.weight_lowvram_function is None and len(e.weight_function) == 0
                and comfy.ops.vbar_above_watermark(e))
        w = e.weight
        p = getattr(w, "_params", None)
        if (not host and len(e.weight_function) == 0 and getattr(e, "quant_format", None) == "int8_tensorwise"
                and getattr(w, "_layout_cls", None) == "TensorWiseINT8Layout"
                and getattr(p, "convrot_groupsize", 256) == 256):
            with comfy.ops.CastBiasWeightContext(e, device=token.device, dtype=w.dtype, offloadable=True) as (qw, _b):
                q, s = qw._qdata, qw._params.scale.reshape(-1)
                if q.dtype == torch.int8 and q.is_contiguous() and s.numel() == q.shape[0]:
                    hm = self.hmat if getattr(qw._params, "convrot", False) else self.empty
                    C.draft_embed_norm_cat(token.reshape(-1), q, s.float().contiguous(), hm, self.empty, hidden.reshape(-1),
                                           self.n_e, self.n_h, eps_e, eps_h, self.fc_in)
                    return
        # host rows or another layout: comfy's lookup, then the fused norms and cat
        if host:
            emb = e.host_rows(token.reshape(1, 1), out_dtype=torch.bfloat16)
        else:
            emb = e(token.reshape(1, 1)).to(torch.bfloat16)
        C.draft_embed_norm_cat(self.empty, self.empty, self.empty, self.empty, emb.reshape(-1).contiguous(),
                               hidden.reshape(-1), self.n_e, self.n_h, eps_e, eps_h, self.fc_in)

    def forward(self, token, hidden, freqs, kv):
        """One draft: token (device, [1] int64), hidden ([1, 1, H] BF16, the previous pre-norm hidden).
        Returns the new pre-norm hidden; the lm_head input (rotated, quantized) is in xr / qx / sx."""
        C = self.C
        note = getattr(vars(self.mtp).get("forward"), "_mono_on_draft", None)  # observers count drafts
        if note is not None:
            note()
        self._embed(token, hidden)
        x = F.linear(self.fc_in, self.fc)
        C.draft_add_rmsnorm(x.view(-1), self.empty, self.n_in, self.eps[2], self.h.view(-1))
        qkv = F.linear(self.h, self.qkv)
        qg, xk, xv = qkv.split(self.split, dim=-1)
        attn = self.layer._attn_core(self.sa, qg, xk, xv, None, freqs, None, kv)
        gated = torch.empty(attn.shape, dtype=attn.dtype, device=attn.device)
        C.draft_sigmoid_mul(attn.reshape(-1).contiguous(), qg.reshape(-1), self.sa.head_dim, gated.view(-1))
        o = F.linear(gated, self.o)
        C.draft_add_rmsnorm(x.view(-1), o.view(-1), self.n_post, self.eps[3], self.h.view(-1))
        gu = F.linear(self.h, self.gate_up)
        act = torch.empty((1, 1, gu.shape[-1] // 2), dtype=torch.bfloat16, device=gu.device)
        C.draft_silu_mul(gu.view(-1), act.view(-1))
        d = F.linear(act, self.down)
        C.draft_norm_rot_quant(x.view(-1), d.view(-1), self.n_out, self.eps[4], self.hmat, self.xr, self.qx, self.sx)
        return x

    def pick(self, nout, argmax_first, top, ranks_row, token, vtok, children):
        """The draft's outputs from the lm_head on (xr, qx, sx): exact int8 logits, or the two-stage int4
        scores when the experimental head is on."""
        from omni_xpu_kernel import int8 as oi
        empty = torch.empty(0, dtype=torch.long, device=self.qx.device)
        with _ctx(self.head, self.h) as (w, _b):
            q, s = draft_lmhead._plain(w)
            e8 = torch.empty(0, dtype=torch.int8, device=self.qx.device)
            ef = torch.empty(0, dtype=torch.float32, device=self.qx.device)
            if self.lm2 is not None and not guard.off("draft_int4"):  # int4 scores, int8 recompute of the candidates
                scores = torch.ops.aten._weight_int4pack_mm_with_scales_and_zeros(
                    self.xr.view(1, -1), self.lm2.b4, draft_lmhead.GROUP, self.lm2.scales, self.lm2.zeros)
                args = (self.qx, self.sx, q, s)
            else:  # the exact int8 logits: the pick reads them as they are
                scores = oi._get_native().int8_linear_prequantized(self.qx.view(1, -1), self.sx.view(1, 1), q, s, None, 2)
                args = (e8, ef, e8, ef)
            self.C.draft_pick(scores.view(-1), *args, nout, argmax_first,
                              empty if top is None else top.view(-1), empty if ranks_row is None else ranks_row.view(-1),
                              token.view(-1), vtok.view(-1), list(children), self.calls)
