# SPDX-License-Identifier: Apache-2.0
"""Route Qwen3.5 decode transformer blocks (1-8 verify rows) through the persistent layer kernels.

A block is ``x += mixer(input_layernorm(x)); x += mlp(post_attention_layernorm(x))``. For decode rows
whose projections are all packed W4A8 (G16, FP8 group scales, ConvRot 256, one codebook):

    Gated-DeltaNet layers (24 of 32)
        layer_gdn   input RMSNorm .. projections .. deferred conv, gates, delta rule, gated norm (1 launch)
        layer_post  out projection + residual + post RMSNorm + MLP + residual             (1 launch)
    full-attention layers (8 of 32)
        layer_pre   input RMSNorm + rotate/quantize + q/k/v projections                   (1 launch)
        qk_norm_rope_kv (Q/K RMSNorm + partial RoPE + KV write), attn_core (attention port)
        layer_post  sigmoid gate + out projection + residual + post RMSNorm + MLP + residual

Also installed here: the attention port for FixedKVCache.attend, the BF16 MTP draft block's attention
through the same rope_kv/attn_core path, RMSNorm scale caching, the fast logits picks (logits.py) and
the per-generate tile-major weight copies (tiles.py).

Each launch is checked against the operator chain it replaces, with the same operands and layout.
The adaptive MTP tree can still change with measured execution cost. Any other
call (prefill or differently quantized weights) runs the original forward. A fast path
that raises is switched off (guard.py) and the call runs the original code, unless the layer state
was already updated in place, in which case the error is re-raised.
"""
from __future__ import annotations

import contextlib
import logging
import os
import threading

import torch
import torch.nn.functional as F

from . import guard, tiles
from .mono import _cast

CLAIM = "__comfy_mono_decode_xpu_layer__"
_stats = {"layer": 0, "fallback": {}}
_lock = threading.Lock()
_consts = {}
_single_gdn = os.environ.get("MONO_DECODE_XPU_GDN1", "1") == "1"
_attn_port = os.environ.get("MONO_DECODE_XPU_ATTN", "1") == "1"
_attn_dpas = os.environ.get("MONO_DECODE_XPU_ATTN_DPAS", "1") == "1"
_gdn_split = os.environ.get("MONO_DECODE_XPU_GDN_SPLIT", "1") == "1"
_rope_fused = os.environ.get("MONO_DECODE_XPU_ROPE", "1") == "1"
_rope_state = {"ok": None}  # fused rope + KV write: None until the one-time check against the live path


def stats():
    with _lock:
        out = {"layer": _stats["layer"], "gdn1": _stats.get("gdn1", 0), "attn": _stats.get("attn", 0),
               "attn_dpas": _stats.get("attn_dpas", 0),
               "rope": _stats.get("rope", 0), "norm_rope": _stats.get("norm_rope", 0),
               "fallback": dict(_stats["fallback"])}
    t = tiles.stats()
    out.update(session_tile_bytes=t["tile_bytes"], session_tile_oom=t["tile_oom"],
               session_tile_live_bytes=t["live_bytes"], session_active=t["active"],
               weight_layout=t["budget"], disabled=guard.disabled())
    return out


def _fallback(why):
    with _lock:
        _stats["fallback"][why] = _stats["fallback"].get(why, 0) + 1


class SubpathError(RuntimeError):
    """A sub-path (fused RoPE / Q/K norm + KV write) failed after it may have written KV. That path
    has already been disabled and logged; the layer path stays on, the call is not replayed."""


def _constants(device):
    c = _consts.get(device)
    if c is None:
        from comfy_kitchen.tensor.int8_utils import _build_hadamard
        h = _build_hadamard(256, device=device, dtype=torch.bfloat16).contiguous()
        allbf = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).to(device)
        # torch's own SiLU / sigmoid for every BF16 input
        c = _consts[device] = (h, F.silu(allbf).contiguous(), torch.sigmoid(allbf).contiguous())
    return c


def _static_plan(block):
    """Per-block structural checks, done once (they read only host-side module state). Any unexpected
    structure means "not supported" (None), never an exception."""
    cached = getattr(block, "_mono_decode_xpu_static", None)
    if cached is not None:
        return cached[0]
    try:
        plan = _static_plan_uncached(block)
    except Exception:
        plan = None
    block._mono_decode_xpu_static = (plan,)
    return plan


def _static_plan_uncached(block):
    from .mono import _static_ok
    gdn = block.layer_type == "linear_attention"
    mix = block.linear_attn if gdn else block.self_attn
    pre = [mix.in_proj_qkv, mix.in_proj_z] if gdn else [mix.q_proj, mix.k_proj, mix.v_proj]
    out = mix.out_proj if gdn else mix.o_proj
    mlp = block.mlp
    lins = pre + [out, mlp.gate_proj, mlp.up_proj, mlp.down_proj]
    plan = None
    ok = not getattr(mlp, "merged_mlp", False) and all(_static_ok(l) for l in lins)
    if ok:
        cb = lins[0].weight._params.codebook
        ok = all(torch.equal(cb.cpu(), l.weight._params.codebook.cpu()) for l in lins[1:])
    for n in (block.input_layernorm, block.post_attention_layernorm):
        ok = ok and n.weight.dtype == torch.bfloat16 and n.weight.numel() == 4096
    if gdn:
        ok = ok and mix.num_key_heads * mix.key_head_dim == mix.key_dim and mix.value_dim == 4096
        ok = ok and mix.conv1d.bias is None and not getattr(mix.conv1d, "weight_function", None)
        for lin in (mix.in_proj_a, mix.in_proj_b):
            ok = ok and getattr(lin, "bias", None) is None and not getattr(lin, "weight_function", None)
    else:
        ok = ok and mix.num_heads * mix.head_dim == 4096
    if ok:
        plan = {"gdn": gdn, "mix": mix, "lins": lins, "npre": len(pre)}
    return plan


def _norm_weight(norm, device):
    # RMSNorm(add=True) weight + 1 (BF16, as the eager scale()), cached on the compute device
    w = norm.weight
    return tiles.cached(("norm", id(norm), id(w), w.data_ptr(), w.device, device),
                        lambda: comfy_cast(norm.scale(), device).contiguous())


def comfy_cast(t, device):
    import comfy.model_management
    return comfy.model_management.cast_to(t, dtype=t.dtype, device=device)


def _args(block, sp, x, es):
    """Kernel arguments for this call, or None (fall back). Reads each linear's device weight (resident,
    or the DynamicVRAM cast held until `es` exits); uses this generate's tile-major copies when there is
    room for all of them, the row-major tensors otherwise."""
    from .mono import _w4
    from comfy_kitchen.tensor import QuantizedTensor
    lins, npre, mix = sp["lins"], sp["npre"], sp["mix"]
    dev = x.device
    weights = [_cast(es, l, x)[0] for l in lins]
    parts = [_w4(l, w) for l, w in zip(lins, weights)]
    if any(p is None for p in parts):
        return None
    tiled = False
    if tiles.tiles_enabled():
        t = [tiles.tiled(l, w) for l, w in zip(lins, weights)]
        if all(p is not None for p in t):
            parts, tiled = t, True
    a = {"tiled": tiled, "pre": sum(parts[:npre], []), "out": parts[npre], "mlp": sum(parts[npre + 1:], []),
         "norm_in": (_norm_weight(block.input_layernorm, dev), block.input_layernorm.eps),
         "norm_post": (_norm_weight(block.post_attention_layernorm, dev), block.post_attention_layernorm.eps)}
    if sp["gdn"]:
        ws, borrowed = [], set()
        for name, lin in (("wa", mix.in_proj_a), ("wb", mix.in_proj_b)):
            w, cast = _cast(es, lin, x)
            if isinstance(w, QuantizedTensor):  # as the eager GDN decode branch does on every call
                w = (tiles.cached(("bf16", id(lin), w._qdata.data_ptr()), lambda w=w: w.dequantize().contiguous())
                     if not cast else w.dequantize())
            elif cast:
                borrowed.add(name)
            if w.device != dev or w.dtype != torch.bfloat16 or tuple(w.shape) != (mix.num_value_heads, 4096):
                return None
            ws.append(w.contiguous())
        cw, cast = _cast(es, mix.conv1d, x)
        if cw.device != dev or cw.dtype != torch.bfloat16:
            return None
        if cast:
            borrowed.add("conv_w")
        a["extra"] = {"wa": ws[0], "wb": ws[1], "conv_w": cw.contiguous()}
        a["borrowed"] = borrowed
    return a


def _owned(plan):
    """A copy of a tiled plan that outlives this call's DynamicVRAM casts: W4 operands are this
    generate's tile copies already; the GDN BF16 operands borrowed from a cast are copied."""
    plan = dict(plan)
    borrowed = plan.pop("borrowed", ())
    if "extra" in plan:
        plan["extra"] = {k: (v.clone() if k in borrowed else v) for k, v in plan["extra"].items()}
    return plan


def _gdn_core(mix, xn, proj, z, kv):
    # GatedDeltaNet.forward decode branch after the input projections (same Kitchen calls)
    import comfy.ops
    import comfy_kitchen
    from comfy_kitchen.tensor import QuantizedTensor
    with comfy.ops.CastBiasWeightContext(mix.conv1d, proj, offloadable=True) as (conv_weight, conv_bias):
        comfy_kitchen.deltanet_conv_step_deferred(proj, kv.conv_state, conv_weight, conv_bias, kv.proj_buf, kv.qkv_buf, kv.ctl)
    with comfy.ops.CastBiasWeightContext(mix.in_proj_a, xn, offloadable=True) as (w_a, _), \
         comfy.ops.CastBiasWeightContext(mix.in_proj_b, xn, offloadable=True) as (w_b, _):
        if isinstance(w_a, QuantizedTensor):
            w_a = w_a.dequantize()
        if isinstance(w_b, QuantizedTensor):
            w_b = w_b.dequantize()
        return comfy_kitchen.gated_delta_decode_deferred(
            xn, w_a, w_b, kv.dt_bias, kv.g_decay, kv.recurrent_state,
            mix.key_dim, mix.num_key_heads, mix.key_head_dim ** -0.5, z, kv.norm_weight, mix.norm.eps,
            kv.qkv_buf, kv.gates_buf, kv.sumsq_buf, kv.ctl)


def _rope_eligible(mix, qn, kn, xv4, freqs, kv):
    from comfy.text_encoders import llama
    if (not _rope_fused or guard.off("rope") or _rope_state["ok"] is False or type(kv) is not llama.FixedKVCache
            or qn.shape[0] != 1):
        return False
    half = mix.rotary_dim // 2
    return (qn.device.type == "xpu" and qn.dtype == torch.bfloat16 and qn.is_contiguous() and kn.is_contiguous()
            and xv4.is_contiguous() and kv.key.dtype == torch.bfloat16 and kv.key.is_contiguous()
            and kv.value.is_contiguous() and kv.position.dtype == torch.long and torch.is_tensor(freqs)
            and freqs.dtype == torch.float32 and freqs.dim() >= 4 and tuple(freqs.shape[-3:]) == (half, 2, 2)
            and freqs.numel() == qn.shape[1] * half * 4)


def _same(a, b):
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def _norm_rope_eligible(mix, rawq, rawk, xv4, freqs, kv):
    from comfy.text_encoders import llama
    from torch.nn.modules import module
    if (not _rope_fused or guard.off("norm_rope") or torch.is_grad_enabled()
            or type(kv) is not llama.FixedKVCache or rawq.shape[0] != 1 or mix.head_dim != 256):
        return False
    if any((module._global_forward_hooks, module._global_forward_pre_hooks,
            module._global_backward_hooks, module._global_backward_pre_hooks)):
        return False
    for norm in (mix.q_norm, mix.k_norm):
        if (type(norm) is not llama.RMSNorm or "forward" in vars(norm)
                or norm._forward_hooks or norm._forward_pre_hooks or norm._backward_hooks
                or norm.weight.dtype != torch.bfloat16 or norm.weight.numel() != 256
                or type(norm).forward.__module__ != "comfy.text_encoders.llama"
                or getattr(type(norm).forward, "__wrapped__", None) is not None):
            return False
    half = mix.rotary_dim // 2
    return (rawq.device.type == "xpu" and rawq.dtype == torch.bfloat16
            and rawq.stride(-1) == 1 and rawq.stride(-2) == 512
            and rawk.is_contiguous() and xv4.stride(-1) == 1 and xv4.stride(-2) == 256
            and kv.key.dtype == torch.bfloat16 and kv.key.is_contiguous() and kv.value.is_contiguous()
            and torch.is_tensor(freqs) and freqs.dtype == torch.float32
            and freqs.dim() >= 4 and tuple(freqs.shape[-3:]) == (half, 2, 2)
            and freqs.numel() == rawq.shape[1] * half * 4)


def _norm_rope_write(mix, rawq, rawk, xv4, freqs, kv):
    from comfy.text_encoders import qwen35 as q35
    from . import _C
    S = rawq.shape[1]
    checked = tiles.cached(("norm_rope_check", rawq.device, rawq.shape[2], rawk.shape[2],
                            mix.rotary_dim, mix.q_norm.eps, mix.k_norm.eps,
                            id(q35.apply_partial_rope)), lambda: {"ok": False})
    reference = None
    if not checked["ok"]:
        qn, kn = mix.q_norm(rawq), mix.k_norm(rawk)
        reference = q35.apply_partial_rope(qn.transpose(1, 2), kn.transpose(1, 2), freqs, mix.rotary_dim)
    try:
        q, k = _C.qk_norm_rope_kv(
            rawq, rawk, xv4, _norm_weight(mix.q_norm, rawq.device), _norm_weight(mix.k_norm, rawq.device),
            freqs.reshape(S, -1, 2, 2), kv.key, kv.value, kv.position[:S],
            mix.q_norm.eps, mix.k_norm.eps, mix.rotary_dim)
    except Exception as exc:
        # The native call can have written KV before reporting an error: do not replay.
        guard.failed("norm_rope", exc, can_fallback=False)
        raise SubpathError("norm_rope") from exc
    if reference is not None:
        if not (_same(q, reference[0]) and _same(k, reference[1])):
            # Another Torch/Omni reduction order (e.g. a different Torch build): rewrite the slots
            # from the live operator chain, keep its results and stop using the fused path.
            guard.failed("norm_rope", RuntimeError("fused Q/K RMSNorm/RoPE differs from the active operator chain"))
            _fallback("norm_rope_check")
            kv.write(reference[1], xv4.transpose(1, 2))
            return reference
        checked["ok"] = True
    with _lock:
        _stats["norm_rope"] = _stats.get("norm_rope", 0) + 1
        _stats["rope"] = _stats.get("rope", 0) + 1
    return q, k


def _rope_selfcheck(device, rd, hq, hk, n=150):
    """Wide-range synthetic rows (rounding differences between rope formulas show up within a few
    dozen) through the fused kernel and the live apply_partial_rope; True if all bits agree."""
    from comfy.text_encoders import qwen35 as q35
    from . import _C
    g = torch.Generator(device="cpu").manual_seed(1234)
    D = 256
    table = torch.randn(1, 1, n * 8, rd // 2, 2, 2, generator=g).to(device)
    key = torch.zeros(1, hk, 8, D, dtype=torch.bfloat16, device=device)
    val = torch.zeros_like(key)
    pos = torch.arange(8, device=device)
    for t in range(n):
        qn = (torch.randn(1, 8, hq, D, generator=g) * torch.exp(torch.randn(1, 8, hq, D, generator=g) * 2))
        kn = torch.randn(1, 8, hk, D, generator=g) * 30
        qn, kn = qn.to(torch.bfloat16).to(device), kn.to(torch.bfloat16).to(device)
        f = table[:, :, t * 8:(t + 1) * 8]
        q, k = _C.rope_kv(qn[0], kn[0], kn[0], f.reshape(8, -1, 2, 2), key, val, pos, rd)
        rq, rk = q35.apply_partial_rope(qn.transpose(1, 2), kn.transpose(1, 2), f, rd)
        if not (_same(q, rq) and _same(k, rk)):
            return False
    return True


def _rope_write(mix, qn, kn, xv4, freqs, kv):
    """q35.apply_partial_rope + FixedKVCache.write in one launch. The first call is checked against the
    live path (whichever kitchen backend serves the rope); a mismatch turns the fused path off."""
    from comfy.text_encoders import qwen35 as q35
    from . import _C
    S = qn.shape[1]
    try:
        q, k = _C.rope_kv(qn[0], kn[0], xv4[0], freqs.reshape(S, -1, 2, 2), kv.key, kv.value, kv.position[:S],
                          mix.rotary_dim)
        if _rope_state["ok"] is None:
            rq, rk = q35.apply_partial_rope(qn.transpose(1, 2), kn.transpose(1, 2), freqs, mix.rotary_dim)
            ok = _same(q, rq) and _same(k, rk) and _rope_selfcheck(qn.device, mix.rotary_dim, qn.shape[2], kn.shape[2])
            _rope_state["ok"] = ok
            if not ok:  # rewrite the slots the eager way and keep the eager results
                _fallback("rope_check")
                kv.write(rk, xv4.transpose(1, 2))
                return rq, rk
    except Exception as e:
        # The native call can have written KV before reporting an error.
        # This is different from the successful numerical self-check above.
        guard.failed("rope", e, can_fallback=False)
        raise SubpathError("rope") from e
    with _lock:
        _stats["rope"] = _stats.get("rope", 0) + 1
    return q, k


def _attn_core(mix, qg, xk, xv, attention_mask, freqs_cis, optimized_attention, past_key_value):
    # GatedAttention.forward after the projections, up to (not including) the sigmoid gate
    import comfy.model_management
    from comfy.text_encoders import qwen35 as q35
    from comfy.text_encoders.llama import SPEC_ROWS, Int8FixedKVCache
    B, S = qg.shape[0], qg.shape[1]
    qg = qg.view(B, S, mix.num_heads, mix.head_dim * 2)
    rawq = qg[..., :mix.head_dim]
    rawk = xk.view(B, S, mix.num_kv_heads, mix.head_dim)
    xv4 = xv.view(B, S, mix.num_kv_heads, mix.head_dim)
    xv = xv4.transpose(1, 2)
    decode = past_key_value is not None and S <= SPEC_ROWS and attention_mask is None
    if decode and _norm_rope_eligible(mix, rawq, rawk, xv4, freqs_cis, past_key_value):
        xq, xk = _norm_rope_write(mix, rawq, rawk, xv4, freqs_cis, past_key_value)
    else:
        qn, kn = mix.q_norm(rawq), mix.k_norm(rawk)
        if decode and _rope_eligible(mix, qn, kn, xv4, freqs_cis, past_key_value):
            xq, xk = _rope_write(mix, qn, kn, xv4, freqs_cis, past_key_value)
        else:
            xq, xk = q35.apply_partial_rope(qn.transpose(1, 2), kn.transpose(1, 2), freqs_cis, mix.rotary_dim)
            if decode:
                past_key_value.write(xk, xv)
    if decode and (isinstance(past_key_value, Int8FixedKVCache) or
                   mix.comfy_attention.method is None and not comfy.model_management.comfy_kitchen_attention_enabled()):
        return past_key_value.attend(xq, xk, xv)
    if decode:
        xk, xv = past_key_value.key, past_key_value.value
        attention_mask = past_key_value.mask(S)
        optimized_attention = q35.optimized_attention_for_device(xq.device)
    elif past_key_value is not None:
        xk, xv = past_key_value.append(xk, xv)
    gqa_kwargs = {"enable_gqa": True} if mix.num_heads != mix.num_kv_heads else {}
    q, k, v = map(q35.AttentionTensorContainer, (xq, xk, xv))
    del xq, xk, xv
    return optimized_attention(q, k, v, mix.num_heads, mask=attention_mask, skip_reshape=True,
                               preferred_attention=mix.comfy_attention, **gqa_kwargs)


def _install_attend():
    """FixedKVCache.attend -> one bit-identical launch (tree verify rows, or the single MTP draft row)."""
    from comfy.text_encoders import llama
    from . import _C
    cls = llama.FixedKVCache
    original = cls.attend
    if getattr(original, CLAIM, False):
        return False
    empty = {}

    def attend(self, xq, xk, xv):
        tree = self.tracker.get("tree") if isinstance(self.tracker, dict) else None
        S = xq.shape[2] if xq.dim() == 4 else 0
        if (_attn_port and not guard.off("attn") and xq.device.type == "xpu" and xq.dtype == torch.bfloat16 and xq.shape[0] == 1
                and xq.shape[-1] == 256 and self.key.dtype == torch.bfloat16 and self.key.is_contiguous()
                and self.value.is_contiguous() and 1 <= S <= 8 and self.seqlen.dtype == torch.int32):
            if tree is not None and tree.mask.numel() == S and tree.mask.dtype == torch.int32:
                mask = tree.mask
            elif tree is None and S == 1:
                mask = empty.get(xq.device)
                if mask is None:
                    mask = empty[xq.device] = torch.empty(0, dtype=torch.int32, device=xq.device)
            else:
                mask = None
            if mask is not None:
                # tree verify rows: DPAS attention split along the context (not bit-identical to
                # Kitchen; see attn_dpas.sycl); the draft's single row keeps the exact port
                dpas = _attn_dpas and tree is not None and not guard.off("attn_dpas")
                try:
                    if dpas:
                        out = _C.attn_dpas(xq, self.key, self.value, self.seqlen, xk, xv, mask, 0)
                    else:
                        out = _C.attn_core(xq, self.key, self.value, self.seqlen, xk, xv, mask, 0)
                except Exception as e:  # reads the cache only: the original recomputes it
                    guard.failed("attn_dpas" if dpas else "attn", e)
                    return original(self, xq, xk, xv)
                with _lock:
                    _stats["attn_dpas" if dpas else "attn"] = _stats.get("attn_dpas" if dpas else "attn", 0) + 1
                return out
        return original(self, xq, xk, xv)

    setattr(attend, CLAIM, True)
    attend.__wrapped__ = original
    cls.attend = attend
    return True


def _install_gated_attention():
    """GatedAttention.forward for decode rows whose block runs eagerly (the BF16 MTP draft block): same
    projections and gate, the middle through _attn_core (fused rope + KV write, attention port)."""
    from comfy.text_encoders import qwen35 as q35
    from comfy.text_encoders.llama import SPEC_ROWS
    cls = q35.GatedAttention
    original = cls.forward
    if getattr(original, CLAIM, False):
        return False

    def forward(self, x, attention_mask=None, freqs_cis=None, optimized_attention=None, past_key_value=None):
        if (x.dim() != 3 or x.shape[0] != 1 or not 1 <= x.shape[1] <= SPEC_ROWS or past_key_value is None
                or attention_mask is not None or x.device.type != "xpu" or x.dtype != torch.bfloat16):
            return original(self, x, attention_mask, freqs_cis, optimized_attention, past_key_value)
        B, S = x.shape[0], x.shape[1]
        combined = getattr(self, "combined_qkv", None)
        qkv = combined.forward(x) if combined is not None else NotImplemented
        if qkv is NotImplemented:
            qg, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        else:
            kv_size = self.num_kv_heads * self.head_dim
            qg, xk, xv = qkv.split((self.num_heads * self.head_dim * 2, kv_size, kv_size), dim=-1)
        gate = qg.view(B, S, self.num_heads, self.head_dim * 2)[..., self.head_dim:].reshape(B, S, -1)
        output = _attn_core(self, qg, xk, xv, attention_mask, freqs_cis, optimized_attention, past_key_value)
        output = output * gate.sigmoid()
        return self.o_proj(output), past_key_value

    setattr(forward, CLAIM, True)
    forward.__wrapped__ = original
    cls.forward = forward
    return True


def _install_norm_scale():
    """RMSNorm(add=True).scale() -> weight + 1 cached per weight (same BF16 values, no launch per call)."""
    from comfy.text_encoders import llama
    cls = llama.RMSNorm
    original = cls.scale
    if getattr(original, CLAIM, False):
        return False

    def scale(self):
        w = self.weight
        if not self.add or w.device.type != "xpu" or (torch.is_grad_enabled() and w.requires_grad):
            return original(self)
        return tiles.cached(("scale", id(self), id(w), w.data_ptr(), w.device), lambda: original(self))

    setattr(scale, CLAIM, True)
    scale.__wrapped__ = original
    cls.scale = scale
    return True


def install():
    from comfy.text_encoders import qwen35 as q35
    _install_attend()
    tiles.install()
    if _rope_fused:
        _install_gated_attention()
    if os.environ.get("MONO_DECODE_XPU_NORM_CACHE", "1") == "1":
        _install_norm_scale()
    if os.environ.get("MONO_DECODE_XPU_LOGITS", "1") == "1":
        from . import logits
        logits.install()
    if os.environ.get("MONO_DECODE_XPU_AUTOTREE", "1") == "1":
        from . import autotree
        autotree.install()
    if os.environ.get("MONO_DECODE_XPU_DRAFT", "1") == "1":
        from . import draft
        draft.install()
    from comfy.text_encoders.llama import SPEC_ROWS
    from . import _C

    Block = q35.Qwen35TransformerBlock
    original = Block.forward
    if getattr(original, CLAIM, False):
        return False

    def forward(self, x, attention_mask=None, freqs_cis=None, optimized_attention=None, past_key_value=None):
        B, S = (x.shape[0], x.shape[1]) if x.dim() == 3 else (0, 0)
        if (guard.off("layer") or past_key_value is None or x.device.type != "xpu" or x.dtype != torch.bfloat16 or B != 1
                or not 1 <= S <= 8 or x.shape[-1] != 4096 or not x.is_contiguous()):
            return original(self, x, attention_mask, freqs_cis, optimized_attention, past_key_value)
        gdn = self.layer_type == "linear_attention"
        if gdn and not past_key_value.decode_step(S):
            return original(self, x, attention_mask, freqs_cis, optimized_attention, past_key_value)
        if not gdn and not (S <= SPEC_ROWS and attention_mask is None):
            return original(self, x, attention_mask, freqs_cis, optimized_attention, past_key_value)
        sp = _static_plan(self)
        if sp is None:
            _fallback("weights")
            return original(self, x, attention_mask, freqs_cis, optimized_attention, past_key_value)
        state = {"mutated": False}  # set once a launch has updated layer state in place (GDN state, x)
        try:
            # Tiled plans read only this generate's own copies: reuse them without recasting weights.
            plan = tiles.plan(self)
            if plan is not None:
                return _run(self, plan, x, S, gdn, attention_mask, freqs_cis, optimized_attention,
                            past_key_value, state)
            with contextlib.ExitStack() as es:
                plan = _args(self, sp, x, es)
                if plan is not None:
                    if plan["tiled"]:
                        try:  # an optimization only: without it, later calls prepare their arguments again
                            tiles.set_plan(self, _owned(plan))
                        except (torch.OutOfMemoryError, RuntimeError):
                            pass
                    return _run(self, plan, x, S, gdn, attention_mask, freqs_cis, optimized_attention,
                                past_key_value, state)
        except SubpathError:
            raise  # that sub-path is already off; KV may be written, so the block is not replayed
        except Exception as e:
            guard.failed("layer", e, can_fallback=not state["mutated"])
            if state["mutated"]:
                raise  # the recurrent state or x already changed: replaying the block would be wrong
            return original(self, x, attention_mask, freqs_cis, optimized_attention, past_key_value)
        _fallback("weights")
        return original(self, x, attention_mask, freqs_cis, optimized_attention, past_key_value)

    def _run(self, plan, x, S, gdn, attention_mask, freqs_cis, optimized_attention, past_key_value, state):
        tiled = plan["tiled"]
        h, silu, sig = _constants(x.device)
        x2 = x.view(S, 4096)
        if gdn and _single_gdn:
            # the whole GDN block in one launch
            mix, kv, ex = self.linear_attn, past_key_value, plan["extra"]
            (nin, ein), (npo, epo) = plan["norm_in"], plan["norm_post"]
            args = (x2, nin, ein, h, plan["pre"] + plan["out"] + plan["mlp"], ex["wa"], ex["wb"], ex["conv_w"],
                    kv.conv_state, kv.proj_buf, kv.qkv_buf, kv.ctl, kv.dt_bias, kv.g_decay, kv.recurrent_state,
                    kv.gates_buf, kv.sumsq_buf, kv.norm_weight, mix.norm.eps, mix.key_head_dim ** -0.5,
                    mix.num_key_heads, npo, epo, silu, 0, 5 if _gdn_split else 11, tiled)
            # Mark before dispatch: a failure does not prove that nothing ran.
            state["mutated"] = True
            _C.layer_gdn(*args)
            if _gdn_split:  # GDN core in one launch, the post block (out proj .. MLP) in layer_post
                y = _C.layer_gdn_core(x2, S)
                _C.layer_post(x2, y, None, sig, *plan["out"], npo, epo, h, silu, *plan["mlp"], 0, 6, tiled)
            with _lock:
                _stats["layer"] += 1
                _stats["gdn1"] = _stats.get("gdn1", 0) + 1
            return x, past_key_value
        nw, eps = plan["norm_in"]
        outs = _C.layer_pre(x2, nw, eps, h, plan["pre"], 0, tiled)
        gate = None
        if gdn:
            mix = self.linear_attn
            xn = outs[0].view(1, S, 4096)
            state["mutated"] = True  # the Kitchen deferred GDN calls advance the state in place
            core = _gdn_core(mix, xn, outs[1].view(1, S, -1), outs[2].view(1, S, -1), past_key_value)
            y = core.reshape(S, 4096)
        else:
            mix = self.self_attn
            state["mutated"] = True  # attention writes KV before attending
            y = _attn_core(mix, outs[1].view(1, S, -1), outs[2].view(1, S, -1), outs[3].view(1, S, -1),
                           attention_mask, freqs_cis, optimized_attention, past_key_value).reshape(S, 4096)
            gate = outs[1]
        nw, eps = plan["norm_post"]
        state["mutated"] = True  # layer_post updates x in place
        _C.layer_post(x2, y.contiguous(), gate, sig, *plan["out"], nw, eps, h, silu, *plan["mlp"], 0, 6, tiled)
        with _lock:
            _stats["layer"] += 1
        return x, past_key_value

    setattr(forward, CLAIM, True)
    forward.__wrapped__ = original
    Block.forward = forward
    return True
