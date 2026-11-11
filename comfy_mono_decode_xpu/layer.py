"""Route Qwen3.5 decode transformer blocks (1-8 verify rows) through the persistent layer segments.

A block is ``x += mixer(input_layernorm(x)); x += mlp(post_attention_layernorm(x))``. For decode rows
whose projections are all packed W4A8 (G16, FP8 group scales, ConvRot 256, one codebook):

    layer_pre   input RMSNorm + rotate/quantize + the mixer's input projections      (1 launch)
    mixer core  GDN: Kitchen deferred conv + gated-delta kernels (unchanged calls);
                attention: q/k norm, partial RoPE, KV write and decode attention (unchanged code)
    layer_post  [sigmoid gate] + out projection + residual + post RMSNorm + MLP + residual (1 launch)

Every segment is bit-identical to the eager kernels, so tokens do not change. Any other call
(prefill, BF16 MTP head, offloaded or differently quantized weights) runs the original forward.
"""
from __future__ import annotations

import threading

import torch
import torch.nn.functional as F

CLAIM = "__comfy_mono_decode_xpu_layer__"
_stats = {"layer": 0, "fallback": {}}
_lock = threading.Lock()
_consts = {}
_single_gdn = __import__("os").environ.get("CMDX_GDN1", "1") == "1"
_attn_port = __import__("os").environ.get("CMDX_ATTN", "1") == "1"


def stats():
    with _lock:
        return {"layer": _stats["layer"], "gdn1": _stats.get("gdn1", 0), "attn": _stats.get("attn", 0),
                "fallback": dict(_stats["fallback"])}


def _fallback(why):
    with _lock:
        _stats["fallback"][why] = _stats["fallback"].get(why, 0) + 1


def _constants(device):
    c = _consts.get(device)
    if c is None:
        from comfy_kitchen.tensor.int8_utils import _build_hadamard
        h = _build_hadamard(256, device=device, dtype=torch.bfloat16).contiguous()
        allbf = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).to(device)
        # torch's own SiLU / sigmoid for every BF16 input
        c = _consts[device] = (h, F.silu(allbf).contiguous(), torch.sigmoid(allbf).contiguous())
    return c


def _w4(lin):
    from .mono import _w4 as w4
    return w4(lin)


def _plan(block):
    """Per-block cached kernel arguments, or None when the block is not eligible."""
    cached = getattr(block, "_cmdx_plan", None)
    if cached is not None:
        key, plan = cached
        if key == _key(block):
            return plan
    plan = _build_plan(block)
    block._cmdx_plan = (_key(block), plan)
    return plan


def _key(block):
    mlp = block.mlp
    return tuple(getattr(getattr(l, "weight", None), "_qdata", None) is not None and l.weight._qdata.data_ptr()
                 for l in (mlp.gate_proj, mlp.up_proj, mlp.down_proj))


def _build_plan(block):
    gdn = block.layer_type == "linear_attention"
    mix = block.linear_attn if gdn else block.self_attn
    pre = [mix.in_proj_qkv, mix.in_proj_z] if gdn else [mix.q_proj, mix.k_proj, mix.v_proj]
    out = mix.out_proj if gdn else mix.o_proj
    mlp = block.mlp
    if getattr(mlp, "merged_mlp", False):
        return None
    parts = [_w4(l) for l in pre + [out, mlp.gate_proj, mlp.up_proj, mlp.down_proj]]
    if any(p is None for p in parts):
        return None
    cb = parts[0][3]
    if not all(torch.equal(cb, p[3]) for p in parts[1:]):
        return None
    for n in (block.input_layernorm, block.post_attention_layernorm):
        if n.weight.device.type != "xpu" or n.weight.dtype != torch.bfloat16 or n.weight.numel() != 4096:
            return None
    if gdn and not (mix.num_key_heads * mix.key_head_dim == mix.key_dim and mix.value_dim == 4096):
        return None
    if not gdn and mix.num_heads * mix.head_dim != 4096:
        return None
    extra = None
    if gdn:
        from comfy_kitchen.tensor import QuantizedTensor
        ws = []
        for lin in (mix.in_proj_a, mix.in_proj_b):
            w = lin.weight
            if getattr(lin, "weight_function", None) or getattr(lin, "bias", None) is not None or w.device.type != "xpu":
                return None
            w = w.dequantize() if isinstance(w, QuantizedTensor) else w
            if w.dtype != torch.bfloat16 or tuple(w.shape) != (mix.num_value_heads, 4096):
                return None
            ws.append(w.contiguous())
        cw = mix.conv1d.weight
        if cw.device.type != "xpu" or cw.dtype != torch.bfloat16 or mix.conv1d.bias is not None or getattr(mix.conv1d, "weight_function", None):
            return None
        extra = {"wa": ws[0], "wb": ws[1], "conv_w": cw.contiguous()}
    return {
        "gdn": gdn,
        "extra": extra,
        "pre": sum(parts[:len(pre)], []),
        "out": parts[len(pre)],
        "mlp": sum(parts[len(pre) + 1:], []),
        "norm_in": (block.input_layernorm.scale().contiguous(), block.input_layernorm.eps),
        "norm_post": (block.post_attention_layernorm.scale().contiguous(), block.post_attention_layernorm.eps),
    }


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


def _attn_core(mix, qg, xk, xv, attention_mask, freqs_cis, optimized_attention, past_key_value):
    # GatedAttention.forward after the projections, up to (not including) the sigmoid gate
    import comfy.model_management
    from comfy.text_encoders import qwen35 as q35
    from comfy.text_encoders.llama import SPEC_ROWS, Int8FixedKVCache
    B, S = qg.shape[0], qg.shape[1]
    qg = qg.view(B, S, mix.num_heads, mix.head_dim * 2)
    xq = qg[..., :mix.head_dim]
    xq = mix.q_norm(xq).transpose(1, 2)
    xk = mix.k_norm(xk.view(B, S, mix.num_kv_heads, mix.head_dim)).transpose(1, 2)
    xv = xv.view(B, S, mix.num_kv_heads, mix.head_dim).transpose(1, 2)
    xq, xk = q35.apply_partial_rope(xq, xk, freqs_cis, mix.rotary_dim)
    decode = past_key_value is not None and S <= SPEC_ROWS and attention_mask is None
    if decode:
        past_key_value.write(xk, xv)
    if decode and (isinstance(past_key_value, Int8FixedKVCache) or
                   mix.comfy_attention.method is None and not comfy.model_management.comfy_kitchen_attention_enabled()):
        tree = getattr(past_key_value, "tracker", {}).get("tree")
        if (_attn_port and tree is not None and type(past_key_value).__name__ == "FixedKVCache" and B == 1
                and xq.shape[-1] == 256 and past_key_value.key.is_contiguous() and past_key_value.value.is_contiguous()
                and past_key_value.key.dtype == torch.bfloat16 and tree.mask.numel() == S):
            # Kitchen's tree-verify decode attention (decode_gqa + tree_merge), bit-identical, one launch
            from . import _C
            with _lock:
                _stats["attn"] = _stats.get("attn", 0) + 1
            return _C.attn_core(xq, past_key_value.key, past_key_value.value, past_key_value.seqlen, xk, xv,
                                tree.mask.to(torch.int32), 0)
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


def install():
    from comfy.text_encoders import qwen35 as q35
    from comfy.text_encoders.llama import SPEC_ROWS
    from . import _C

    Block = q35.Qwen35TransformerBlock
    original = Block.forward
    if getattr(original, CLAIM, False):
        return False

    def forward(self, x, attention_mask=None, freqs_cis=None, optimized_attention=None, past_key_value=None):
        B, S = (x.shape[0], x.shape[1]) if x.dim() == 3 else (0, 0)
        if (past_key_value is None or x.device.type != "xpu" or x.dtype != torch.bfloat16 or B != 1
                or not 1 <= S <= 8 or x.shape[-1] != 4096 or not x.is_contiguous()):
            return original(self, x, attention_mask, freqs_cis, optimized_attention, past_key_value)
        gdn = self.layer_type == "linear_attention"
        if gdn and not past_key_value.decode_step(S):
            return original(self, x, attention_mask, freqs_cis, optimized_attention, past_key_value)
        if not gdn and not (S <= SPEC_ROWS and attention_mask is None):
            return original(self, x, attention_mask, freqs_cis, optimized_attention, past_key_value)
        plan = _plan(self)
        if plan is None:
            _fallback("weights")
            return original(self, x, attention_mask, freqs_cis, optimized_attention, past_key_value)
        h, silu, sig = _constants(x.device)
        x2 = x.view(S, 4096)
        if gdn and _single_gdn:
            # the whole GDN block in one launch
            mix, kv, ex = self.linear_attn, past_key_value, plan["extra"]
            (nin, ein), (npo, epo) = plan["norm_in"], plan["norm_post"]
            _C.layer_gdn(x2, nin, ein, h, plan["pre"] + plan["out"] + plan["mlp"], ex["wa"], ex["wb"], ex["conv_w"],
                         kv.conv_state, kv.proj_buf, kv.qkv_buf, kv.ctl, kv.dt_bias, kv.g_decay, kv.recurrent_state,
                         kv.gates_buf, kv.sumsq_buf, kv.norm_weight, mix.norm.eps, mix.key_head_dim ** -0.5,
                         mix.num_key_heads, npo, epo, silu, 0, 11)
            with _lock:
                _stats["layer"] += 1
                _stats["gdn1"] = _stats.get("gdn1", 0) + 1
            return x, past_key_value
        nw, eps = plan["norm_in"]
        outs = _C.layer_pre(x2, nw, eps, h, plan["pre"], 0)
        gate = None
        if gdn:
            mix = self.linear_attn
            xn = outs[0].view(1, S, 4096)
            core = _gdn_core(mix, xn, outs[1].view(1, S, -1), outs[2].view(1, S, -1), past_key_value)
            y = core.reshape(S, 4096)
        else:
            mix = self.self_attn
            y = _attn_core(mix, outs[1].view(1, S, -1), outs[2].view(1, S, -1), outs[3].view(1, S, -1),
                           attention_mask, freqs_cis, optimized_attention, past_key_value).reshape(S, 4096)
            gate = outs[1]
        nw, eps = plan["norm_post"]
        _C.layer_post(x2, y.contiguous(), gate, sig, *plan["out"], nw, eps, h, silu, *plan["mlp"], 0, 6)
        with _lock:
            _stats["layer"] += 1
        return x, past_key_value

    setattr(forward, CLAIM, True)
    forward.__wrapped__ = original
    Block.forward = forward
    return True
