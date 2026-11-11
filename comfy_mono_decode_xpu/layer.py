# SPDX-License-Identifier: Apache-2.0
"""Route Qwen3.5 decode transformer blocks (1-8 verify rows) through the persistent layer kernels.

A block is ``x += mixer(input_layernorm(x)); x += mlp(post_attention_layernorm(x))``. For decode rows
whose projections are all packed W4A8 (G16, FP8 group scales, ConvRot 256, one codebook):

    Gated-DeltaNet layers (24 of 32)
        layer_gdn   input RMSNorm .. projections .. deferred conv, gates, delta rule, gated norm (1 launch)
        layer_post  out projection + residual + post RMSNorm + MLP + residual             (1 launch)
    full-attention layers (8 of 32)
        layer_pre   input RMSNorm + rotate/quantize + q/k/v projections                   (1 launch)
        q/k RMSNorm (eager), rope_kv (partial RoPE + KV write), attn_core (attention port)
        layer_post  sigmoid gate + out projection + residual + post RMSNorm + MLP + residual

Also installed here: the attention port for FixedKVCache.attend, the BF16 MTP draft block's attention
through the same rope_kv/attn_core path, RMSNorm scale caching, the fast logits picks (logits.py) and
per-generate tile-major weight copies under DynamicVRAM.

Every launch is bit-identical to the eager kernels it replaces, so tokens do not change. Any other
call (prefill, offloaded or differently quantized weights) runs the original forward.
"""
from __future__ import annotations

import contextlib
import threading

import torch
import torch.nn.functional as F

CLAIM = "__comfy_mono_decode_xpu_layer__"
_stats = {"layer": 0, "fallback": {}}
_lock = threading.Lock()
_consts = {}
_single_gdn = __import__("os").environ.get("MONO_DECODE_XPU_GDN1", "1") == "1"
_attn_port = __import__("os").environ.get("MONO_DECODE_XPU_ATTN", "1") == "1"
_gdn_split = __import__("os").environ.get("MONO_DECODE_XPU_GDN_SPLIT", "1") == "1"
_tiled_ok = __import__("os").environ.get("MONO_DECODE_XPU_TILED", "1") == "1"  # 0: row-major even when resident (A/B)
_rope_fused = __import__("os").environ.get("MONO_DECODE_XPU_ROPE", "1") == "1"
_rope_state = {"ok": None}  # fused rope + KV write: None until the one-time check against the live path


def stats():
    with _lock:
        return {"layer": _stats["layer"], "gdn1": _stats.get("gdn1", 0), "attn": _stats.get("attn", 0),
                "session_tile_bytes": _stats.get("session_tile_bytes", 0), "rope": _stats.get("rope", 0), "fallback": dict(_stats["fallback"])}


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


def _vbar(mod, device):
    """DynamicVRAM-managed module whose weight lives off-device between uses."""
    return hasattr(mod, "_v") and getattr(mod, "weight", None) is not None and mod.weight.device != device


def _static_plan(block):
    """Per-block structural checks, done once (they read only host-side module state)."""
    cached = getattr(block, "_mono_decode_xpu_static", None)
    if cached is not None:
        return cached[0]
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
    block._mono_decode_xpu_static = (plan,)
    return plan


def _norm_weight(norm, device):
    # RMSNorm(add=True) weight + 1 (BF16, as the eager scale()), cached on the compute device
    w = norm.weight
    cached = getattr(norm, "_mono_decode_xpu_w1", None)
    key = (w.data_ptr(), w.device, device)
    if cached is None or cached[0] != key:
        cached = (key, comfy_cast(norm.scale(), device).contiguous())
        norm._mono_decode_xpu_w1 = cached
    return cached[1]


def comfy_cast(t, device):
    import comfy.model_management
    return comfy.model_management.cast_to(t, dtype=t.dtype, device=device)


def _cast(es, mod, x):
    """The module weight for this call: resident tensor, or the DynamicVRAM cast (held until es exits)."""
    import comfy.ops
    if _vbar(mod, x.device):
        w, _ = es.enter_context(comfy.ops.CastBiasWeightContext(mod, x, offloadable=True, compute_dtype=x.dtype))
        return w, True
    return mod.weight, False


# DynamicVRAM: tile-major copies live only for one generate (built on first use, dropped at the end),
# and only when the device has room for them; otherwise the engine reads the row-major cast weights.
_session = {"active": False, "tiles": {}, "ok": False, "bytes": 0}
_session_margin = int(float(__import__("os").environ.get("MONO_DECODE_XPU_TILE_MARGIN_GB", "3")) * (1 << 30))


def session_begin(device):
    _session["active"] = True
    _session["tiles"] = {}
    _session["bytes"] = 0
    ok = False
    if _tiled_ok and __import__("os").environ.get("MONO_DECODE_XPU_SESSION_TILES", "1") == "1":
        try:
            free, _total = torch.xpu.mem_get_info(device)
            ok = free > 5 * (1 << 30) + _session_margin  # W4 weights + scales of the 9B text model
        except Exception:
            ok = False
    _session["ok"] = ok


def session_end():
    had = bool(_session["tiles"])
    _session.update(active=False, tiles={}, ok=False, bytes=0)
    if had:
        torch.xpu.empty_cache()


def _session_tiles(lin, w):
    from .mono import tile_major
    q, p = w._qdata, w._params
    key = (id(lin), q.data_ptr())  # a module re-faulted elsewhere, or another module at this address: new copy
    hit = _session["tiles"].get(key)
    if hit is None or hit[1][0].shape != q.shape:
        parts = [tile_major(q, 16), tile_major(p.scale.view(torch.uint8), 16).view(torch.float8_e4m3fn),
                 p.s_channel.float().contiguous(), p.codebook.float().contiguous()]
        hit = (q, parts)
        _session["tiles"][key] = hit
        _session["bytes"] += q.numel() + p.scale.numel()
        _stats["session_tile_bytes"] = _stats.get("session_tile_bytes", 0) + q.numel() + p.scale.numel()
    return hit[1]


def _args(block, sp, x, es):
    """Kernel arguments for this call, or None (fall back)."""
    from .mono import _w4
    from comfy_kitchen.tensor import QuantizedTensor
    lins, npre, mix = sp["lins"], sp["npre"], sp["mix"]
    dev = x.device
    tiled = _tiled_ok and not any(_vbar(l, dev) for l in lins)
    if tiled:
        parts = [_w4(l) for l in lins]  # cached tile-major copies of resident weights
    else:
        use_tiles = _session["active"] and _session["ok"]
        parts = []
        for l in lins:
            w, _ = _cast(es, l, x)
            p = _w4(l, w)
            if p is not None and use_tiles:
                p = _session_tiles(l, w)
            parts.append(p)
        tiled = use_tiles
    if any(p is None for p in parts):
        return None
    a = {"tiled": tiled, "pre": sum(parts[:npre], []), "out": parts[npre], "mlp": sum(parts[npre + 1:], []),
         "norm_in": (_norm_weight(block.input_layernorm, dev), block.input_layernorm.eps),
         "norm_post": (_norm_weight(block.post_attention_layernorm, dev), block.post_attention_layernorm.eps)}
    if sp["gdn"]:
        ws = []
        for lin in (mix.in_proj_a, mix.in_proj_b):
            w, cast = _cast(es, lin, x)
            if not cast:
                cached = getattr(lin, "_mono_decode_xpu_bf16", None)
                if cached is None or cached[0] is not w:
                    cached = (w, (w.dequantize() if isinstance(w, QuantizedTensor) else w).contiguous())
                    lin._mono_decode_xpu_bf16 = cached
                w = cached[1]
            elif isinstance(w, QuantizedTensor):
                w = w.dequantize()  # as the eager GDN decode branch does on every call
            if w.device != dev or w.dtype != torch.bfloat16 or tuple(w.shape) != (mix.num_value_heads, 4096):
                return None
            ws.append(w.contiguous())
        cw, _ = _cast(es, mix.conv1d, x)
        if cw.device != dev or cw.dtype != torch.bfloat16:
            return None
        a["extra"] = {"wa": ws[0], "wb": ws[1], "conv_w": cw.contiguous()}
    return a


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
    if not _rope_fused or _rope_state["ok"] is False or type(kv) is not llama.FixedKVCache or qn.shape[0] != 1:
        return False
    half = mix.rotary_dim // 2
    return (qn.device.type == "xpu" and qn.dtype == torch.bfloat16 and qn.is_contiguous() and kn.is_contiguous()
            and xv4.is_contiguous() and kv.key.dtype == torch.bfloat16 and kv.key.is_contiguous()
            and kv.value.is_contiguous() and kv.position.dtype == torch.long and torch.is_tensor(freqs)
            and freqs.dtype == torch.float32 and freqs.dim() >= 4 and tuple(freqs.shape[-3:]) == (half, 2, 2)
            and freqs.numel() == qn.shape[1] * half * 4)


def _same(a, b):
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


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
    xq = qg[..., :mix.head_dim]
    qn = mix.q_norm(xq)
    kn = mix.k_norm(xk.view(B, S, mix.num_kv_heads, mix.head_dim))
    xv4 = xv.view(B, S, mix.num_kv_heads, mix.head_dim)
    xv = xv4.transpose(1, 2)
    decode = past_key_value is not None and S <= SPEC_ROWS and attention_mask is None
    fused = decode and _rope_eligible(mix, qn, kn, xv4, freqs_cis, past_key_value)
    if fused:
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
        if (_attn_port and xq.device.type == "xpu" and xq.dtype == torch.bfloat16 and xq.shape[0] == 1
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
                with _lock:
                    _stats["attn"] = _stats.get("attn", 0) + 1
                return _C.attn_core(xq, self.key, self.value, self.seqlen, xk, xv, mask, 0)
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
        qg = self.q_proj(x)
        gate = qg.view(B, S, self.num_heads, self.head_dim * 2)[..., self.head_dim:].reshape(B, S, -1)
        xk = self.k_proj(x)
        xv = self.v_proj(x)
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
        if not self.add or w.device.type != "xpu":
            return original(self)
        key = (id(w), w.data_ptr(), w.dtype)
        cached = self.__dict__.get("_mono_decode_xpu_scale")
        if cached is None or cached[0] != key:
            cached = (key, original(self))
            self.__dict__["_mono_decode_xpu_scale"] = cached
        return cached[1]

    setattr(scale, CLAIM, True)
    scale.__wrapped__ = original
    cls.scale = scale
    return True


def _install_session():
    from comfy.text_encoders import qwen35 as q35, llama
    targets = [(q35.Qwen35, "_generate_mtp"), (llama.BaseGenerate, "generate")]
    for cls, name in targets:
        original = getattr(cls, name)
        if getattr(original, CLAIM, False):
            continue

        def wrapped(self, *a, _original=original, **k):
            if _session["active"]:
                return _original(self, *a, **k)
            try:
                dev = next(self.parameters()).device
                session_begin(dev if dev.type == "xpu" else torch.device("xpu"))
            except Exception:
                session_begin(torch.device("xpu"))
            try:
                return _original(self, *a, **k)
            finally:
                session_end()

        setattr(wrapped, CLAIM, True)
        wrapped.__wrapped__ = original
        setattr(cls, name, wrapped)


def install():
    from comfy.text_encoders import qwen35 as q35
    _install_attend()
    _install_session()
    if _rope_fused:
        _install_gated_attention()
    if __import__("os").environ.get("MONO_DECODE_XPU_NORM_CACHE", "1") == "1":
        _install_norm_scale()
    if __import__("os").environ.get("MONO_DECODE_XPU_LOGITS", "1") == "1":
        from . import logits
        logits.install()
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
        sp = _static_plan(self)
        if sp is None:
            _fallback("weights")
            return original(self, x, attention_mask, freqs_cis, optimized_attention, past_key_value)
        with contextlib.ExitStack() as es:
            plan = _args(self, sp, x, es)
            if plan is not None:
                return _run(self, plan, x, S, gdn, attention_mask, freqs_cis, optimized_attention, past_key_value)
        _fallback("weights")
        return original(self, x, attention_mask, freqs_cis, optimized_attention, past_key_value)

    def _run(self, plan, x, S, gdn, attention_mask, freqs_cis, optimized_attention, past_key_value):
        tiled = plan["tiled"]
        h, silu, sig = _constants(x.device)
        x2 = x.view(S, 4096)
        if gdn and _single_gdn:
            # the whole GDN block in one launch
            mix, kv, ex = self.linear_attn, past_key_value, plan["extra"]
            (nin, ein), (npo, epo) = plan["norm_in"], plan["norm_post"]
            _C.layer_gdn(x2, nin, ein, h, plan["pre"] + plan["out"] + plan["mlp"], ex["wa"], ex["wb"], ex["conv_w"],
                         kv.conv_state, kv.proj_buf, kv.qkv_buf, kv.ctl, kv.dt_bias, kv.g_decay, kv.recurrent_state,
                         kv.gates_buf, kv.sumsq_buf, kv.norm_weight, mix.norm.eps, mix.key_head_dim ** -0.5,
                         mix.num_key_heads, npo, epo, silu, 0, 5 if _gdn_split else 11, tiled)
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
            core = _gdn_core(mix, xn, outs[1].view(1, S, -1), outs[2].view(1, S, -1), past_key_value)
            y = core.reshape(S, 4096)
        else:
            mix = self.self_attn
            y = _attn_core(mix, outs[1].view(1, S, -1), outs[2].view(1, S, -1), outs[3].view(1, S, -1),
                           attention_mask, freqs_cis, optimized_attention, past_key_value).reshape(S, 4096)
            gate = outs[1]
        nw, eps = plan["norm_post"]
        _C.layer_post(x2, y.contiguous(), gate, sig, *plan["out"], nw, eps, h, silu, *plan["mlp"], 0, 6, tiled)
        with _lock:
            _stats["layer"] += 1
        return x, past_key_value

    setattr(forward, CLAIM, True)
    forward.__wrapped__ = original
    Block.forward = forward
    return True
