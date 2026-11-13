# SPDX-License-Identifier: Apache-2.0
"""XPU graph for the Qwen3.5 MTP draft head (MTPHead.forward) inside an MTP generate.

A draft's device work is short (about 0.75 ms) and its eager launch path (two norms, cat, fc, the
attention block's projections, fused Q/K norm + RoPE + KV write, DPAS attention, gate, MLP, norm)
costs about as much host time. The draft that follows a step's host sync starts on an idle device,
so that host time is device idle. Upstream replays the draft as a CUDA graph; on XPU it runs eager.

Everything the head reads is device-side at fixed addresses during a generate: upstream's static
carriers (draft embedding, hidden, rope slice), the MTP KV cache with its device position / seqlen
(written by mtp_kv.prepare before the head runs), generation-owned BF16 linear weights (draft.py)
and resident norm weights. So after a few eager drafts (which also run the owned linears' and the
fused rope's first-use checks) the head is captured once per generate and replayed: the same
kernels on the same inputs, so the same bits. Anything else (DynamicVRAM-managed or unowned weights,
a different carrier, hooks) runs the original forward.
"""
from __future__ import annotations

import os
import threading

import torch

from . import guard, tiles

CLAIM = "__comfy_mono_decode_xpu_graph_draft__"
WARMUP = int(os.environ.get("MONO_DECODE_XPU_GRAPH_DRAFT_WARMUP", "4"))
CHECK = os.environ.get("MONO_DECODE_XPU_GRAPH_DRAFT_CHECK") == "1"  # validation: eager rerun per replay
_lock = threading.Lock()
_stats = {"replay": 0, "capture": 0, "eager": 0, "check_mismatch": 0, "fallback": {}}


def stats():
    with _lock:
        return dict(_stats, fallback=dict(_stats["fallback"]))


def _fallback(why):
    with _lock:
        _stats["fallback"][why] = _stats["fallback"].get(why, 0) + 1


def _weights_fixed(mtp):
    """Every weight the head reads stays at one address for the generate: BF16 linears owned by
    draft.py (copied and checked), everything else resident on the device and outside DynamicVRAM."""
    from . import draft
    owned = set()
    for m in draft._linears(mtp):
        o = tiles.cached(("draft", id(m)), lambda: {})
        if o.get("w") is None or not o.get("checked"):
            return False
        owned.add(id(m))
    for m in mtp.modules():
        if id(m) in owned:
            continue
        if hasattr(m, "_v") or getattr(m, "weight_lowvram_function", None) is not None or getattr(m, "weight_function", None):
            return False
        for name in ("weight", "bias"):
            w = getattr(m, name, None)
            if isinstance(w, torch.Tensor) and w.device.type != "xpu":
                return False
    return True


def _same(a, b):
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def install():
    from comfy.text_encoders import qwen35 as q35
    from comfy.text_encoders import llama
    from torch.nn.modules import module as nn_module
    T = getattr(q35, "MTPHead", None)
    original = getattr(T, "forward", None)
    if original is None or getattr(original, CLAIM, False) or not hasattr(torch, "xpu") or not hasattr(torch.xpu, "XPUGraph"):
        return False

    def forward(self, embeds, hidden, freqs_cis, past_key_value):
        call = lambda: original(self, embeds, hidden, freqs_cis, past_key_value)
        kv = past_key_value
        from . import mtp_loop
        if (guard.off("graph_draft") or not tiles.session_active() or torch.is_grad_enabled() or mtp_loop.in_async_loop()
                or not all(isinstance(t, torch.Tensor) and t.device.type == "xpu" for t in (embeds, hidden, freqs_cis))
                or embeds.dim() != 3 or embeds.shape[:2] != (1, 1) or hidden.shape != embeds.shape
                or embeds.dtype != torch.bfloat16 or type(kv) is not llama.FixedKVCache
                or (kv.tracker or {}).get("tree") is not None
                or any((nn_module._global_forward_hooks, nn_module._global_forward_pre_hooks))):
            return call()
        key = (id(self), embeds.data_ptr(), hidden.data_ptr(), freqs_cis.data_ptr(), tuple(freqs_cis.shape),
               kv.key.data_ptr(), kv.value.data_ptr(), kv.position.data_ptr(), kv.seqlen.data_ptr())
        state = tiles.cached(("graph_draft", id(self)), lambda: {"key": None})
        if state["key"] != key:
            state.clear()
            state.update(key=key, steps=0, graph=None, out=None)
        try:
            if state["graph"] is not None:
                state["graph"].replay()
                n1, x = state["out"]
                with _lock:
                    _stats["replay"] += 1
                if CHECK:  # rewrites the same KV slot with the same values
                    rn, rx = call()
                    if not (_same(n1, rn) and _same(x, rx)):
                        with _lock:
                            _stats["check_mismatch"] += 1
                return n1, x
            if state["steps"] < WARMUP or not _weights_fixed(self):
                state["steps"] += 1
                with _lock:
                    _stats["eager"] += 1
                return call()
            torch.xpu.synchronize(embeds.device)
            # capture records without running; replay once so this draft's work (and its KV write)
            # happens exactly once
            g = torch.xpu.XPUGraph()
            with torch.xpu.graph(g):
                out = call()
            g.replay()
            state["graph"], state["out"] = g, out
            with _lock:
                _stats["capture"] += 1
            return out
        except Exception as e:  # before replay nothing ran; the eager head recomputes the draft
            guard.failed("graph_draft", e)
            _fallback(type(e).__name__)
            state.update(graph=None, out=None, steps=-(1 << 30))
            return call()

    setattr(forward, CLAIM, True)
    forward.__wrapped__ = original
    T.forward = forward
    return True
