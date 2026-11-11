"""Whole-verify XPU graph for Qwen3.5 MTP decode (experimental, resident weights only).

The verify forward of ``Llama2_.forward`` interleaves per-layer host bookkeeping
(``past_kv.prepare``/``advance``) with device-only layer cores. The bookkeeping is shared
by all layers and writes its device state (positions, seqlen, deferred-delta ctl) before
the first core reads it, so one verify step is:

    eager:  hidden/rotary static buffers, every past_kv.prepare(rows)
    graph:  all decoder layer cores + final norm into a static output
    eager:  every past_kv.advance(rows)

The graph is captured once per generate (after a few eager steps) and replayed
afterwards. Weights must be resident at fixed addresses: any prefetch queue, DynamicVRAM
vbar, or a changed layer/weight set makes the call run the original forward.
"""
from __future__ import annotations

import os
import threading

import torch

CLAIM = "__comfy_mono_decode_xpu_graph__"
WARMUP = int(os.environ.get("MONO_DECODE_XPU_GRAPH_WARMUP", "3"))
_stats = {"replay": 0, "capture": 0, "eager": 0, "fallback": {}}
_lock = threading.Lock()


def _fallback(why):
    with _lock:
        _stats["fallback"][why] = _stats["fallback"].get(why, 0) + 1


def stats():
    with _lock:
        return {"replay": _stats["replay"], "capture": _stats["capture"], "eager": _stats["eager"],
                "fallback": dict(_stats["fallback"])}


def _resident(model):
    for m in model.layers.modules():
        if hasattr(m, "_v") or getattr(m, "weight_lowvram_function", None) is not None:
            return False
        if getattr(m, "weight_function", None):
            return False
        w = getattr(m, "weight", None)
        if isinstance(w, torch.Tensor) and w.device.type != "xpu":
            return False
    return True


def install():
    import comfy.text_encoders.qwen35 as q35
    from comfy.text_encoders.llama import FixedKV, rope_matrix

    T = q35.Qwen35Transformer
    original = T.forward
    if getattr(original, CLAIM, False):
        return False

    def forward(self, x, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None,
                final_layer_norm_intermediate=True, dtype=None, position_ids=None, embeds_info=[],
                past_key_values=None, input_ids=None, deepstack_embeds=None, visual_pos_masks=None,
                decode_buffers=None, freqs_cis=None):
        call = lambda: original(self, x, attention_mask=attention_mask, embeds=embeds, num_tokens=num_tokens,
                                intermediate_output=intermediate_output,
                                final_layer_norm_intermediate=final_layer_norm_intermediate, dtype=dtype,
                                position_ids=position_ids, embeds_info=embeds_info, past_key_values=past_key_values,
                                input_ids=input_ids, deepstack_embeds=deepstack_embeds,
                                visual_pos_masks=visual_pos_masks, decode_buffers=decode_buffers, freqs_cis=freqs_cis)
        if (decode_buffers is None or embeds is None or past_key_values is None or attention_mask is not None
                or intermediate_output is not None or deepstack_embeds is not None or freqs_cis is not None
                or embeds.device.type != "xpu" or not self.graph_dynamic_vbar_blocks):
            return call()
        rows = embeds.shape[1]
        if not (2 <= rows <= 8) or not all(isinstance(kv, FixedKV) for kv in past_key_values):
            return call()
        state = getattr(self, "_mono_decode_xpu_graph", None)
        key = (id(past_key_values[0]), rows, embeds.shape[0], embeds.dtype)
        if state is None or state["key"] != key:
            if not _resident(self):
                _fallback("not_resident")
                return call()
            state = self._mono_decode_xpu_graph = {"key": key, "steps": 0, "graph": None, "out": None}
        hidden_buffer, rotary_buffer = decode_buffers
        hidden_buffer.copy_(embeds)
        rotary_buffer.copy_(rope_matrix(self.compute_freqs_cis(position_ids, embeds.device)))
        for kv in past_key_values:
            if type(kv) is not FixedKV:
                kv.prepare(rows)
        # type(kv) is FixedKV prepares once inside layer 0's core in the original; Qwen3.5 has none.
        optimized_attention = q35.optimized_attention_for_device(embeds.device, mask=False, small_input=True)

        def cores():
            for i, layer in enumerate(self.layers):
                out, _ = layer(x=hidden_buffer, attention_mask=None, freqs_cis=rotary_buffer,
                               optimized_attention=optimized_attention, past_key_value=past_key_values[i])
                hidden_buffer.copy_(out)
            return self.norm(hidden_buffer) if self.norm is not None else hidden_buffer

        if state["graph"] is not None:
            state["graph"].replay()
            out = state["out"]
            with _lock:
                _stats["replay"] += 1
        elif state["steps"] < WARMUP:
            out = cores()
            state["steps"] += 1
            with _lock:
                _stats["eager"] += 1
        else:
            dev = embeds.device
            torch.xpu.synchronize(dev)
            # capture over a copy of the live state is unsafe (kernels write KV/GDN state); run the
            # step itself inside the capture, then replay it once so this step's work happens exactly once.
            g = torch.xpu.XPUGraph()
            with torch.xpu.graph(g):
                captured = cores()
            g.replay()
            state["graph"], state["out"] = g, captured
            out = captured
            with _lock:
                _stats["capture"] += 1
        for kv in past_key_values:
            kv.advance(rows)
        return out, None, list(past_key_values)

    setattr(forward, CLAIM, True)
    forward.__wrapped__ = original
    T.forward = forward
    return True
