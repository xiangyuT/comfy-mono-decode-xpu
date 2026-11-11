# SPDX-License-Identifier: Apache-2.0
"""Minimal regressions for stateful error recovery and inference-mode norm caching."""
import sys
import types

import pytest
import torch

from comfy_mono_decode_xpu import guard, layer, mono, tiles


@pytest.mark.parametrize("failure", ["kv", "x"])
def test_state_write_is_not_replayed(C, monkeypatch, failure):
    x = torch.zeros(1, 1, 4096, dtype=torch.bfloat16, device="xpu")
    kv = torch.zeros(1, device="xpu")
    replay = []

    class Block:
        layer_type = "full_attention"
        self_attn = None
        forward = lambda *args, **kwargs: replay.append(1)

    # Only the ComfyUI entry types are stand-ins; writes use real XPU tensors.
    mods = {name: types.ModuleType(name) for name in
            ("comfy", "comfy.text_encoders", "comfy.text_encoders.qwen35",
             "comfy.text_encoders.llama")}
    mods["comfy.text_encoders.qwen35"].Qwen35TransformerBlock = Block
    mods["comfy.text_encoders.llama"].SPEC_ROWS = 8
    for name, module in mods.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(guard, "_off", {})
    monkeypatch.setattr(guard, "_is_interrupt", lambda error: False)
    monkeypatch.setattr(mono, "comfyui_has_combined_linear", lambda: False)
    for name in ("_install_attend", "_install_gated_attention"):
        monkeypatch.setattr(layer, name, lambda: False)
    monkeypatch.setattr(tiles, "install", lambda: 0)
    monkeypatch.setenv("MONO_DECODE_XPU_NORM_CACHE", "0")
    monkeypatch.setenv("MONO_DECODE_XPU_LOGITS", "0")
    monkeypatch.setattr(layer, "_static_plan", lambda block: True)
    monkeypatch.setattr(layer, "_args", lambda *args:
                        dict(tiled=False, norm_in=(None, 0), norm_post=(None, 0), pre=[], out=[], mlp=[]))
    monkeypatch.setattr(layer, "_constants", lambda device: (None, None, None))
    monkeypatch.setattr(C, "layer_pre", lambda *args: [x] * 4)

    def attend(*args):
        kv.add_(1)
        if failure == "kv":
            raise RuntimeError("after state write")
        return x

    def post(*args):
        x.add_(1)
        raise RuntimeError("after state write")

    monkeypatch.setattr(layer, "_attn_core", attend)
    monkeypatch.setattr(C, "layer_post", post)
    assert layer.install()
    with pytest.raises(RuntimeError, match="after state write"):
        Block().forward(x, past_key_value=kv)
    assert not replay and kv.item() == 1 and x.max().item() == int(failure == "x")


def test_inference_norm_cache_has_generation_lifetime(monkeypatch):
    with torch.inference_mode():
        w = torch.ones(16, dtype=torch.bfloat16)
    calls = []
    norm = types.SimpleNamespace(weight=w, scale=lambda: calls.append(1) or w + 1)
    owner = types.SimpleNamespace(modules=lambda: iter([norm]))
    monkeypatch.setattr(layer, "comfy_cast", lambda value, device: value)
    try:
        tiles.begin(owner)
        a = layer._norm_weight(norm, torch.device("cpu"))
        assert layer._norm_weight(norm, torch.device("cpu")) is a and calls == [1]
        tiles.end()
        tiles.begin(owner)
        assert torch.equal(layer._norm_weight(norm, torch.device("cpu")), a)
        assert calls == [1, 1]
    finally:
        tiles.end()
