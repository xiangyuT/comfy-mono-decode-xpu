# SPDX-License-Identifier: Apache-2.0
"""Minimal regressions for stateful error recovery and inference-mode norm caching."""
import sys
import types
import json
from contextlib import nullcontext

import pytest
import torch

from comfy_mono_decode_xpu import guard, layer, mono, tiles


@pytest.mark.parametrize("failure", ["kv", "x", "subpath"])
def test_state_write_is_not_replayed(C, monkeypatch, failure):
    pytest.importorskip("comfy.ops")  # the layer path imports ComfyUI's ops (stubbed entry types below)
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
    for name in ("_install_attend", "_install_gated_attention"):
        monkeypatch.setattr(layer, name, lambda: False)
    monkeypatch.setattr(tiles, "install", lambda: 0)
    monkeypatch.setenv("MONO_DECODE_XPU_NORM_CACHE", "0")
    monkeypatch.setenv("MONO_DECODE_XPU_LOGITS", "0")
    monkeypatch.setenv("MONO_DECODE_XPU_AUTOTREE", "0")
    monkeypatch.setenv("MONO_DECODE_XPU_DRAFT", "0")
    monkeypatch.setattr(layer, "_static_plan", lambda block: True)
    monkeypatch.setattr(layer, "_args", lambda *args:
                        dict(tiled=False, norm_in=(None, 0), norm_post=(None, 0), pre=[], out=[], mlp=[]))
    monkeypatch.setattr(layer, "_constants", lambda device: (None, None, None, None))
    monkeypatch.setattr(C, "layer_pre", lambda *args: [x] * 4)

    def attend(*args):
        kv.add_(1)
        if failure == "kv":
            raise RuntimeError("after state write")
        if failure == "subpath":  # a fused RoPE route failed after its KV write and disabled itself
            raise layer.SubpathError("after state write")
        return x

    def post(*args):
        x.add_(1)
        raise RuntimeError("after state write")

    monkeypatch.setattr(layer, "_attn_core", attend)
    monkeypatch.setattr(C, "layer_post_h16", post)
    monkeypatch.setattr(C, "layer_post", post)
    assert layer.install()
    with pytest.raises(RuntimeError, match="after state write"):
        Block().forward(x, past_key_value=kv)
    assert not replay and kv.item() == 1 and x.max().item() == int(failure == "x")
    # A sub-route failure leaves the layer kernels enabled; any other failure disables them.
    assert guard.off("layer") == (failure != "subpath")


def _norm_rope_inputs(monkeypatch, C, kernel):
    S, H, Hk, D, rd = 2, 2, 1, 256, 64
    rawq = torch.randn(1, S, H, 2 * D).to(torch.bfloat16)[..., :D]
    rawk = torch.randn(1, S, Hk, D).to(torch.bfloat16)
    xv4 = torch.randn(1, S, Hk, D).to(torch.bfloat16)
    ref_q, ref_k = torch.randn(1, H, S, D).to(torch.bfloat16), torch.randn(1, Hk, S, D).to(torch.bfloat16)
    writes = []
    kv = types.SimpleNamespace(key=torch.zeros(1, Hk, 8, D, dtype=torch.bfloat16), value=None,
                               position=torch.arange(8), write=lambda k, v: writes.append((k, v)))
    class Norm:
        eps = 1e-6

        def __call__(self, t):
            return t

    mix = types.SimpleNamespace(q_norm=Norm(), k_norm=Norm(), rotary_dim=rd)
    q35 = types.ModuleType("comfy.text_encoders.qwen35")
    q35.apply_partial_rope = lambda q, k, f, r: (ref_q, ref_k)
    mods = {"comfy": types.ModuleType("comfy"), "comfy.text_encoders": types.ModuleType("comfy.text_encoders"),
            "comfy.text_encoders.qwen35": q35}
    for name, module in mods.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(guard, "_off", {})
    monkeypatch.setattr(guard, "_is_interrupt", lambda error: False)
    monkeypatch.setattr(layer, "_norm_weight", lambda n, device: torch.ones(D, dtype=torch.bfloat16))
    monkeypatch.setattr(C, "qk_norm_rope_kv", kernel)
    return (mix, rawq, rawk, xv4, torch.zeros(S, rd // 2, 2, 2), kv), (ref_q, ref_k), writes


def test_norm_rope_mismatch_rewrites_kv_and_keeps_layer(C, monkeypatch):
    """A different reduction order (another Torch build) must not fail the request."""
    def kernel(rawq, rawk, xv4, wq, wk, freqs, key, value, position, eq, ek, rd):
        key.fill_(7)  # the fused route wrote its own (different) K into the cache
        return torch.zeros(1, rawq.shape[2], rawq.shape[1], 256, dtype=torch.bfloat16), \
            torch.zeros(1, rawk.shape[2], rawk.shape[1], 256, dtype=torch.bfloat16)

    args, (ref_q, ref_k), writes = _norm_rope_inputs(monkeypatch, C, kernel)
    q, k = layer._norm_rope_write(*args)
    assert q is ref_q and k is ref_k
    assert len(writes) == 1 and writes[0][0] is ref_k  # KV slots rewritten from the live chain
    assert guard.off("norm_rope") and not guard.off("layer")


def test_norm_rope_native_error_is_not_replayed(C, monkeypatch):
    def kernel(*a):
        raise RuntimeError("native failure")

    args, _, writes = _norm_rope_inputs(monkeypatch, C, kernel)
    with pytest.raises(layer.SubpathError):
        layer._norm_rope_write(*args)
    assert not writes and guard.off("norm_rope") and not guard.off("layer")


def test_copy_budget_error_reads_row_major(monkeypatch):
    def broken(device):
        raise AttributeError("unexpected parameter layout")

    owner = types.SimpleNamespace(modules=lambda: iter([]))
    monkeypatch.setattr(tiles, "_copy_budget", broken)
    monkeypatch.setattr(tiles, "_tiled_ok", True)
    monkeypatch.delenv("MONO_DECODE_XPU_SESSION_TILES", raising=False)
    try:
        tiles.begin(owner)
        assert tiles.tiled(object(), types.SimpleNamespace(device=torch.device("cpu"))) is None
        assert not tiles.tiles_enabled() and tiles.stats()["budget"]["layout"] == "row"
    finally:
        tiles.end()


def test_inactive_vbar_budget_preserves_active_owners(monkeypatch):
    device = torch.device("xpu:0")
    active, shared = object(), types.SimpleNamespace(loaded_size=lambda: 128)
    control = types.SimpleNamespace(get_xpu_allocator_mode=lambda: "native_hook",
                                    get_xpu_ur_hook_stats=lambda: {"tracked_alloc_calls": 1})
    monkeypatch.setitem(sys.modules, "comfy_aimdo", types.SimpleNamespace(control=control))
    mm = types.ModuleType("comfy.model_management")
    monkeypatch.setitem(sys.modules, "comfy", types.SimpleNamespace(model_management=mm))
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    monkeypatch.setitem(tiles._session, "model", active)

    def loaded(used=False, owner=None, dev=device, dynamic=True, bar=shared):
        model = types.SimpleNamespace(modules=lambda: iter([owner]), dynamic_vbars={dev: bar})
        return types.SimpleNamespace(device=dev, currently_used=used,
                                     model=types.SimpleNamespace(model=model, is_dynamic=lambda: dynamic))

    # Current owner, in-use model, other device and non-dynamic storage cannot contribute.
    # Shared inactive VBARs contribute their physical backing once.
    protected = types.SimpleNamespace(loaded_size=lambda: 512)
    mm.current_loaded_models = [loaded(owner=active, bar=protected), loaded(bar=protected),
                                loaded(used=True, bar=object()), loaded(dev=torch.device("xpu:1")),
                                loaded(dynamic=False), loaded(), loaded()]
    assert tiles._inactive_vram_budget(device) == 128
    control.get_xpu_allocator_mode = lambda: "global"
    assert tiles._inactive_vram_budget(device) == 0
    control.get_xpu_allocator_mode = lambda: "native_hook"
    control.get_xpu_ur_hook_stats = lambda: {}
    assert tiles._inactive_vram_budget(device) == 0


def test_reclaimable_pages_do_not_override_model_capacity(monkeypatch):
    device = torch.device("xpu:0")
    owner = torch.nn.Linear(16, 16, bias=False)
    owner._v = object()
    owner.weight._qdata = torch.zeros(16, 16, dtype=torch.int8)
    owner.weight._params = types.SimpleNamespace(group_size=16, convrot_groupsize=256,
        scale=torch.ones(32, dtype=torch.uint8), s_channel=torch.ones(16), codebook=torch.ones(16))
    mm = types.ModuleType("comfy.model_management")
    mm.minimum_inference_memory = lambda: 100
    monkeypatch.setitem(sys.modules, "comfy", types.SimpleNamespace(model_management=mm))
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    monkeypatch.setitem(tiles._session, "model", owner)
    monkeypatch.setattr(tiles, "_inactive_vram_budget", lambda device: 10000)
    monkeypatch.setattr(tiles, "_margin", 64)
    monkeypatch.setattr(torch.xpu, "memory_reserved", lambda device: 100)
    monkeypatch.setattr(torch.xpu, "memory_allocated", lambda device: 100)
    monkeypatch.setattr(torch.xpu, "mem_get_info", lambda device: (0, 1024))
    props = types.SimpleNamespace(total_memory=1024)
    monkeypatch.setattr(torch.xpu, "get_device_properties", lambda device: props)
    assert tiles._copy_budget(device)["layout"] == "tile"
    props.total_memory = 800
    assert tiles._copy_budget(device)["layout"] == "row"


@pytest.mark.parametrize("allocated,free,expected", [
    (1575362048, 13634584576, "tile"),
    (1575362048, 11500000000, "row"),
    (20000000000, 13634584576, "row"),
])
def test_copy_budget_reuses_allocator_cache(monkeypatch, allocated, free, expected):
    total, reserved = 33456746496, max(18131976192, allocated)
    reserve, model_bytes, copy_bytes = 5153960755.2, 7343366112, 3888119808
    comfy = types.ModuleType("comfy")
    management = types.ModuleType("comfy.model_management")
    management.minimum_inference_memory = lambda: reserve
    comfy.model_management = management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    monkeypatch.setattr(torch.xpu, "get_device_properties", lambda device:
                        types.SimpleNamespace(total_memory=total))
    monkeypatch.setattr(torch.xpu, "memory_reserved", lambda device: reserved)
    monkeypatch.setattr(torch.xpu, "memory_allocated", lambda device: allocated)
    monkeypatch.setattr(torch.xpu, "mem_get_info", lambda device: (free, total))
    monkeypatch.setattr(tiles, "_margin", 3 * 1024 ** 3)
    monkeypatch.setattr(tiles, "_inactive_vram_budget", lambda device: 0)
    packed, scale = object(), object()
    weight = types.SimpleNamespace(device=torch.device("cpu"), _qdata=packed,
                                   _params=types.SimpleNamespace(group_size=16, convrot_groupsize=256, scale=scale))
    module = types.SimpleNamespace(weight=weight, _v=object())
    owner = types.SimpleNamespace(modules=lambda: iter([module]), parameters=lambda: iter([weight]))
    monkeypatch.setattr(tiles, "_storage_bytes", lambda tensors:
                        copy_bytes if tensors == [packed, scale] else model_bytes)
    try:
        tiles.begin(owner)
        budget = tiles._copy_budget(torch.device("xpu:0"))
        assert budget["layout"] == expected
        assert budget["dynamic_vram"]
        assert budget["allocated_bytes"] == allocated and budget["reserved_bytes"] == reserved
        assert budget["available_bytes"] == max(0, min(total - allocated - reserve - model_bytes,
                                                     free - reserve - tiles._margin))
        if expected == "tile":
            assert total - reserved - reserve - model_bytes < copy_bytes
            assert budget["available_bytes"] >= copy_bytes
    finally:
        tiles.end()


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


@pytest.mark.parametrize("device,diagnostic,trim,expected", [
    ("xpu", "0", "0", []),
    ("xpu", "1", "0", ["memory"]),
    ("xpu", "0", "1", ["memory", "synchronize", "empty_cache", "memory"]),
    ("cpu", "1", "1", []),
])
def test_decode_boundary_controls(monkeypatch, device, diagnostic, trim, expected):
    calls = []
    monkeypatch.setenv("MONO_DECODE_XPU_DIAGNOSTICS", diagnostic)
    monkeypatch.setenv("MONO_DECODE_XPU_PREFILL_TRIM", trim)
    monkeypatch.setattr(tiles, "_decode_memory", lambda device: calls.append("memory") or {})
    monkeypatch.setattr(torch.xpu, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.xpu, "synchronize", lambda device: calls.append("synchronize"))
    monkeypatch.setattr(torch.xpu, "empty_cache", lambda: calls.append("empty_cache"))
    tiles._prepare_decode(torch.device(device))
    assert calls == expected


def test_decode_boundary_runs_once_per_generate(monkeypatch):
    calls = []
    owner = types.SimpleNamespace(modules=lambda: iter([]))
    monkeypatch.setattr(tiles, "_prepare_decode", lambda device: calls.append(device))
    monkeypatch.setattr(tiles, "_copy_budget", lambda device:
                        dict(layout="row", copy_bytes=1, available_bytes=0))
    monkeypatch.setattr(tiles, "_tiled_ok", True)
    monkeypatch.setenv("MONO_DECODE_XPU_SESSION_TILES", "1")
    device = torch.device("cpu")
    try:
        for generation in range(2):
            tiles.begin(owner)
            for step in range(3):
                assert tiles.tiled(object(), types.SimpleNamespace(device=device)) is None
            tiles.end()
        assert calls == [device, device]
    finally:
        tiles.end()


@pytest.mark.parametrize("instance_forward", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_generation_diagnostics_restore_forward(monkeypatch, caplog, instance_forward, fail):
    class Body:
        def forward(self, embeds=None, past_key_values=None):
            if fail:
                raise RuntimeError("generation failed")
            return embeds

    body = Body()
    if instance_forward:
        body.forward = body.forward
    original = body.forward
    model = types.SimpleNamespace(model=body)
    monkeypatch.setenv("MONO_DECODE_XPU_DIAGNOSTICS", "1")
    monkeypatch.delenv("MONO_DECODE_XPU_PROFILE", raising=False)
    embeds = torch.zeros(1, 1, 4)
    caplog.set_level("INFO")
    try:
        with tiles._diagnose_generation(model):
            for step in range(33):
                assert body.forward(embeds=embeds) is embeds
    except RuntimeError as error:
        assert fail and str(error) == "generation failed"
    assert body.forward == original
    assert ("forward" in vars(body)) == instance_forward
    if not fail:
        assert "forward=1 " in caplog.text and "forward=33 " in caplog.text


def test_generation_diagnostics_disabled(monkeypatch):
    monkeypatch.setenv("MONO_DECODE_XPU_DIAGNOSTICS", "0")
    monkeypatch.delenv("MONO_DECODE_XPU_PROFILE", raising=False)
    with tiles._diagnose_generation(object()):
        pass


@pytest.mark.parametrize("case,expected", [
    ("ready", False), ("prefill", True), ("missing_plan", True),
    ("row", True), ("inactive", True), ("disabled", True),
    ("guard", True), ("cpu", True), ("no_kv", True), ("no_prefetch", False),
])
@pytest.mark.parametrize("fail", [False, True])
def test_tile_plan_prefetch_reuse(monkeypatch, case, expected, fail):
    blocks = [object(), object()]
    seen = []

    class Body:
        layers = blocks
        prefetch_dynamic_vbars = case != "no_prefetch"

        def forward(self, embeds=None, past_key_values=None):
            seen.append(self.prefetch_dynamic_vbars)
            if fail:
                raise RuntimeError("decode failed")
            return embeds

    body = Body()
    original = body.forward
    original_prefetch = body.prefetch_dynamic_vbars
    owner = types.SimpleNamespace(model=body, modules=lambda: iter([]))
    monkeypatch.setattr(guard, "_off", {"layer": "test"} if case == "guard" else {})
    monkeypatch.setattr(tiles, "_tiled_ok", True)
    monkeypatch.setenv("MONO_DECODE_XPU_SESSION_TILES", "1")
    monkeypatch.setenv("MONO_DECODE_XPU_REUSE_TILES", "0" if case == "disabled" else "1")
    embeds = types.SimpleNamespace(device=torch.device("cpu" if case == "cpu" else "xpu"),
                                   dtype=torch.bfloat16, ndim=3,
                                   shape=(1, 100 if case == "prefill" else 8, 4096))
    try:
        tiles.begin(owner)
        for block in blocks:
            tiles.set_plan(block, {"tiled": True})
        if case == "missing_plan":
            tiles._session["plans"].pop(blocks[-1])
        elif case == "row":
            tiles._session["ok"] = False
        elif case == "inactive":
            tiles.end()
        try:
            with tiles._reuse_tile_plans(owner):
                result = body.forward(embeds=embeds, past_key_values=None if case == "no_kv" else [])
                assert result is embeds
                assert body.prefetch_dynamic_vbars == expected
        except RuntimeError as error:
            assert fail and str(error) == "decode failed"
        assert seen == [expected]
        assert body.forward == original and "forward" not in vars(body)
        assert body.prefetch_dynamic_vbars == original_prefetch
    finally:
        tiles.end()


@pytest.mark.parametrize("case,expected", [
    ("ready", "rows"), ("resident", "full"), ("prefetched", "full"),
    ("prefill", "full"), ("cpu", "full"), ("weight_function", "full"),
    ("lowvram", "full"), ("max_norm", "full"), ("format", "full"), ("disabled", "full"),
])
@pytest.mark.parametrize("fail", [False, True])
def test_host_embedding_rows(monkeypatch, case, expected, fail):
    class Embedding:
        _v = object()
        quant_format = "fp8" if case == "format" else "int8_tensorwise"
        weight = types.SimpleNamespace(device=torch.device("cpu"))
        weight_function = [object()] if case == "weight_function" else []
        bias_function = []
        weight_lowvram_function = object() if case == "lowvram" else None
        max_norm = 1 if case == "max_norm" else None

        def forward(self, tokens, out_dtype=None):
            return "full"

        def host_rows(self, tokens, out_dtype=None):
            assert out_dtype == torch.bfloat16
            if fail:
                raise RuntimeError("row gather failed")
            return "rows"

    embedding = Embedding()
    if case == "prefetched":
        embedding._prefetch = {"resident": True}
    residency = {"above_watermark": case != "resident"}
    ops = types.ModuleType("comfy.ops")
    ops.vbar_above_watermark = lambda module: (
        residency["above_watermark"] and getattr(module, "_prefetch", None) is None)
    comfy = types.ModuleType("comfy")
    comfy.ops = ops
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.ops", ops)
    original = embedding.forward
    owner = types.SimpleNamespace(model=types.SimpleNamespace(embed_tokens=embedding), modules=lambda: iter([]))
    tokens = types.SimpleNamespace(device=torch.device("cpu" if case == "cpu" else "xpu"),
                                   numel=lambda: 100 if case == "prefill" else 8)
    monkeypatch.setenv("MONO_DECODE_XPU_HOST_ROWS", "0" if case == "disabled" else "1")
    try:
        tiles.begin(owner)
        try:
            with tiles._host_embedding_rows(owner):
                assert embedding.forward(tokens, out_dtype=torch.bfloat16) == expected
                if case == "ready" and not fail:
                    residency["above_watermark"] = False
                    assert embedding.forward(tokens, out_dtype=torch.bfloat16) == "full"
                    residency["above_watermark"] = True
                    assert embedding.forward(tokens, out_dtype=torch.bfloat16) == "rows"
                    embedding._prefetch = {"resident": True}
                    assert embedding.forward(tokens, out_dtype=torch.bfloat16) == "full"
        except RuntimeError as error:
            assert fail and expected == "rows" and str(error) == "row gather failed"
        assert embedding.forward == original and "forward" not in vars(embedding)
    finally:
        tiles.end()


@pytest.mark.parametrize("streamed", [False, True])
def test_fused_draft_embedding_avoids_full_table_cast(monkeypatch, streamed):
    from comfy_mono_decode_xpu.draft_head import DraftHead

    calls = []
    params = types.SimpleNamespace(scale=torch.ones(4, 1), convrot=False)
    quantized = types.SimpleNamespace(_qdata=torch.ones(4, 4, dtype=torch.int8), _params=params)
    embedding = types.SimpleNamespace(
        _v=object(), quant_format="int8_tensorwise", weight_function=[], bias_function=[],
        weight_lowvram_function=None, max_norm=None,
        weight=types.SimpleNamespace(device=torch.device("cpu"), dtype=torch.bfloat16,
                        _layout_cls="TensorWiseINT8Layout", _params=params),
        host_rows=lambda tokens, out_dtype: calls.append("rows") or torch.ones(1, 1, 4, dtype=out_dtype))
    token = types.SimpleNamespace(device=torch.device("xpu"), numel=lambda: 1)
    token.reshape = lambda *shape: token
    ops = types.ModuleType("comfy.ops")
    ops.vbar_above_watermark = lambda module: streamed
    ops.CastBiasWeightContext = lambda *args, **kwargs: (
        calls.append("cast") or nullcontext((quantized, None)))
    comfy = types.ModuleType("comfy")
    comfy.ops = ops
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.ops", ops)
    monkeypatch.setenv("MONO_DECODE_XPU_HOST_ROWS", "1")
    empty = torch.empty(0)

    def fused(*args):
        if streamed:
            assert args[0].numel() == 0 and torch.equal(args[4], torch.ones(4, dtype=torch.bfloat16))
            calls.append("fused norms")
        else:
            assert args[0] is token and args[1] is quantized._qdata and args[4].numel() == 0
            calls.append("fused gather")

    head = types.SimpleNamespace(embed=embedding, C=types.SimpleNamespace(draft_embed_norm_cat=fused),
                                 eps=(1e-6, 1e-6), empty=empty, n_e=empty, n_h=empty, fc_in=empty)
    tiles.begin(types.SimpleNamespace(modules=lambda: iter([])))
    try:
        DraftHead._embed(head, token, torch.zeros(1, 1, 4))
        assert calls == (["rows", "fused norms"] if streamed else ["cast", "fused gather"])
    finally:
        tiles.end()


@pytest.mark.parametrize("case", ["ready", "prefill", "cpu", "prefetch", "missing_plan",
                                 "weight_function", "lowvram", "budget", "oom", "error", "disabled"])
def test_owned_output_head(monkeypatch, case):
    from comfy_kitchen.tensor.base import QuantizedTensor
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

    qdata = torch.arange(64, dtype=torch.int8).reshape(4, 16)
    scale = torch.ones(4, 1)
    weight = QuantizedTensor(qdata, "TensorWiseINT8Layout", TensorWiseINT8Layout.Params(
        scale=scale, orig_dtype=torch.bfloat16, orig_shape=tuple(qdata.shape)))
    head = types.SimpleNamespace(weight=weight, bias=None, _v=object(), quant_format="int8_tensorwise",
                                 weight_function=[object()] if case == "weight_function" else [],
                                 bias_function=[], weight_lowvram_function=object() if case == "lowvram" else None)
    body = types.SimpleNamespace(lm_head=head, prefetch_dynamic_vbars=case == "prefetch", layers=[object()])
    owner = types.SimpleNamespace(model=body, modules=lambda: iter([]))
    input = types.SimpleNamespace(device=torch.device("cpu" if case == "cpu" else "xpu"),
                                  dtype=torch.bfloat16, ndim=3, shape=(1, 100 if case == "prefill" else 1, 16))
    calls, releases = [], []
    ops = types.ModuleType("comfy.ops")
    comfy = types.ModuleType("comfy")
    comfy.ops = ops

    def cast(module, input=None, *args, **kwargs):
        calls.append(module)
        return weight, None, (None, object(), None)

    ops.cast_bias_weight = cast
    ops.uncast_bias_weight = lambda *args: releases.append(args)
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.ops", ops)
    monkeypatch.setattr(tiles, "plan", lambda block: None if case == "missing_plan" else object())
    monkeypatch.setattr(tiles, "_copy_budget", lambda device: {"available_bytes": 0 if case == "budget" else 1024})
    monkeypatch.setenv("MONO_DECODE_XPU_HEAD_CACHE", "0" if case == "disabled" else "1")
    if case in ("oom", "error"):
        def fail_clone(self):
            raise torch.OutOfMemoryError("test allocation") if case == "oom" else RuntimeError("test clone")
        monkeypatch.setattr(QuantizedTensor, "clone", fail_clone)
    tiles.begin(owner)
    try:
        with tiles._owned_output_head(owner):
            if case == "error":
                with pytest.raises(RuntimeError, match="test clone"):
                    ops.cast_bias_weight(head, input, offloadable=True)
                assert len(releases) == 1
            else:
                first = ops.cast_bias_weight(head, input, offloadable=True)
                second = ops.cast_bias_weight(head, input, offloadable=True)
                if case == "ready":
                    assert len(calls) == len(releases) == 1
                    assert first[0] is second[0] and first[2] is None
                    assert type(first[0]) is QuantizedTensor
                    assert first[0]._qdata.data_ptr() != qdata.data_ptr()
                    assert first[0]._params.scale.data_ptr() != scale.data_ptr()
                    assert torch.equal(first[0]._qdata, qdata) and torch.equal(first[0]._params.scale, scale)
                    assert tiles._session["head_cache"] == dict(bytes=80, hits=1, reason="ready")
                    qdata.zero_()
                    scale.zero_()
                    assert first[0]._qdata.count_nonzero() == 63 and first[0]._params.scale.count_nonzero() == 4
                    body.prefetch_dynamic_vbars = True
                    assert ops.cast_bias_weight(head, input, offloadable=True)[0] is weight
                    assert ops.cast_bias_weight(module=head, input=input, offloadable=True)[0] is weight
                else:
                    assert len(calls) == 2 and not releases
                    assert first[0] is second[0] is weight
            assert head.weight is weight
        assert ops.cast_bias_weight is cast
    finally:
        tiles.end()


def test_tile_prefetch_restores_when_plan_becomes_unavailable(monkeypatch):
    block = object()

    class Body:
        layers = [block]
        prefetch_dynamic_vbars = True

        def forward(self, embeds=None, past_key_values=None):
            return self.prefetch_dynamic_vbars

    body = Body()
    owner = types.SimpleNamespace(model=body, modules=lambda: iter([]))
    embeds = types.SimpleNamespace(device=torch.device("xpu"), dtype=torch.bfloat16,
                                   ndim=3, shape=(1, 8, 4096))
    monkeypatch.setattr(guard, "_off", {})
    monkeypatch.setattr(tiles, "_tiled_ok", True)
    monkeypatch.setenv("MONO_DECODE_XPU_REUSE_TILES", "1")
    monkeypatch.setenv("MONO_DECODE_XPU_SESSION_TILES", "1")
    try:
        tiles.begin(owner)
        with tiles._reuse_tile_plans(owner):
            assert body.forward(embeds=embeds, past_key_values=[]) is True
            tiles.set_plan(block, {"tiled": True})
            assert body.forward(embeds=embeds, past_key_values=[]) is False
            assert body.prefetch_dynamic_vbars is False
            tiles._session["plans"].clear()
            assert body.forward(embeds=embeds, past_key_values=[]) is True
            tiles.set_plan(block, {"tiled": True})
            assert body.forward(embeds=embeds, past_key_values=[]) is False
        assert body.prefetch_dynamic_vbars is True
    finally:
        tiles.end()


@pytest.mark.parametrize("steps,fail_at,expected", [
    (8, None, []),
    (12, None, ["start", "metadata", "stop", "export", "table"]),
    (20, None, ["start", "metadata", "stop", "export", "table"]),
    (20, 10, ["start", "metadata", "stop", "export", "table"]),
])
def test_generation_profile_window(monkeypatch, steps, fail_at, expected):
    events = []

    class Profile:
        def start(self):
            events.append("start")

        def add_metadata_json(self, name, value):
            assert name == "mono_decode_xpu"
            metadata = json.loads(value)
            assert metadata["revision"] == "vocabulary-transfer-v3"
            assert "weight_budget" in metadata and "prefetch_skips" in metadata
            events.append("metadata")

        def stop(self):
            events.append("stop")

        def export_chrome_trace(self, path):
            assert path == "decode-test.json"
            events.append("export")

        def key_averages(self):
            return self

        def table(self, sort_by, row_limit):
            assert sort_by == "self_device_time_total"
            events.append("table")
            return "profile table"

    class Body:
        calls = 0

        def forward(self, embeds=None):
            self.calls += 1
            if self.calls == fail_at:
                raise RuntimeError("generation failed")
            return embeds

    body = Body()
    original = body.forward
    model = types.SimpleNamespace(model=body)
    embeds = types.SimpleNamespace(device=torch.device("xpu"), shape=(1, 1, 4))
    monkeypatch.setenv("MONO_DECODE_XPU_DIAGNOSTICS", "0")
    monkeypatch.setenv("MONO_DECODE_XPU_PROFILE", "decode-test.json")
    monkeypatch.setattr(torch.profiler, "profile", lambda **kwargs: Profile())
    try:
        with tiles._diagnose_generation(model):
            for step in range(steps):
                assert body.forward(embeds=embeds) is embeds
    except RuntimeError as error:
        assert fail_at is not None and str(error) == "generation failed"
    assert events == expected
    assert body.forward == original and "forward" not in vars(body)
