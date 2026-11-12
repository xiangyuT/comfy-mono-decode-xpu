# SPDX-License-Identifier: Apache-2.0
"""Actual ComfyUI combined storage and MLP entry, without loading a checkpoint."""
import contextlib
from types import SimpleNamespace

import pytest
import torch

from comfy_mono_decode_xpu import mono
from conftest import bits_equal
from test_w4a8_linear import random_w4


def _weight(n, k, generator):
    from comfy_kitchen.tensor import QuantizedTensor
    from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout as Layout
    q, scale, channel, codebook = random_w4(n, k, generator, dev="cpu")
    return QuantizedTensor(q, Layout.__name__, Layout.Params(
        scale=scale, s_channel=channel, codebook=codebook,
        orig_dtype=torch.bfloat16, orig_shape=(n, k)))


def _combine(group):
    """Use ComfyUI's actual in-place gathering; prefetch pins are exercised by the UI lifecycle."""
    from comfy import memory_management as memory
    weights = [m.weight for m in group.linears]
    storage = torch.empty(memory.vram_aligned_size(weights), dtype=torch.uint8, device="xpu")
    views = memory.interpret_gathered_like(weights, storage)
    for src, dst in zip(weights, views):
        dst.__tensor_flatten__()
        for attr in src.__tensor_flatten__()[0]:
            getattr(dst, attr).copy_(getattr(src, attr))
    group._v_weight = memory.cat_gathered(views, storage)
    assert group._v_weight is not None
    group._v = (None, 0, storage.numel())
    for m in group.linears:
        m._prefetch = {"resident": True}
        m._v_weight = None


def test_combined_members_follow_current_residency(C):
    ops = pytest.importorskip("comfy.ops")
    if not hasattr(ops, "CombinedLinear"):
        pytest.skip("ComfyUI CombinedLinear interface is not installed")
    g = torch.Generator().manual_seed(52)
    members = [SimpleNamespace(weight=_weight(n, 256, g), bias=None, _full_precision_mm=False)
               for n in (64, 32, 32)]
    group = ops.CombinedLinear(members)
    x = torch.zeros(1, 256, dtype=torch.bfloat16, device="xpu")
    with contextlib.ExitStack() as es:
        for _ in range(2):
            _combine(group)
            committed = group._v_weight
            for m in members:
                w, cast = mono._cast(es, m, x)
                assert cast and bits_equal(w._qdata.cpu(), m.weight._qdata)
                assert bits_equal(w._params.scale.cpu(), m.weight._params.scale)
                assert bits_equal(w._params.s_channel.cpu(), m.weight._params.s_channel)
                assert w._qdata.untyped_storage().data_ptr() == committed._qdata.untyped_storage().data_ptr()
                assert group._v_weight is committed and m._v_weight is None
            for m in members:
                del m._prefetch
            # A stale combined object outside its prefetch window must not supply a view.
            w, cast = mono._cast(es, members[0], x)
            assert w is members[0].weight and not cast
            members[0].weight = _weight(64, 256, g)


def test_combined_mlp_matches_live_input_act(C, monkeypatch):
    ops = pytest.importorskip("comfy.ops")
    if not hasattr(ops, "CombinedLinear"):
        pytest.skip("ComfyUI CombinedLinear interface is not installed")
    from comfy.text_encoders import llama
    g = torch.Generator().manual_seed(53)
    cfg = SimpleNamespace(hidden_size=4096, intermediate_size=12288, mlp_activation="silu")
    mlp = llama.MLP(cfg, device="meta", dtype=torch.bfloat16, ops=torch.nn)
    for m, n, k in ((mlp.gate_proj, 12288, 4096), (mlp.up_proj, 12288, 4096),
                    (mlp.down_proj, 4096, 12288)):
        m.weight = torch.nn.Parameter(_weight(n, k, g), requires_grad=False)
        m._full_precision_mm = False
        m.weight_function, m.bias_function = [], []
    _combine(mlp.combined_gate_up)
    mlp.down_proj.weight = torch.nn.Parameter(mlp.down_proj.weight.to("xpu"), requires_grad=False)
    norm = llama.RMSNorm(4096, eps=1e-6, add=True, device="xpu", dtype=torch.bfloat16)
    norm.weight.data.copy_((torch.randn(4096, generator=g) * .1).bfloat16().xpu())
    original = llama.MLP.forward
    while getattr(original, mono.CLAIM, False):
        original = original.__wrapped__
    monkeypatch.setattr(llama.MLP, "forward", original)  # restore after install
    assert mono.install()
    before = mono.stats()["mono"]
    with torch.inference_mode():
        for rows in (1, 3, 8):
            x = torch.randn(1, rows, 4096, generator=g).bfloat16().xpu()
            reference = original(mlp, x, norm=norm)
            out = mlp(x, norm=norm)
            assert bits_equal(out, reference), rows
            # The layer kernel normalizes internally. Check its RMSNorm and projections
            # against the new combined input-act entry as well as the standalone MLP.
            with contextlib.ExitStack() as es:
                parts = [mono._w4(m, mono._cast(es, m, x)[0])
                         for m in mlp.combined_gate_up.linears]
                pre = C.layer_pre(x.reshape(rows, 4096), norm.scale(), norm.eps,
                                  mono._constants(x.device)[0], sum(parts, []), 0, False)
                normalized = torch.nn.functional.rms_norm(x, (4096,), norm.scale(), norm.eps)
                projected = mlp.combined_gate_up.linear_input_act(x, "rms_norm", norm.scale(), norm.eps)
                assert bits_equal(pre[0], normalized.reshape(rows, 4096)), rows
                assert bits_equal(torch.cat(pre[1:], dim=-1), projected.reshape(rows, -1)), rows
    assert mono.stats()["mono"] == before + 3
    assert all(m._v_weight is None for m in mlp.combined_gate_up.linears)
