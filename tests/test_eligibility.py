# SPDX-License-Identifier: Apache-2.0
"""Eligibility checks never raise: weights that are not the supported packed W4A8 layout are rejected."""
import types

import torch

from comfy_mono_decode_xpu import mono, tiles


def _lin(params, qdata, **extra):
    w = types.SimpleNamespace(_params=params, _qdata=qdata)
    return types.SimpleNamespace(weight=w, bias=None, **extra)


def test_int8_tensorwise_params_rejected():
    # INT8 tensor-wise Params have only scale / orig_dtype (no group_size, codebook, s_channel)
    p = types.SimpleNamespace(scale=torch.ones(16, 1), orig_dtype=torch.bfloat16)
    lin = _lin(p, torch.zeros(16, 256, dtype=torch.int8))
    assert mono._layout_ok(lin.weight) is False
    assert mono._static_ok(lin) is False
    assert mono._w4(lin, lin.weight) is None


def test_missing_fields_and_plain_tensors_rejected():
    assert mono._layout_ok(None) is False
    assert mono._layout_ok(torch.zeros(4, 4)) is False
    lin = _lin(object(), torch.zeros(16, 128, dtype=torch.int8))
    assert mono._static_ok(lin) is False


def test_tile_major_is_a_permutation():
    t = torch.arange(32 * 64, dtype=torch.int32).view(32, 64).to(torch.uint8)
    u = tiles.tile_major(t, 16)
    assert u.shape == t.shape
    assert torch.equal(torch.sort(u.flatten().to(torch.int32)).values, torch.sort(t.flatten().to(torch.int32)).values)
    # first 256 bytes of the copy = rows 0..15, columns 0..15 of the source
    assert torch.equal(u.flatten()[:256].view(16, 16), t[:16, :16])


def test_valid_w4a8_and_malformed_scale_are_distinguished():
    from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout
    p = AsymW4A8Int8Layout.Params(
        scale=torch.ones(16, 16).to(torch.float8_e4m3fn), orig_dtype=torch.bfloat16,
        orig_shape=(16, 256), s_channel=torch.ones(16), codebook=torch.arange(16).float())
    lin = _lin(p, torch.zeros(16, 128, dtype=torch.int8))
    assert mono._layout_ok(lin.weight)
    # Simulate malformed external metadata beyond the frozen Params constructor.
    object.__setattr__(p, "scale", p.scale[:, :8].contiguous())
    assert not mono._layout_ok(lin.weight)
