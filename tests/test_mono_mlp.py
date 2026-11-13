# SPDX-License-Identifier: Apache-2.0
"""mono_mlp (one launch) against the chain of separately tested kernels it fuses:
rot_quant -> w4a8 gate/up -> SiLU(gate) * up -> rot_quant -> w4a8 down. Row-major and tile-major
weight layouts must give the same bits."""
import pytest
import torch
import torch.nn.functional as F

from comfy_mono_decode_xpu.tiles import tile_major
from conftest import bits_equal
from test_w4a8_linear import random_w4


def test_mono_mlp_matches_chain(C):
    pytest.importorskip("comfy_kitchen")
    from comfy_kitchen.tensor.int8_utils import _build_hadamard
    g = torch.Generator(device="cpu").manual_seed(1)
    H = _build_hadamard(256, device="xpu", dtype=torch.bfloat16)
    allbf = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).xpu()
    lut = F.silu(allbf).contiguous()
    gate, up, down = random_w4(12288, 4096, g), random_w4(12288, 4096, g), random_w4(4096, 12288, g)
    up = (up[0], up[1], up[2], gate[3])      # one shared codebook, as the kernels require
    down = (down[0], down[1], down[2], gate[3])
    tiled = [(tile_major(q, 16), tile_major(s.view(torch.uint8), 16).view(torch.float8_e4m3fn), c, cb)
             for q, s, c, cb in (gate, up, down)]

    def lin(q8, s8, w):
        return C.w4a8_decode_linear(q8, s8.view(-1), w[0], w[1], w[2], w[3], None, 16, 2, 0, False)

    for M in (1, 3, 8):
        xn = (torch.randn(M, 4096, generator=g)).to(torch.bfloat16).xpu()
        q1, s1 = C.rot_quant(xn, H, 3)
        gt, u = lin(q1, s1, gate), lin(q1, s1, up)
        h = (lut[gt.view(torch.int16).long() & 0xFFFF].float() * u.float()).to(torch.bfloat16)
        q2, s2 = C.rot_quant(h.contiguous(), H, 3)
        ref = lin(q2, s2, down)
        args = [t for w in (gate, up, down) for t in w]
        out = C.mono_mlp(xn, H, lut, *args, 0, 4, False)
        assert bits_equal(out, ref), ("row-major", M)
        out_t = C.mono_mlp(xn, H, lut, *[t for w in tiled for t in w], 0, 4, True)
        assert bits_equal(out_t, ref), ("tile-major", M)

    props = torch.xpu.get_device_properties(xn.device)
    for workers in (-1, props.gpu_subslice_count + 1):
        assert bits_equal(C.mono_mlp(xn, H, lut, *args, workers, 4, False), ref)
    capacity_error = "SLM requirement" if props.device_id == 0xE20B else "tiles per work-group"
    with pytest.raises(RuntimeError, match=capacity_error):
        C.mono_mlp(xn, H, lut, *args, 1, 4, False)
