# SPDX-License-Identifier: Apache-2.0
"""w4a8_decode_linear against omni_xpu_kernel's w4a8_linear_prequantized, with random packed weights."""
import pytest
import torch

from conftest import bits_equal


def random_w4(N, K, g, dev="xpu"):
    """A random packed W4A8 weight: int4 codes, FP8 group scales (G16, bytes 0x00-0x7F), channel scales
    and a 16-entry codebook."""
    q = torch.randint(-128, 128, (N, K // 2), generator=g, dtype=torch.int8)
    s = torch.randint(0x18, 0x58, (N, K // 16), generator=g, dtype=torch.uint8).view(torch.float8_e4m3fn)
    c = torch.rand(N, generator=g) * 1e-3 + 1e-4
    cb = torch.arange(-8, 8, dtype=torch.float32)
    return q.to(dev), s.to(dev), c.to(dev), cb.to(dev)


@pytest.mark.parametrize("N,K", [(4096, 4096), (12288, 4096), (4096, 12288)])
def test_w4a8_decode_linear_matches_omni(C, N, K):
    oi = pytest.importorskip("omni_xpu_kernel.int8")
    g = torch.Generator(device="cpu").manual_seed(N + K)
    w, s, c, cb = random_w4(N, K, g)
    for M in (1, 2, 3, 8):
        x = torch.randint(-127, 128, (M, K), generator=g, dtype=torch.int8).xpu()
        xs = (torch.rand(M, generator=g) * 0.05 + 1e-3).xpu()
        ref = oi.w4a8_linear_prequantized(x, xs, w, s, c, codebook=cb, bias=None, group_size=16, out_dtype=torch.bfloat16)
        out = C.w4a8_decode_linear(x, xs, w, s, c, cb, None, 16, 2, 0, False)
        assert bits_equal(out, ref), M
