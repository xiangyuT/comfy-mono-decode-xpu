# SPDX-License-Identifier: Apache-2.0
"""rope_kv against the documented arithmetic (each product rounded to float, then the sum, then BF16)
and the K/V cache write at the given positions."""
import torch

from conftest import bits_equal


def _reference(x, f, rd):
    # x: [S, H, D] bf16 (CPU), f: [S, rd/2, 2, 2] float32 (CPU); separate float32 ops = no FMA
    half = rd // 2
    t = x.float()
    lo, hi = t[..., :half], t[..., half:rd]
    f00, f01 = f[:, None, :, 0, 0], f[:, None, :, 0, 1]
    f10, f11 = f[:, None, :, 1, 0], f[:, None, :, 1, 1]
    p, q = lo * f00, hi * f01
    o0 = p + q
    p, q = hi * f11, lo * f10
    o1 = p + q
    out = x.clone()
    out[..., :half] = o0.to(torch.bfloat16)
    out[..., half:rd] = o1.to(torch.bfloat16)
    return out


def test_rope_kv_matches_reference(C):
    g = torch.Generator(device="cpu").manual_seed(0)
    rd, D, Hq, Hk, cap = 64, 256, 16, 4, 64
    for t in range(60):
        S = (1, 3, 8)[t % 3]
        qn = (torch.randn(S, Hq, D, generator=g) * torch.exp(torch.randn(S, Hq, D, generator=g) * 2)).to(torch.bfloat16)
        kn = (torch.randn(S, Hk, D, generator=g) * 30).to(torch.bfloat16)
        vn = torch.randn(S, Hk, D, generator=g).to(torch.bfloat16)
        f = torch.randn(S, rd // 2, 2, 2, generator=g)
        pos = torch.randperm(cap, generator=g)[:S]
        key = torch.zeros(1, Hk, cap, D, dtype=torch.bfloat16, device="xpu")
        val = torch.zeros_like(key)
        q, k = C.rope_kv(qn.xpu(), kn.xpu(), vn.xpu(), f.xpu(), key, val, pos.xpu(), rd)
        rq, rk = _reference(qn, f, rd), _reference(kn, f, rd)
        assert bits_equal(q.cpu(), rq.transpose(0, 1).unsqueeze(0).contiguous())
        assert bits_equal(k.cpu(), rk.transpose(0, 1).unsqueeze(0).contiguous())
        ek = torch.zeros(1, Hk, cap, D, dtype=torch.bfloat16)
        ev = torch.zeros_like(ek)
        ek[0, :, pos] = rk.transpose(0, 1)
        ev[0, :, pos] = vn.transpose(0, 1)
        assert bits_equal(key.cpu(), ek) and bits_equal(val.cpu(), ev)
