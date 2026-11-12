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


def test_qk_norm_rope_kv_matches_live_chain(C):
    from omni_xpu_kernel import norm
    g = torch.Generator().manual_seed(211)
    with torch.inference_mode():
        for i in range(108):
            S = (1, 3, 8)[i % 3]
            Hq, Hk = ((16, 4), (4, 2), (1, 1))[(i // 3) % 3]
            rd = (64, 128, 256)[(i // 9) % 3]
            kind = i // 27
            qg = torch.randn(1, S, Hq, 512, generator=g)
            k = torch.randn(1, S, Hk, 256, generator=g)
            if kind == 1:
                qg *= torch.exp(torch.randn(qg.shape, generator=g) * 2)
                k *= 30
            elif kind == 2:
                qg *= 1e-4
                k *= 1e-4
            elif kind == 3:
                qg.zero_()
                qg[..., ::2] = -0.0
                k.zero_()
                k[..., ::2] = -0.0
            qg, k = qg.bfloat16().xpu(), k.bfloat16().xpu()
            q = qg[..., :256]
            v = torch.randn(k.shape, generator=g).bfloat16().xpu()
            wq = (1 + torch.randn(256, generator=g) * .3).bfloat16().xpu()
            wk = (1 + torch.randn(256, generator=g) * .3).bfloat16().xpu()
            f = torch.randn(S, rd // 2, 2, 2, generator=g)
            pos = torch.randperm(64, generator=g)[:S]
            key = torch.randn(1, Hk, 64, 256, generator=g).bfloat16()
            val = torch.randn(1, Hk, 64, 256, generator=g).bfloat16()
            # Dense degenerate Q uses the same Omni norm as K; other gated Q views
            # use the actual Torch reduction. Neither reference uses the new kernel.
            nq = (norm.rms_norm(wq, q.reshape(-1, 256), 1e-6).reshape(q.shape) if q.is_contiguous()
                  else torch.nn.functional.rms_norm(q, (256,), wq, 1e-6))
            nk = norm.rms_norm(wk, k.reshape(-1, 256), 2e-5).reshape(k.shape)
            rq, rk = _reference(nq.cpu()[0], f, rd), _reference(nk.cpu()[0], f, rd)
            expected_key, expected_val = key.clone(), val.clone()
            expected_key[0, :, pos] = rk.transpose(0, 1)
            expected_val[0, :, pos] = v.cpu()[0].transpose(0, 1)
            key, val = key.xpu(), val.xpu()
            outq, outk = C.qk_norm_rope_kv(q, k, v, wq, wk, f.xpu(), key, val, pos.xpu(), 1e-6, 2e-5, rd)
            assert outq.is_contiguous() and outk.is_contiguous()
            assert bits_equal(outq.cpu(), rq.transpose(0, 1).unsqueeze(0).contiguous()), i
            assert bits_equal(outk.cpu(), rk.transpose(0, 1).unsqueeze(0).contiguous()), i
            assert bits_equal(key.cpu(), expected_key) and bits_equal(val.cpu(), expected_val), i
