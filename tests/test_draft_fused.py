# SPDX-License-Identifier: Apache-2.0
"""Fused draft-head kernels (csrc/draft_fused.sycl) against the eager ops they replace, within BF16 rounding:
the draft only proposes tokens, so these follow the eager arithmetic in FP32 without its reduction order."""
import torch

H, I = 4096, 12288
dev = "xpu"


def _close(a, b, tol=2e-2):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item() < tol


def _rms(x, w, eps):
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=w, eps=eps)


def _hadamard():
    from comfy_kitchen.tensor.int8_utils import _build_hadamard
    return _build_hadamard(256, device=dev, dtype=torch.float32).contiguous()


def test_embed_norm_cat(C):
    g = torch.Generator().manual_seed(0)
    V = 1000
    q = torch.randint(-127, 128, (V, H), generator=g, dtype=torch.int8).to(dev)
    s = (torch.rand(V, generator=g) * 1e-3 + 1e-4).to(dev)
    hm = _hadamard()
    hidden = torch.randn(H, generator=g).bfloat16().to(dev)
    we, wh = (torch.rand(H, generator=g) + 0.5).bfloat16().to(dev), (torch.rand(H, generator=g) + 0.5).bfloat16().to(dev)
    tok = torch.tensor([123], device=dev)
    out = torch.empty(2 * H, dtype=torch.bfloat16, device=dev)
    C.draft_embed_norm_cat(tok, q, s, hm, torch.empty(0, device=dev), hidden, we, wh, 1e-6, 1e-6, out)
    row = (q[123].float() * s[123]).view(-1, 256) @ hm  # comfy_kitchen dequantize_int8_embedding (un-rotate)
    emb = row.reshape(-1).bfloat16()
    ref = torch.cat([_rms(emb, we, 1e-6), _rms(hidden, wh, 1e-6)])
    assert _close(out, ref, 1e-2)
    # a given embedding row instead of the gather
    out2 = torch.empty_like(out)
    e = torch.empty(0, device=dev)
    C.draft_embed_norm_cat(e, e, e, e, emb.contiguous(), hidden, we, wh, 1e-6, 1e-6, out2)
    assert _close(out2, ref, 1e-2)


def test_add_rmsnorm_sigmoid_silu(C):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(H, generator=g).bfloat16().to(dev)
    h = torch.randn(H, generator=g).bfloat16().to(dev)
    w = (torch.rand(H, generator=g) + 0.5).bfloat16().to(dev)
    xs = x.clone()
    out = torch.empty_like(x)
    C.draft_add_rmsnorm(xs, h, w, 1e-6, out)
    ref_x = x + h
    assert torch.equal(xs, ref_x) and _close(out, _rms(ref_x, w, 1e-6), 1e-2)
    out_n = torch.empty_like(x)
    C.draft_add_rmsnorm(xs, torch.empty(0, device=dev), w, 1e-6, out_n)  # norm only, x unchanged
    assert torch.equal(xs, ref_x) and _close(out_n, _rms(ref_x, w, 1e-6), 1e-2)
    a = torch.randn(H, generator=g).bfloat16().to(dev)
    qg = torch.randn(2 * H, generator=g).bfloat16().to(dev)
    gated = torch.empty_like(a)
    C.draft_sigmoid_mul(a, qg, 256, gated)
    gate = qg.view(16, 512)[:, 256:].reshape(-1)
    assert _close(gated, a * gate.sigmoid(), 1e-2)
    gu = torch.randn(2 * I, generator=g).bfloat16().to(dev)
    act = torch.empty(I, dtype=torch.bfloat16, device=dev)
    C.draft_silu_mul(gu, act)
    assert _close(act, torch.nn.functional.silu(gu[:I]) * gu[I:], 1e-2)


def test_norm_rot_quant(C):
    g = torch.Generator().manual_seed(2)
    x = torch.randn(H, generator=g).bfloat16().to(dev)
    d = torch.randn(H, generator=g).bfloat16().to(dev)
    w = (torch.rand(H, generator=g) + 0.5).bfloat16().to(dev)
    hm = _hadamard()
    xr = torch.empty(H, dtype=torch.bfloat16, device=dev)
    qx = torch.empty(H, dtype=torch.int8, device=dev)
    sx = torch.empty(1, dtype=torch.float32, device=dev)
    xs = x.clone()
    C.draft_norm_rot_quant(xs, d, w, 1e-6, hm, xr, qx, sx)
    assert torch.equal(xs, x + d)
    n = _rms(x + d, w, 1e-6)
    rot = (n.float().view(-1, 256) @ hm).reshape(-1)
    assert _close(xr, rot, 1e-2)
    assert _close(qx.float() * sx, rot, 2e-2)


def test_decode_embedding_gather(C):
    g = torch.Generator().manual_seed(19)
    q = torch.randint(-128, 128, (101, H), dtype=torch.int8, generator=g).xpu()
    s = torch.rand(101, 1, generator=g).xpu()
    for rows in (1, 3, 8):
        ids = (torch.arange(rows) * 13).reshape(1, rows).xpu()
        for scales in (s, s[:1]):
            out, valid = C.embedding_gather(ids, q, scales)
            ref = q[ids].float() * (scales[ids] if scales.numel() > 1 else scales)
            assert valid.item() and torch.equal(out.view(torch.int32), ref.view(torch.int32))
    out, valid = C.embedding_gather(torch.tensor([-1, 101], device="xpu"), q, s)
    assert not valid.item() and out.count_nonzero().item() == 0
