# SPDX-License-Identifier: Apache-2.0
"""attn_dpas (tree verify rows and the draft row) against an fp64 reference. It is not bit-identical
to Kitchen's decode_gqa + tree_merge, so its error is bounded by the Kitchen kernel's own error on the
same inputs."""
import pytest
import torch

H, HK, D = 16, 4, 256


def _mask(parents):
    m = []
    for r, p in enumerate(parents):
        m.append((m[p] if p >= 0 else 0) | (1 << r))
    return m


def _reference(q, k, v, L, xk, xv, mask):  # fp64: prefix [0, L) plus the tree rows each row admits
    q, k, v, xk, xv = (t.double().cpu() for t in (q, k, v, xk, xv))
    S = q.shape[2]
    out = torch.empty(1, S, H * D, dtype=torch.float64)
    for h in range(H):
        kh = h // (H // HK)
        for s in range(S):
            rows = [t for t in range(S) if mask is not None and (mask[s] >> t) & 1]
            K = torch.cat([k[0, kh, :L], xk[0, kh, rows]])
            V = torch.cat([v[0, kh, :L], xv[0, kh, rows]])
            out[0, s, h * D:(h + 1) * D] = torch.softmax(K @ q[0, h, s] / 16.0, 0) @ V
    return out


def _kitchen(q, k, v, sl, xk, xv, mask_t):
    from omni_xpu_kernel import kitchen
    if mask_t is None:  # the draft row: plain decode attention over the prefix
        return kitchen.decode_attention(q, k, v, sl, True)[0]
    out, lse = kitchen.decode_attention(q, k, v, sl, False)
    merged = torch.empty_like(out)
    kitchen.decode_tree_merge(out, lse, q, xk, xv, mask_t, merged)
    return merged


@pytest.mark.parametrize("L,cap,parents", [
    (0, 512, [-1, 0, 1]),
    (63, 4096, [-1, 0, 1, 2, 3, 4, 5, 6]),
    (777, 4096, [-1, 0, 0, 0, 0, 1, 1, 1]),
    (3500, 11400, [-1, 0, 1, 2, 3, 4, 5, 6]),
    (11392, 11400, [-1, 0, 0, 1, 1]),  # the cache's last partial tile
    (1000, 4096, None),  # the MTP draft's single row, no tree
    (8000, 11400, None),
    # Full 8-row tiles for NMT=1/2/3 and a padded NMT=4 tile: all compiled branches.
    (1001, 4096, [-1, 0]),
    (4001, 8193, [-1, 0, 0, 1]),
    (8001, 8193, [-1, 0, 0, 1, 1, 2]),
    (8191, 8193, [-1, 0, 0, 1, 1, 2, 2]),
    # Full query packs: staged cache-capacity tail, including tail-only contexts.
    (17, 17, [-1, 0]),
    (15, 15, [-1, 0, 0, 1]),
    (31, 31, [-1, 0, 0, 1, 1, 2]),
    (8193, 8193, [-1, 0, 1, 2, 3, 4, 5, 6]),
])
def test_attn_dpas_error_within_kitchen(C, L, cap, parents):
    pytest.importorskip("omni_xpu_kernel")
    g = torch.Generator(device="cpu").manual_seed(L + cap)
    S = 1 if parents is None else len(parents)
    q = (torch.randn(1, H, S, D, generator=g) * 2).to(torch.bfloat16).xpu()
    k = (torch.randn(1, HK, cap, D, generator=g) * 1.5).to(torch.bfloat16).xpu()
    v = torch.randn(1, HK, cap, D, generator=g).to(torch.bfloat16).xpu()
    xk = (torch.randn(1, HK, S, D, generator=g) * 1.5).to(torch.bfloat16).xpu()
    xv = torch.randn(1, HK, S, D, generator=g).to(torch.bfloat16).xpu()
    sl = torch.tensor([L], dtype=torch.int32, device="xpu")
    mask = None if parents is None else _mask(parents)
    mt = None if mask is None else torch.tensor(mask, dtype=torch.int32, device="xpu")
    ref = _reference(q, k, v, L, xk, xv, mask)
    empty = torch.empty(0, dtype=torch.int32, device="xpu")
    out = C.attn_dpas(q, k, v, sl, xk, xv, empty if mt is None else mt, 0).double().cpu()
    err = (out - ref).abs()
    assert torch.isfinite(out).all()
    if L > 0:
        kerr = (_kitchen(q, k, v, sl, xk, xv, mt).double().cpu().reshape(ref.shape) - ref).abs()
        assert err.max() <= 1.5 * kerr.max() + 1e-3 and err.mean() <= 1.5 * kerr.mean() + 1e-5
    else:  # no prefix: only the tree rows; one BF16 rounding of the output
        assert err.max() <= 1.6e-2
