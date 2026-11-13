# SPDX-License-Identifier: Apache-2.0
"""draft_pick (csrc/draft_lmhead.sycl), the second stage of the two-stage draft lm_head: the exact int8
logits of its candidates ((acc * x_scale) * w_scale in fp32, rounded to bf16) and torch XPU's topk tie
order / argmax rules, against torch on the exact logits, with approximate scores that are exact, noisy,
or tie-heavy."""
import pytest
import torch

N, K = 60000, 1024


def _case(seed, ties):
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(-127, 128, (N, K), generator=g, dtype=torch.int8)
    sw = torch.rand(N, generator=g) * 1e-3 + 1e-4
    if ties:  # duplicate the strongest rows across the vocabulary: exact ties at the top
        qx0 = torch.randint(-127, 128, (K,), generator=g, dtype=torch.int8)
        acc0 = (w.long() * qx0.long()).sum(-1)
        best = acc0.topk(4).indices
        for j, r in enumerate(best.tolist()):
            for d in range(1, 4):
                dst = (r + d * 12347 + j * 999) % N
                w[dst], sw[dst] = w[r], sw[r]
        qx = qx0
    else:
        qx = torch.randint(-127, 128, (K,), generator=g, dtype=torch.int8)
    sx = torch.rand(1, generator=g) * 0.05 + 0.01
    acc = (w.long() * qx.long()).sum(-1).float()
    exact = ((acc * sx) * sw).bfloat16()
    return w, sw, qx, sx, exact


@pytest.mark.parametrize("seed,ties,noise", [(1, False, 0.0), (2, False, 0.02), (3, True, 0.0), (4, True, 0.01)])
def test_draft_pick_matches_torch(C, seed, ties, noise):
    dev = "xpu"
    w, sw, qx, sx, exact = _case(seed, ties)
    g = torch.Generator().manual_seed(seed + 100)
    scale = exact.float().abs().max()
    approx = (exact.float() + noise * scale * torch.randn(N, generator=g)).bfloat16()
    ex = exact.to(dev)
    ref_top = ex.topk(7).indices
    ref_am = ex.argmax()
    w, sw, qx, sx, approx = w.to(dev), sw.to(dev), qx.to(dev), sx.to(dev), approx.to(dev)
    calls = torch.zeros(1, dtype=torch.long, device=dev)
    rows = 8
    for nout, argmax_first, children in ((7, False, [1, 2, 5]), (3, True, [3, 4, 6]), (1, True, [2])):
        top = torch.full((7,), -1, dtype=torch.long, device=dev)
        ranks = torch.full((7,), -1, dtype=torch.long, device=dev)
        token = torch.full((1,), -1, dtype=torch.long, device=dev)
        vtok = torch.full((rows,), -1, dtype=torch.long, device=dev)
        C.draft_pick(approx, qx, sx, w, sw, nout, argmax_first, top, ranks, token, vtok, children, calls)
        want = ref_top[:nout].tolist()
        assert top.tolist()[:nout] == want and ranks.tolist()[:nout] == want
        tok = int(ref_am) if argmax_first else want[0]
        assert token.item() == tok
        expect_v = [-1] * rows
        expect_v[children[0]] = tok
        for j, c in enumerate(children[1:], 1):
            expect_v[c] = want[j]
        assert vtok.tolist() == expect_v
    assert calls.item() == 3
    # exact scores (no weight): the pick reads them as the logits
    if noise == 0.0:
        e8, ef = torch.empty(0, dtype=torch.int8, device=dev), torch.empty(0, dtype=torch.float32, device=dev)
        top = torch.full((7,), -1, dtype=torch.long, device=dev)
        token = torch.full((1,), -1, dtype=torch.long, device=dev)
        vtok = torch.full((8,), -1, dtype=torch.long, device=dev)
        C.draft_pick(ex.contiguous(), e8, ef, e8, ef, 7, False, top, top[:0], token, vtok, [1, 2], calls)
        assert top.tolist() == ref_top.tolist() and token.item() == ref_top[0].item()
