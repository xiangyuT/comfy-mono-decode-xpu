# SPDX-License-Identifier: Apache-2.0
"""mtp_accept / mtp_prep_* (csrc/mtp_loop.sycl) against upstream's host logic: the accept walk, the
committed tokens, auto-tree ranks, the next token / hidden row, the counters, the deferred-delta ctl
(pending, slots) and FixedKVState.commit_path's attention KV slot moves of an off-chain path."""
import random

import pytest
import torch

from comfy_mono_decode_xpu.mtp_loop import _upstream_walk

SLOTS, H, HK, CAP, D = 8, 64, 4, 96, 256


def _tree(spec, rows):
    """VerifyTree's parent / children / depth / chain for a nested child-list spec (qwen35)."""
    parent, children, depth = [], [], []

    def add(node, p):
        r = len(parent)
        parent.append(p)
        children.append([])
        depth.append(0 if p < 0 else depth[p] + 1)
        if p >= 0:
            children[p].append(r)
        for child in node:
            add(child, r)
    add(spec, -1)
    chain = 0
    while children[chain]:
        chain = children[chain][0]
    while len(parent) < rows:
        parent.append(-1)
        children.append([])
        depth.append(0)
    return parent, children, depth, chain


def _spec(widths):  # qwen35.tree_spec
    node = []
    for width in reversed(widths):
        node = [node] + [[] for _ in range(width - 1)]
    return node


SPECS = [_spec(w) for w in ((4, 3), (7,), (1, 1, 1, 1, 1, 1, 1), (2, 2, 3), (3, 1, 1, 2))] + \
        [[[[[[]], []], []], []], [[[[[]]]]]]  # tree4s1m8, chain 4


@pytest.mark.parametrize("spec_index", range(len(SPECS)))
def test_mtp_accept_matches_host_logic(C, spec_index):
    rng = random.Random(1000 + spec_index)
    parent, children, depth_of, chain = _tree(SPECS[spec_index], SLOTS)
    rows = SLOTS
    dev = "xpu"
    ch = torch.full((SLOTS, SLOTS), -1, dtype=torch.int32)
    for r, cs in enumerate(children):
        for j, c in enumerate(cs):
            ch[r, j] = c
    ch = ch.reshape(-1).to(dev)
    for trial in range(40):
        # tokens: the verify's choice at each row, the draft tokens; plant matches so paths go deep and off-chain
        toks = [rng.randrange(50) for _ in range(rows)]
        dr = [rng.randrange(50) for _ in range(rows)]
        for r in range(rows):
            if children[r] and rng.random() < 0.8:
                dr[rng.choice(children[r])] = toks[r]
        cand = [rng.randrange(50) for _ in range(49)]
        for node in range(chain):
            if rng.random() < 0.7:
                cand[7 * node + rng.randrange(7)] = toks[node]
        pos, mtp_pos = rng.randrange(0, 40), rng.randrange(0, 30)
        path, commit, ranks, row, on_chain, kept = _upstream_walk(children, toks, dr, chain, cand)

        st = torch.tensor([pos, mtp_pos, 0, 0], dtype=torch.long, device=dev)
        t_toks = torch.tensor([toks], dtype=torch.long, device=dev)
        t_dr = torch.tensor([dr], dtype=torch.long, device=dev)
        t_cand = torch.tensor(cand, dtype=torch.long, device=dev).view(7, 7)
        x = torch.randn(1, rows, H).to(torch.bfloat16).to(dev)
        nt, hb = torch.empty(1, 1, dtype=torch.long, device=dev), torch.empty(1, 1, H, dtype=torch.bfloat16, device=dev)
        ctl = torch.zeros(2 + 3 * SLOTS, dtype=torch.int32, device=dev)
        keys = [torch.randn(1, HK, CAP, D).to(torch.bfloat16).to(dev) for _ in range(4)]
        before = [k.clone() for k in keys]
        ptrs = torch.tensor([p - (1 << 64) if p >= (1 << 63) else p for p in (k.data_ptr() for k in keys)],
                            dtype=torch.long, device=dev)
        result = torch.zeros(32, dtype=torch.long, device=dev)
        C.mtp_accept(st, t_toks, t_dr, t_cand, ch, rows, chain, x, nt, hb, ctl, ptrs, HK, CAP, D, result)
        r = result.tolist()
        assert r[0] == len(path) and r[1:1 + len(path)] == commit and r[9] == row and bool(r[10]) == on_chain
        assert r[12:12 + r[11]] == ranks and r[20] == kept and r[19] == pos
        assert st.tolist()[:2] == [pos + len(path), mtp_pos + kept]
        c = ctl.tolist()
        assert c[0] == len(path) and c[2:2 + SLOTS] == path + list(range(len(path), SLOTS))
        assert nt.item() == toks[row] and torch.equal(hb.view(-1), x[0, row])
        # FixedKVState.commit_path on the host copies
        for k_before, k_after in zip(before, keys):
            ref = k_before.clone()
            if not on_chain:
                for k, src in enumerate(path[1:], 1):
                    if src != k:
                        ref[:, :, (pos + k) % CAP] = ref[:, :, (pos + src) % CAP]
            assert torch.equal(ref.view(torch.int16), k_after.view(torch.int16))


def test_mtp_prep_matches_host_prepare(C):
    dev = "xpu"
    st = torch.tensor([37, 11, 0, 0], dtype=torch.long, device=dev)
    ftab = torch.randn(1, 1, 64, 32, 2, 2, device=dev)
    freqs = torch.empty(1, 1, 1, 32, 2, 2, device=dev)
    mpos, mseq = torch.empty(SLOTS, dtype=torch.long, device=dev), torch.empty(1, dtype=torch.int32, device=dev)
    C.mtp_prep_draft(st, 2, ftab, freqs, mpos, mseq)
    assert torch.equal(freqs, ftab[:, :, 39:40]) and mpos.tolist() == list(range(13, 21)) and mseq.item() == 14
    apos, aseq = torch.empty(SLOTS, dtype=torch.long, device=dev), torch.empty(1, dtype=torch.int32, device=dev)
    vpos = torch.empty(1, 8, dtype=torch.long, device=dev)
    tpos = torch.tensor([0, 1, 2, 3, 4, 1, 2, 1], dtype=torch.long, device=dev)
    ctl = torch.zeros(2 + 3 * SLOTS, dtype=torch.int32, device=dev)
    tctl = torch.arange(100, 116, dtype=torch.int32, device=dev)
    C.mtp_prep_verify(st, apos, aseq, vpos, tpos, ctl, tctl, 1, 5)
    assert apos.tolist() == list(range(37, 45)) and aseq.item() == 37
    assert vpos.view(-1).tolist() == [37 + 5 + d for d in tpos.tolist()]
    assert ctl.tolist()[1] == 1 and ctl.tolist()[10:] == list(range(100, 116))
