# SPDX-License-Identifier: Apache-2.0
"""fast_topk / fast_pick against torch XPU topk / argmax, including ties, +-0 and all-equal rows."""
import pytest
import torch

from conftest import bits_equal

V = 248320


def _rows():
    g = torch.Generator(device="cpu").manual_seed(0)
    for trial in range(120):
        x = (torch.randn(1, V, generator=g) * 3).to(torch.bfloat16)
        if trial % 2:
            x = (torch.round(x.float() * 2) / 2).to(torch.bfloat16)  # many ties
        if trial % 3 == 0:
            x[0, torch.randint(0, V, (6,), generator=g)] = x.max()
        if trial % 7 == 0:
            x[0, torch.randint(0, V, (3,), generator=g)] = 0.0
            x[0, 5] = -0.0
        if trial % 11 == 0:
            x = torch.zeros(1, V, dtype=torch.bfloat16)
        yield x


def test_topk_argmax_pick_match_torch(C):
    for x in _rows():
        x = x.xpu()
        for k in (1, 2, 3, 7, 8):
            r = x.topk(k, dim=-1)
            v, i = C.fast_topk(x, k, False)
            assert torch.equal(r.indices, i), k
            assert torch.equal(r.values.view(torch.int16), v.view(torch.int16)), k
        a = x.argmax(dim=-1)
        _, ai = C.fast_topk(x, 1, True)
        assert torch.equal(a.view(-1), ai.view(-1))
        v8, i8, am = C.fast_pick(x, 8)
        r8 = x.topk(8, dim=-1)
        assert torch.equal(r8.indices, i8) and torch.equal(r8.values.view(torch.int16), v8.view(torch.int16))
        assert torch.equal(a.view(-1), am.view(-1))


def test_rejects_unsupported(C):
    with pytest.raises(RuntimeError):
        C.fast_topk(torch.zeros(1, 1024, dtype=torch.bfloat16, device="xpu"), 4, False)  # row too short
    with pytest.raises(RuntimeError):
        C.fast_topk(torch.zeros(1, V, dtype=torch.bfloat16, device="xpu"), 9, False)  # k > 8


def test_masked_assign_mode_matches_indexed_assignment(C):
    from comfy_mono_decode_xpu.autotree import _MaskedAssign
    g = torch.Generator(device="cpu").manual_seed(3)
    x = torch.randn(8, 20, generator=g).xpu()
    mask = x < 0.1
    ref = x.clone()
    ref[mask] = torch.finfo(x.dtype).min
    out = x.clone()
    with _MaskedAssign():
        out[mask] = torch.finfo(x.dtype).min
    assert bits_equal(out, ref)


@pytest.mark.parametrize("top_k,top_p,min_p", [(20, 0.95, 0.0), (20, 0.95, 0.05), (0, 0.9, 0.0)])
def test_sampling_filters_unchanged(C, top_k, top_p, min_p):
    llama = pytest.importorskip("comfy.text_encoders.llama")
    from comfy_mono_decode_xpu.autotree import _MaskedAssign
    original = getattr(llama.BaseGenerate.processed_probs, "__wrapped__", llama.BaseGenerate.processed_probs)
    g = torch.Generator(device="cpu").manual_seed(top_k + int(top_p * 100))
    lg = (torch.randn(8, 248320, generator=g) * 4).xpu()
    lg[:, :64] = lg[:, :1]  # ties at the top
    ref_p, ref_i = original(None, lg.clone(), 1.0, top_k, top_p, min_p)
    with _MaskedAssign():
        p, i = original(None, lg.clone(), 1.0, top_k, top_p, min_p)
    assert bits_equal(p, ref_p) and (ref_i is None or torch.equal(i, ref_i))
