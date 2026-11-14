# SPDX-License-Identifier: Apache-2.0
"""topk_rows (csrc/topk_rows.sycl) against torch XPU topk on [R, N] fp32 rows: values and indices bit for bit,
including the order of tied values (gather order), +-0, NaN and rows with fewer distinct values than k."""
import pytest
import torch

N = 248320


def _rows(kind, R, g):
    if kind == "bf16":  # verify logits: bf16 values in fp32
        return torch.randn(R, N, generator=g).bfloat16().float()
    if kind == "ties":
        return torch.randint(0, 6, (R, N), generator=g).float()
    if kind == "few":  # moderate ties, divided like a temperature
        return (torch.randn(R, N, generator=g) * 3).round() / 0.7
    if kind == "zeros":
        x = torch.randint(-3, 3, (R, N), generator=g).float()
        x[x == 0] = -0.0
        x[:, ::97] = 0.0
        return x
    if kind == "equal":
        return torch.ones(R, N)
    x = torch.randn(R, N, generator=g).bfloat16().float()
    x[:, torch.randint(0, N, (50,), generator=g)] = float("nan")
    return x


@pytest.mark.parametrize("kind", ["bf16", "ties", "few", "zeros", "equal", "nan"])
@pytest.mark.parametrize("R,k", [(8, 20), (8, 64), (3, 7), (1, 1)])
def test_topk_rows_matches_torch(C, kind, R, k):
    g = torch.Generator().manual_seed(hash((kind, R, k)) % (1 << 31))
    x = _rows(kind, R, g).to("xpu")
    want_v, want_i = torch.topk(x, k)
    v, i = C.topk_rows(x, k)
    assert torch.equal(i, want_i)
    assert torch.equal(v.view(torch.int32), want_v.view(torch.int32))
    if kind not in ("few", "nan"):  # BF16-valued rows: the BF16 input gives torch.topk of its FP32 values
        xb = x.bfloat16()           # (NaN rows are not reproduced: torch's own top-k there depends on its gather)
        want_v, want_i = torch.topk(xb.float(), k)
        v, i = C.topk_rows(xb, k)
        assert torch.equal(i, want_i) and torch.equal(v.view(torch.int32), want_v.view(torch.int32))


@pytest.mark.parametrize("T,top_k,top_p,min_p", [(1.0, 20, 0.95, 0.0), (0.7, 64, 0.95, 0.05), (1.0, 0, 0.9, 0.0),
                                                 (1.0, 64, 1.0, 0.1), (0.5, 7, 0.8, 0.0)])
def test_verify_rows_sampling_matches_upstream(C, T, top_k, top_p, min_p):
    """ComfyUI's sample_token on the lazily served BF16 rows (sampling.verify_rows) against the FP32 rows
    without the mode: the same tokens from the same generator."""
    llama = pytest.importorskip("comfy.text_encoders.llama")
    from comfy_mono_decode_xpu import sampling
    g0 = torch.Generator().manual_seed(7)
    lg = (torch.randn(8, N, generator=g0) * 4).bfloat16().to("xpu")
    gen = object.__new__(llama.BaseGenerate)
    for seed in range(3):
        ref = llama.BaseGenerate.sample_token(gen, lg.float(), T, top_k, top_p, min_p, 1.0, [],
                                              torch.Generator(device="xpu").manual_seed(seed))
        ctx, rows = sampling.verify_rows(C, lg)
        with ctx:
            out = llama.BaseGenerate.sample_token(gen, rows, T, top_k, top_p, min_p, 1.0, [],
                                                  torch.Generator(device="xpu").manual_seed(seed))
        assert torch.equal(out, ref)
        fused_generator = torch.Generator(device="xpu").manual_seed(seed)
        fused = sampling.fused_sample(C, lg, dict(temperature=T, top_k=top_k, top_p=top_p, min_p=min_p), fused_generator)
        if T == 1.0 and 1 <= top_k <= 64:
            expected_generator = torch.Generator(device="xpu").manual_seed(seed)
            expected = llama.BaseGenerate.sample_token(gen, lg.float(), T, top_k, top_p, min_p, 1.0, [], expected_generator)
            assert torch.equal(fused, expected)
            assert torch.equal(fused_generator.get_state(), expected_generator.get_state())
        else:
            assert fused is None


def test_sampling_cutoff_boundaries(C):
    """Make the token at a top-p cutoff win whenever it survives filtering.

    Random multinomial trials rarely observe a one-ulp prefix-sum difference;
    these fixed exponential samples make that boundary directly observable.
    """
    llama = pytest.importorskip("comfy.text_encoders.llama")
    model = object.__new__(llama.BaseGenerate)
    g = torch.Generator().manual_seed(411)
    for k in (7, 17, 20, 32, 64):
        values = torch.randn(1, k, generator=g).bfloat16().float().sort(descending=True).values.xpu()
        ids = torch.arange(k, device="xpu").reshape(1, k)
        cdf = values.softmax(-1).cumsum(-1)
        for cut in (1, k // 2, k - 1):
            boundary = cdf[0, cut - 1]
            thresholds = (torch.nextafter(boundary, torch.full_like(boundary, -float("inf"))), boundary,
                          torch.nextafter(boundary, torch.full_like(boundary, float("inf"))))
            noise = torch.ones_like(values)
            noise[0, cut] = 1e-15
            for threshold in thresholds:
                top_p = float(threshold)
                ref, _ = model.processed_probs(values.clone(), 1., 0, top_p, 0.)
                expected = (ref / noise).argmax(-1, keepdim=True)
                out, valid = C.sampling_pick(values, ids, noise, top_p, 0.)
                assert valid.item() and torch.equal(out, expected), (k, cut, top_p)
