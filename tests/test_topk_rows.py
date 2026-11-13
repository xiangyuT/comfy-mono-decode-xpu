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
