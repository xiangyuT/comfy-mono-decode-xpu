# SPDX-License-Identifier: Apache-2.0
"""rot_quant (ConvRot rotation + row-wise INT8 quantization) against comfy-kitchen's eager reference."""
import pytest
import torch


def test_rot_quant_matches_kitchen_eager(C):
    pytest.importorskip("comfy_kitchen")
    from comfy_kitchen.backends.eager.quantization import quantize_and_rotate_rowwise
    from comfy_kitchen.tensor.int8_utils import _build_hadamard
    h = _build_hadamard(256, device="xpu", dtype=torch.bfloat16)
    g = torch.Generator(device="cpu").manual_seed(0)
    for K in (4096, 12288):
        for M in (1, 3, 8):
            for kind in ("normal", "outlier", "wide", "tiny", "zero"):
                for _ in range(4):
                    x = torch.randn(M, K, generator=g)
                    if kind == "outlier":
                        x[:, torch.randint(0, K, (8,), generator=g)] *= 300
                    if kind == "wide":
                        x = x * torch.exp(torch.randn(M, K, generator=g) * 3)
                    if kind == "tiny":
                        x = x * 1e-36
                    if kind == "zero":
                        x = x * 0
                    x = x.to(torch.bfloat16).xpu()
                    qr, sr = quantize_and_rotate_rowwise(x, h, 256)
                    q, s = C.rot_quant(x, h, 3)
                    assert torch.equal(q, qr), (K, M, kind)
                    assert torch.equal(s.view(-1), sr.view(-1).float()), (K, M, kind)
