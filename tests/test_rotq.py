import sys, torch
sys.path.insert(0, "/work")
from comfy_mono_decode_xpu import _C as C
from comfy_kitchen.tensor.int8_utils import _build_hadamard
from comfy_kitchen.backends.eager.quantization import quantize_and_rotate_rowwise
h = _build_hadamard(256, device="xpu", dtype=torch.bfloat16)
# formula check
k = torch.arange(256); j = torch.arange(256)
neg = torch.zeros(256, 256, dtype=torch.int64)
for d in range(4):
    neg ^= ((((k[:, None] >> (2*d)) & 3) + ((j[None, :] >> (2*d)) & 3)) == 3).long()
hf = torch.where(neg.bool(), -0.0625, 0.0625).to(torch.bfloat16)
print("H formula equal:", torch.equal(hf, h.cpu()))
g = torch.Generator(device="cpu").manual_seed(0)
cases = []
for K in (4096, 8192, 12288):
    for M in (1, 2, 4, 8):
        for kind in ("normal", "outlier", "wide", "tiny", "zero"):
            for rep in range(20):
                x = torch.randn(M, K, generator=g)
                if kind == "outlier": x[:, torch.randint(0, K, (8,), generator=g)] *= 300
                if kind == "wide": x = x * torch.exp(torch.randn(M, K, generator=g) * 3)
                if kind == "tiny": x = x * 1e-36
                if kind == "zero": x = x * 0
                cases.append((K, M, kind, x.to(torch.bfloat16).xpu()))
for order in (3,):
    bad_q = bad_s = tot = 0; first = None
    for K, M, kind, x in cases:
        qr, sr = quantize_and_rotate_rowwise(x, h, 256)
        q, s = C.rot_quant(x, h, order)
        dq = (q != qr).sum().item(); ds = (s.view(-1) != sr.view(-1).float()).sum().item()
        bad_q += dq; bad_s += ds; tot += x.numel()
        if (dq or ds) and first is None: first = (K, M, kind, dq, ds)
    print(f"order {order}: q mismatches {bad_q}/{tot}  scale mismatches {bad_s}  first {first}")

import time
x = torch.randn(8, 4096, dtype=torch.bfloat16, device="xpu")
for name, fn in (("eager", lambda: quantize_and_rotate_rowwise(x, h, 256)), ("cmdx", lambda: C.rot_quant(x, h, 3))):
    for _ in range(20): fn()
    torch.xpu.synchronize(); t = time.perf_counter()
    for _ in range(200): fn()
    torch.xpu.synchronize(); print(name, "us/call M8K4096 (host incl.)", (time.perf_counter()-t)/200*1e6)
