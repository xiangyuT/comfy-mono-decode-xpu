"""Exactness of ported RMSNorm / BF16 linear candidates vs the eager decode ops (real weights)."""
import sys
sys.path.insert(0, "/artifacts/scripts"); sys.path.insert(0, "/work")
from load_te import load
clip, tm = load()
import torch, comfy.ldm.common_dit as cd
from comfy_mono_decode_xpu import _C as C
dev = torch.device("xpu")
g = torch.Generator(device="cpu").manual_seed(5)
L = tm.model.layers
def xs(R, kind):
    x = torch.randn(R, 4096, generator=g)
    if kind == "outlier": x[:, torch.randint(0, 4096, (8,), generator=g)] *= 300
    if kind == "wide": x = x * torch.exp(torch.randn(R, 4096, generator=g) * 2)
    if kind == "small": x = x * 1e-3
    return x.to(torch.bfloat16).to(dev)
if False: print("== RMSNorm (input_layernorm / post_attention_layernorm, add=True)")
res = {}
for li in ():
    for nm in ("input_layernorm", "post_attention_layernorm"):
        norm = getattr(L[li], nm)
        w1 = norm.scale().contiguous()
        for R in (1, 3, 8):
            for kind in ("normal", "outlier", "wide", "small"):
                x = xs(R, kind)
                ref = norm(x)
                for v in (1, 7, 0, 6):
                    o = C.rms_norm_probe(x, w1, norm.eps, v)
                    res[v] = res.get(v, 0) + int((o.view(torch.int16) != ref.view(torch.int16)).sum())
if False: print("RMS mismatches by variant (bit0 fma, bits1-2 rsqrt native/precise/1/sqrt):", res, flush=True)
print("== in_proj_a / in_proj_b BF16 linear vs at::linear")
lres = {}
for li in [i for i in range(32) if (i + 1) % 4]:
    gdn = L[li].linear_attn
    for nm in ("in_proj_a", "in_proj_b"):
        w = getattr(gdn, nm).weight
        w = (w.dequantize() if hasattr(w, "dequantize") else w).to(torch.bfloat16).contiguous()
        for R in (1, 2, 3, 4, 5, 6, 7, 8):
            for kind in ("normal", "outlier", "wide", "normal", "wide", "small", "normal", "wide"):
                x = xs(R, kind)
                ref = torch.nn.functional.linear(x.view(1, R, 4096), w).view(R, -1)
                for v in (1164, 1165):
                    o = C.linear_probe(x, w, v)
                    lres[v] = lres.get(v, 0) + int((o.view(torch.int16) != ref.view(torch.int16)).sum())
print("LIN mismatches by variant:", lres, "of", 4 * 2 * 3 * 3 * 4 * 32, flush=True)
