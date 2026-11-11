"""Bit-exactness + micro for mono_decode_xpu.w4a8_decode_linear vs omni_xpu_kernel w4a8_linear_prequantized.
Real checkpoint weights (layer 0 / 3 projections), synthetic int8 activations."""
import sys, json, struct, time, torch
sys.path.insert(0, "/work")
from comfy_mono_decode_xpu import _C as C
from omni_xpu_kernel import int8 as oi
import omni_xpu_kernel, safetensors.torch as st
print("omni core:", omni_xpu_kernel.__file__)
P = "/models/extra/text_encoders/qwen3.5_9b_w4a8_gptq.safetensors"
L = "model.language_model.layers."
names = {"qkv": L+"0.linear_attn.in_proj_qkv", "z": L+"0.linear_attn.in_proj_z", "out": L+"0.linear_attn.out_proj",
         "gate": L+"0.mlp.gate_proj", "down": L+"0.mlp.down_proj", "q3": L+"3.self_attn.q_proj"}
from safetensors import safe_open
f = safe_open(P, "pt")
keys = set(f.keys())
dev = "xpu"
def load(n):
    w = f.get_tensor(n+".weight").to(dev); s = f.get_tensor(n+".weight_s_rel").to(dev)
    c = f.get_tensor(n+".weight_s_channel").float().to(dev)
    cb = f.get_tensor(n+".weight_codebook").float().to(dev) if n+".weight_codebook" in keys else None
    return w, s, c, cb
res = []
ev = lambda: torch.xpu.Event(enable_timing=True)
def timeit(fn, iters=200):
    for _ in range(10): fn(0)
    torch.xpu.synchronize(); a, b = ev(), ev(); a.record()
    for i in range(iters): fn(i)
    b.record(); torch.xpu.synchronize(); return a.elapsed_time(b) / iters
for key, n in names.items():
    if n+".weight" not in keys: continue
    w, s, c, cb = load(n); N, K = w.shape[0], w.shape[1]*2
    copies = max(2, int(400e6 // (w.numel()+s.numel())) )
    ws = [(w.clone(), s.clone()) for _ in range(copies)]
    for M in (1, 2, 3, 4, 8):
        g = torch.Generator(device="cpu").manual_seed(M*7+N)
        x = torch.randint(-127, 128, (M, K), generator=g, dtype=torch.int8).to(dev)
        xs = (torch.rand(M, generator=g)*0.05+1e-3).to(dev)
        ref = oi.w4a8_linear_prequantized(x, xs, w, s, c, codebook=cb, bias=None, group_size=16, out_dtype=torch.bfloat16)
        best = None
        for cols, fast in ([(108, True), (104, True), (116, True), (108, False)] if M == 8 else [(12, True), (11, True), (14, True), (12, False)]):
            out = C.w4a8_decode_linear(x, xs, w, s, c, cb, None, 16, 2, cols, fast)
            exact = torch.equal(out.view(torch.int16), ref.view(torch.int16))
            t = timeit(lambda i: C.w4a8_decode_linear(x, xs, ws[i % copies][0], ws[i % copies][1], c, cb, None, 16, 2, cols, fast))
            if best is None or t < best[1]: best = (f"{cols}{'f' if fast else ''}", t, exact)
            if not exact: print("MISMATCH", key, M, cols, fast, (out.float()-ref.float()).abs().max().item())
        tr = timeit(lambda i: oi.w4a8_linear_prequantized(x, xs, ws[i % copies][0], ws[i % copies][1], c, codebook=cb, bias=None, group_size=16, out_dtype=torch.bfloat16))
        byts = w.numel() + s.numel()
        r = {"proj": key, "N": N, "K": K, "M": M, "ref_ms": tr, "mono_decode_xpu_ms": best[1], "cols": best[0], "exact": best[2],
             "ref_GBs": byts/tr/1e6, "mono_decode_xpu_GBs": byts/best[1]/1e6}
        print(json.dumps(r), flush=True); res.append(r)
# same-card read roof: 1 GiB random
buf = torch.randint(-2**31, 2**31-1, (2**28,), dtype=torch.int32, device=dev)
for wgs in (2048, 4096, 8192):
    t = timeit(lambda i: C.read_checksum(buf, wgs), iters=20)
    print(json.dumps({"read_wgs": wgs, "GBs": buf.numel()*4/t/1e6}))
ck = C.read_checksum(buf, 4096).to(torch.int64).sum().item() & 0xffffffff
host = buf.cpu().view(torch.int32).to(torch.int64).sum().item() & 0xffffffff
print("checksum ok", ck == host)
json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "/artifacts/runs/micro.json", "w"), indent=1)
