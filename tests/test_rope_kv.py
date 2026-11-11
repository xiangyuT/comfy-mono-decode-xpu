"""rope_kv vs the eager apply_partial_rope + FixedKVCache.write, on dumped real inputs and synthetic ones."""
import sys, os
import torch
sys.path.insert(0, os.environ.get("ROPEMOD", "/artifacts/exp/ropemod"))
if os.environ.get("ROPEMOD"):
    import mono_decode_xpu_rope as C
else:
    sys.path.insert(0, "/work"); from comfy_mono_decode_xpu import _C as C
dev = torch.device("xpu")
dumps = torch.load("/dev/shm/rope_dumps.pt")
bad = n = 0
cap = 512
for i, d in enumerate(dumps):
    xq, xk, f = d["xq"].to(dev), d["xk"].to(dev), d["f"].to(dev)
    S = xq.shape[2]
    qn = xq.transpose(1, 2).contiguous()[0]; kn = xk.transpose(1, 2).contiguous()[0]
    vn = torch.randn_like(kn)
    key = torch.zeros(1, 4, cap, 256, dtype=torch.bfloat16, device=dev); val = torch.zeros_like(key)
    pos = torch.arange(100 + i, 100 + i + S, device=dev)
    q, k = C.rope_kv(qn, kn, vn, f.reshape(-1, 32, 2, 2), key, val, pos, d["rd"])
    rk = torch.zeros_like(key); rv = torch.zeros_like(key)
    rk.index_copy_(2, pos, d["ok"].to(dev)); rv.index_copy_(2, pos, vn.unsqueeze(0).transpose(1, 2))
    b = (q.view(torch.int16) != d["oq"].to(dev).view(torch.int16)).sum().item() + (k.view(torch.int16) != d["ok"].to(dev).view(torch.int16)).sum().item()
    b += (key.view(torch.int16) != rk.view(torch.int16)).sum().item() + (val.view(torch.int16) != rv.view(torch.int16)).sum().item()
    bad += b; n += 1
print(f"ROPE dumps {n} mismatches {bad}")
# synthetic: wide-range inputs through the same eager function on device
sys.path.insert(0, "/llm/ComfyUI")
from comfy_kitchen.backends.eager.rope import apply_rope_split_half1
import comfy_kitchen
g = torch.Generator(device="cpu").manual_seed(0); sb = db = ed = 0
for t in range(200):
    S = [1, 8, 3][t % 3]
    qn = (torch.randn(S, 16, 256, generator=g) * torch.exp(torch.randn(S, 16, 256, generator=g) * 2)).to(torch.bfloat16).to(dev)
    kn = (torch.randn(S, 4, 256, generator=g) * 30).to(torch.bfloat16).to(dev)
    ang = torch.rand(S, 32, generator=g) * 1000
    c, s_ = torch.cos(ang), torch.sin(ang)
    f = torch.stack([torch.stack([c, -s_], -1), torch.stack([s_, c], -1)], -2).to(dev)  # [S, 32, 2, 2]
    key = torch.zeros(1, 4, cap, 256, dtype=torch.bfloat16, device=dev); val = torch.zeros_like(key)
    pos = torch.arange(S, device=dev)
    q, k = C.rope_kv(qn, kn, kn, f, key, val, pos, 64)
    xq = qn.unsqueeze(0).transpose(1, 2); xk = kn.unsqueeze(0).transpose(1, 2)
    fm = f.view(1, 1, S, 32, 2, 2)
    rq = torch.cat([apply_rope_split_half1(xq[..., :64], fm), xq[..., 64:]], -1)
    rk = torch.cat([apply_rope_split_half1(xk[..., :64], fm), xk[..., 64:]], -1)
    dq, dk = comfy_kitchen.apply_rope_split_half(xq[..., :64], xk[..., :64], fm.unsqueeze(0) if fm.ndim == 5 else fm)
    dq = torch.cat([dq, xq[..., 64:]], -1); dk = torch.cat([dk, xk[..., 64:]], -1)
    db += (q.view(torch.int16) != dq.view(torch.int16)).sum().item() + (k.view(torch.int16) != dk.view(torch.int16)).sum().item()
    ed += (rq.view(torch.int16) != dq.view(torch.int16)).sum().item()
    sb += (q.view(torch.int16) != rq.view(torch.int16)).sum().item() + (k.view(torch.int16) != rk.view(torch.int16)).sum().item()
print(f"ROPE synthetic vs eager {sb}  vs dispatch {db}  eager-vs-dispatch(q) {ed}")
