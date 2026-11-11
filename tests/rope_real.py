# SPDX-License-Identifier: Apache-2.0
"""rope_kv variants vs the live dispatch of q35.apply_partial_rope (OmniXPU kitchen backend registered)."""
import sys, os
sys.path.insert(0, "/artifacts/scripts")
from load_te import load
clip, tm = load()
import torch, importlib.util
from comfy.text_encoders import qwen35 as q35
mods = {}
for v in (int(os.environ.get("ROPEV", "0")),):
    spec = importlib.util.spec_from_file_location("mono_decode_xpu_rope", f"/artifacts/exp/ropemod{v}/mono_decode_xpu_rope.so")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); mods[v] = m
dev = torch.device("xpu")
g = torch.Generator(device="cpu").manual_seed(0)
table = torch.randn(1, 1, 4000, 32, 2, 2, generator=g).to(dev)  # wide-range factors, sliced like ftab
bad = {v: [0, 0, 0] for v in (0, 1)}
for t in range(300):
    S = [1, 8, 3][t % 3]
    qn = (torch.randn(1, S, 16, 256, generator=g) * torch.exp(torch.randn(1, S, 16, 256, generator=g) * 2)).to(torch.bfloat16).to(dev)
    kn = (torch.randn(1, S, 4, 256, generator=g) * 30).to(torch.bfloat16).to(dev)
    p0 = int(torch.randint(0, 3000, (1,), generator=g))
    f = table[:, :, p0:p0 + S]
    rq, rk = q35.apply_partial_rope(qn.transpose(1, 2), kn.transpose(1, 2), f, 64)
    for v, m in mods.items():
        key = torch.zeros(1, 4, 64, 256, dtype=torch.bfloat16, device=dev); val = torch.zeros_like(key)
        q, k = m.rope_kv(qn[0], kn[0], kn[0], f.reshape(-1, 32, 2, 2) if S == 1 else f[0, 0], key, val, torch.arange(S, device=dev), 64)
        mq = (q.view(torch.int16) != rq.view(torch.int16)); bad[v][0] += mq.sum().item(); bad[v][1] += (k.view(torch.int16) != rk.view(torch.int16)).sum().item()
        if mq.any() and v == 0 and bad[v][2] < 3:
            bad[v][2] += 1; i = mq.nonzero()[0].tolist(); print("ROPEREAL S", S, "idx", i, "mine", q[tuple(i)].item(), "live", rq[tuple(i)].item())
            _, h, s_, d = i; j = d % 32; fr = f[0, 0, s_, j].double().cpu(); xx = qn[0, s_, h].double().cpu(); lo, hi = xx[j], xx[j + 32]
            R = lambda v: torch.tensor(v, dtype=torch.float64).float().double()
            print("ROPEREAL emu", (R(R(lo*fr[0,0]) + R(hi*fr[0,1])) if d < 32 else R(R(hi*fr[1,1]) + R(lo*fr[1,0]))).to(torch.bfloat16).item(), "fstride", f.stride(), f[0,0].stride(), f[0,0].is_contiguous())
print("ROPEREAL variant", os.environ.get("ROPEV", "0"), bad)
