# SPDX-License-Identifier: Apache-2.0
import sys, torch
sys.path.insert(0, "/work")
from comfy_mono_decode_xpu import _C as C
torch.manual_seed(0)
V = 248320; bad = 0; n = 0
for trial in range(300):
    x = (torch.randn(1, V, device="xpu") * 3).to(torch.bfloat16)
    if trial % 2: x = torch.round(x.float() * 2).to(torch.bfloat16) / 2
    if trial % 3 == 0:
        m = x.max(); idx = torch.randint(0, V, (6,), device="xpu"); x[0, idx] = m
    if trial % 7 == 0: x[0, torch.randint(0, V, (3,), device="xpu")] = 0.0; x[0, 5] = -0.0
    if trial % 11 == 0: x = torch.zeros(1, V, device="xpu", dtype=torch.bfloat16)  # all ties incl. +-0 mix
    for k in (1, 2, 3, 7, 8):
        r = x.topk(k, dim=-1); v, i = C.fast_topk(x, k, False)
        bad += int(not torch.equal(r.indices, i)) + int(not torch.equal(r.values.view(torch.int16), v.view(torch.int16))); n += 2
    a = x.argmax(dim=-1, keepdim=True); _, ai = C.fast_topk(x, 1, True)
    bad += int(not torch.equal(a.view(-1), ai.view(-1))); n += 1
    v8, i8, am = C.fast_pick(x, 8); r8 = x.topk(8, dim=-1)
    bad += int(not torch.equal(r8.indices, i8)) + int(not torch.equal(r8.values.view(torch.int16), v8.view(torch.int16)))
    bad += int(not torch.equal(a.view(-1), am.view(-1))); n += 3
print("TOPK mismatches", bad, "of", n)
x = torch.randn(1, V, device="xpu").to(torch.bfloat16)
for name, fn in (("torch topk7", lambda: x.topk(7, dim=-1)), ("fast topk7", lambda: C.fast_topk(x, 7, False)),
                 ("torch argmax", lambda: x.argmax(dim=-1)), ("fast argmax", lambda: C.fast_topk(x, 1, True)), ("fast pick8+argmax", lambda: C.fast_pick(x, 8))):
    for _ in range(5): fn()
    torch.xpu.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.XPU]) as p:
        for _ in range(20): fn()
        torch.xpu.synchronize()
    ks = [e for e in p.events() if e.device_type == torch.autograd.DeviceType.XPU]
    print(f"TIME {name}: {sum(e.device_time_total for e in ks)/20:.1f} us/call, {len(ks)/20:.0f} kernels")
