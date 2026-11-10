import sys, time, json
sys.path.insert(0, "/artifacts/scripts"); sys.path.insert(0, "/work")
from load_te import load
clip, tm = load()
import torch, torch.nn.functional as F
from comfy_mono_decode_xpu import _C as C
from comfy_kitchen.tensor.int8_utils import _build_hadamard
dev = torch.device("xpu")
H = _build_hadamard(256, device=dev, dtype=torch.bfloat16)
allbf = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).to(dev)
LUT = F.silu(allbf).contiguous()
def wparts(lin):
    w = lin.weight; p = w._params
    return [w._qdata.contiguous(), p.scale.contiguous(), p.s_channel.float().contiguous(), p.codebook.float().contiguous()]
def eager(mlp, xn):
    return mlp.down_proj(F.silu(mlp.gate_proj(xn)) * mlp.up_proj(xn))
def chain(xn2, a):
    q1, s1 = C.rot_quant(xn2, H, 3)
    gq, gs, gsc, gcb, uq, us, usc, ucb, dq, ds, dsc, dcb = a
    gt = C.w4a8_decode_linear(q1, s1.view(-1), gq, gs, gsc, gcb, None, 16, 2, 0, False)
    up = C.w4a8_decode_linear(q1, s1.view(-1), uq, us, usc, ucb, None, 16, 2, 0, False)
    h = (LUT[gt.view(torch.int16).long() & 0xffff].float() * up.float()).to(torch.bfloat16)
    hh = (F.silu(gt) * up)
    q2, s2 = C.rot_quant(h.contiguous(), H, 3)
    return C.w4a8_decode_linear(q2, s2.view(-1), dq, ds, dsc, dcb, None, 16, 2, 0, False), torch.equal(h, hh)
res = {}
g = torch.Generator(device="cpu").manual_seed(3)
bad_total = 0
for li in (0, 3, 17, 31):
    mlp = tm.model.layers[li].mlp
    a = wparts(mlp.gate_proj) + wparts(mlp.up_proj) + wparts(mlp.down_proj)
    for R in (1, 3, 8):
        for kind in ("normal", "outlier", "wide"):
            x = torch.randn(R, 4096, generator=g)
            if kind == "outlier": x[:, torch.randint(0, 4096, (8,), generator=g)] *= 200
            if kind == "wide": x = x * torch.exp(torch.randn(R, 4096, generator=g) * 2)
            xn = x.to(torch.bfloat16).to(dev)
            ch, _ = chain(xn, a)
            mo = C.mono_mlp(xn, H, LUT, *a, 32)
            b = (mo.view(torch.int16) != ch.view(torch.int16)).sum().item(); bad_total += b
    print(f"MONO layer {li}: mismatches vs chain so far {bad_total}", flush=True)
mlp = tm.model.layers[0].mlp; a = wparts(mlp.gate_proj) + wparts(mlp.up_proj) + wparts(mlp.down_proj)
for R in (1, 8):
    xn = torch.randn(R, 4096, dtype=torch.bfloat16, device=dev)
    for name, fn in [("chain", lambda: chain(xn, a))] + [(f"mono-stop{st}", (lambda st: lambda: C.mono_mlp(xn, H, LUT, *a, 32, st))(st)) for st in (1, 2, 3, 4)]:
        for _ in range(10): fn()
        torch.xpu.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.XPU]) as p:
            for _ in range(20): fn()
            torch.xpu.synchronize()
        ks = [e for e in p.events() if e.device_type == torch.autograd.DeviceType.XPU]
        tot = sum(e.device_time_total for e in ks) / 20
        print(f"TIME R{R} {name}: kernels/call={len(ks)/20:.0f} kernel_us={tot:.1f}", flush=True)
xn = torch.randn(8, 4096, dtype=torch.bfloat16, device=dev)
for _ in range(3): C.mono_mlp(xn, H, LUT, *a, 32, 4)
torch.xpu.synchronize()
d = C.mono_mlp_debug().view(64, 8)[:32, :7].cpu().double()
d = d - d[:, :1]
names = ["start", "p1done", "b1pass", "p2done", "b2pass", "b3pass", "p4done"]
print("PHASE (clock ticks, min/median/max over WGs):", flush=True)
for i, n in enumerate(names):
    col = d[:, i]; print(f"PHASE {n:7s} min {col.min().item():10.0f} med {col.median().item():10.0f} max {col.max().item():10.0f}", flush=True)
sys.exit(0)
