import sys, torch
sys.path.insert(0, "/work")
from comfy_mono_decode_xpu import _C as C
from safetensors import safe_open
import torch.nn.functional as F
f = safe_open("/models/extra/text_encoders/qwen3.5_9b_w4a8_gptq.safetensors", "pt")
dev = "xpu"
def parts(n):
    return [f.get_tensor(n + ".weight").to(dev), f.get_tensor(n + ".weight_s_rel").to(dev), f.get_tensor(n + ".weight_s_channel").float().to(dev), f.get_tensor(n + ".weight_codebook").float().to(dev)]
sets = []
for li in range(4):
    b = f"model.language_model.layers.{li}.mlp."
    sets.append(parts(b + "gate_proj") + parts(b + "up_proj") + parts(b + "down_proj"))
from comfy_kitchen.tensor.int8_utils import _build_hadamard
H = _build_hadamard(256, device=dev, dtype=torch.bfloat16)
LUT = F.silu(torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).to(dev)).contiguous()
xn = torch.randn(8, 4096, dtype=torch.bfloat16, device=dev)
for stop in [int(a) for a in sys.argv[1:]]:
    fn = lambda i: C.mono_mlp(xn, H, LUT, *sets[i % 4], 32, stop)
    for i in range(8): fn(i)
    torch.xpu.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.XPU]) as p:
        for i in range(40): fn(i)
        torch.xpu.synchronize()
    ks = [e for e in p.events() if e.device_type == torch.autograd.DeviceType.XPU]
    print(f"stop {stop}: kernel_us={sum(e.device_time_total for e in ks)/40:.1f}", flush=True)
for i in range(4): C.mono_mlp(xn, H, LUT, *sets[i % 4], 32, int(__import__("os").environ.get("PSTOP", "4")))
torch.xpu.synchronize()
d = C.mono_mlp_debug().view(64, 8)[:32, :7].cpu().double()
d = d - d[:, :1]
names = ["start", "p1done", "b1pass", "p2done", "b2pass", "b3pass", "p4done"]
for i, n in enumerate(names):
    col = d[:, i] / 2.8e3
    print(f"PHASE {n:7s} us min {col.min().item():7.1f} med {col.median().item():7.1f} max {col.max().item():7.1f}", flush=True)
