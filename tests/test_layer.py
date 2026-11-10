"""layer_pre / layer_post vs the eager modules (real W4A8 kitchen path, no cmdx patches)."""
import sys, time
sys.path.insert(0, "/artifacts/scripts"); sys.path.insert(0, "/work")
from load_te import load
clip, tm = load()
import torch, torch.nn.functional as F, comfy.ops
from comfy_mono_decode_xpu import _C as C
from comfy_kitchen.tensor.int8_utils import _build_hadamard
dev = torch.device("xpu")
H = _build_hadamard(256, device=dev, dtype=torch.bfloat16).contiguous()
allbf = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).to(dev)
SILU = F.silu(allbf).contiguous(); SIG = torch.sigmoid(allbf).contiguous()
from comfy_mono_decode_xpu import mono as _mono
def wp(lin):
    return _mono._w4(lin)
g = torch.Generator(device="cpu").manual_seed(7)
def xs(R, kind, scale=1.0):
    x = torch.randn(R, 4096, generator=g) * scale
    if kind == "outlier": x[:, torch.randint(0, 4096, (8,), generator=g)] *= 100
    if kind == "wide": x = x * torch.exp(torch.randn(R, 4096, generator=g))
    return x.to(torch.bfloat16).to(dev)
def bad(a, b): return int((a.contiguous().view(torch.int16) != b.contiguous().view(torch.int16)).sum())
tot = {}
with comfy.ops.use_quantized_matmul(clip.cond_stage_model, dev), torch.no_grad():
    for li in (0, 3, 6, 31):
        L = tm.model.layers[li]
        gdn = L.layer_type == "linear_attention"
        mix = L.linear_attn if gdn else L.self_attn
        pre_lins = [mix.in_proj_qkv, mix.in_proj_z] if gdn else [mix.q_proj, mix.k_proj, mix.v_proj]
        olin = mix.out_proj if gdn else mix.o_proj
        nw_in = L.input_layernorm.scale().contiguous(); nw_post = L.post_attention_layernorm.scale().contiguous()
        mlp = L.mlp
        mm = wp(mlp.gate_proj) + wp(mlp.up_proj) + wp(mlp.down_proj)
        for R in (1, 3, 8):
            for kind in ("normal", "outlier", "wide"):
                x = xs(R, kind, 0.5)
                # pre
                xn_ref = L.input_layernorm(x)
                refs = [lin(xn_ref.view(1, R, 4096)).view(R, -1) for lin in pre_lins]
                mats = sum([wp(l) for l in pre_lins], [])
                outs = C.layer_pre(x, nw_in, L.input_layernorm.eps, H, mats, 0)
                tot["pre.xn"] = tot.get("pre.xn", 0) + bad(outs[0], xn_ref)
                for i, r in enumerate(refs): tot[f"pre.{i}"] = tot.get(f"pre.{i}", 0) + bad(outs[1 + i], r)
                # post
                y = xs(R, "normal", 0.3)
                gate = xs(R, "normal", 2.0) if not gdn else None
                yy = y * gate.sigmoid() if gate is not None else y
                x1 = x + olin(yy.view(1, R, 4096)).view(R, 4096)
                x2 = x1 + mlp(x1.view(1, R, 4096), norm=L.post_attention_layernorm).view(R, 4096)
                xo = x.clone()
                C.layer_post(xo, y.contiguous(), gate, SIG, *wp(olin), nw_post, L.post_attention_layernorm.eps, H, SILU, *mm, 0, 6)
                tot["post"] = tot.get("post", 0) + bad(xo, x2)
        print("layer", li, "gdn" if gdn else "attn", tot, flush=True)
    # timing (layer 0 GDN, layer 3 attn), R=8
    for li in (0, 3):
        L = tm.model.layers[li]; gdn = L.layer_type == "linear_attention"; mix = L.linear_attn if gdn else L.self_attn
        pre_lins = [mix.in_proj_qkv, mix.in_proj_z] if gdn else [mix.q_proj, mix.k_proj, mix.v_proj]
        olin = mix.out_proj if gdn else mix.o_proj
        mats = sum([wp(l) for l in pre_lins], []); mm = wp(L.mlp.gate_proj) + wp(L.mlp.up_proj) + wp(L.mlp.down_proj)
        x = xs(8, "normal", 0.5); y = xs(8, "normal", 0.3)
        nw_in = L.input_layernorm.scale().contiguous(); nw_post = L.post_attention_layernorm.scale().contiguous()
        fns = {"pre": lambda: C.layer_pre(x, nw_in, 1e-6, H, mats, 0),
               "post": lambda: C.layer_post(x.clone(), y, None, SIG, *wp(olin), nw_post, 1e-6, H, SILU, *mm, 0, 6)}
        for n, fn in fns.items():
            for _ in range(5): fn()
            torch.xpu.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.XPU]) as p:
                for _ in range(20): fn()
                torch.xpu.synchronize()
            ks = [e for e in p.events() if e.device_type == torch.autograd.DeviceType.XPU and "Layer" in e.name]
            print(f"TIME layer{li} {n}: kernel_us={sum(e.device_time_total for e in ks)/20:.1f}", flush=True)
