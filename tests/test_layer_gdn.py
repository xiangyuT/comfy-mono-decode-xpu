"""Real in-process generate; for decode GDN blocks compare layer_gdn (on cloned state) against the eager block."""
import sys, os, json
ARGV = list(sys.argv)
sys.path.insert(0, "/artifacts/scripts"); sys.path.insert(0, "/work")
from load_te import load
clip, tm = load()
import torch
from comfy_mono_decode_xpu import _C as _C0, patch, mono, layer
class _P: pass
C = _P(); [setattr(C, n, getattr(_C0, n)) for n in dir(_C0) if not n.startswith("__")]
patch.install(); import comfy_kitchen; patch.install_prep(); mono.install()
from comfy.text_encoders import qwen35 as q35
if os.environ.get("GDNMOD"):
    sys.path.insert(0, "/artifacts/exp/gdnmod"); import mono_decode_xpu_gdn
    C.layer_gdn = mono_decode_xpu_gdn.layer_gdn; C.layer_gdn_scratch = mono_decode_xpu_gdn.layer_gdn_scratch
from comfy_kitchen.tensor import QuantizedTensor
N = int(ARGV[1]) if len(ARGV) > 1 else 200
res = {"calls": 0}
def acc(k, v): res[k] = res.get(k, 0) + int(v)
def bad(a, b):
    if a.dtype in (torch.bfloat16, torch.float16): return (a.view(torch.int16) != b.view(torch.int16)).sum()
    return (a.contiguous().view(torch.int32) != b.contiguous().view(torch.int32)).sum()
h, silu, sig = layer._constants(torch.device("xpu"))
orig = q35.Qwen35TransformerBlock.forward
def hook(self, x, attention_mask=None, freqs_cis=None, optimized_attention=None, past_key_value=None):
    kv = past_key_value
    S = x.shape[1] if x.dim() == 3 else 0
    if (res["calls"] < N and self.layer_type == "linear_attention" and kv is not None and x.shape[0] == 1 and 1 <= S <= 8
            and kv.decode_step(S)):
        import contextlib
        sp = layer._static_plan(self)
        es = contextlib.ExitStack()
        plan = layer._args(self, sp, x, es) if sp is not None else None
        if plan is not None:
            mix = self.linear_attn
            cl = {k: getattr(kv, k).clone() for k in ("key", "value", "proj_buf", "qkv_buf", "gates_buf", "sumsq_buf")}
            x_in = x.clone()
            out, _ = orig(self, x, attention_mask, freqs_cis, optimized_attention, past_key_value)
            wa, wb = plan["extra"]["wa"], plan["extra"]["wb"]
            w4 = plan["pre"] + plan["out"] + plan["mlp"]
            x2 = x_in.view(S, 4096).contiguous()
            nin, ein = plan["norm_in"]; npo, epo = plan["norm_post"]
            C.layer_gdn(x2, nin, ein, h, w4, wa, wb, plan["extra"]["conv_w"], cl["key"], cl["proj_buf"], cl["qkv_buf"], kv.ctl,
                        kv.dt_bias, kv.g_decay, cl["value"], cl["gates_buf"], cl["sumsq_buf"], kv.norm_weight, mix.norm.eps,
                        mix.key_head_dim ** -0.5, mix.num_key_heads, npo, epo, silu, 0, int(os.environ.get("STOP", "11")), plan["tiled"])
            acc("x", bad(x2, out.view(S, 4096)))
            for k in cl: acc(k, bad(cl[k], getattr(kv, k)))
            res["calls"] += 1
            es.close()
            return out, past_key_value
    return orig(self, x, attention_mask, freqs_cis, optimized_attention, past_key_value)
q35.Qwen35TransformerBlock.forward = hook
W = json.load(open("/artifacts/scripts/moss-api.json"))
SYS = W["18:14"]["inputs"]["value"]; BASE = W["18:17"]["inputs"]["value"]
tokens = clip.tokenize(BASE.replace("a model", "a giant"), skip_template=False, min_length=1, thinking=False, system_prompt=SYS)
ids = clip.generate(tokens, do_sample=True, max_length=64, temperature=0.7, top_k=64, top_p=0.95, min_p=0.05,
                    repetition_penalty=1.0, seed=0, presence_penalty=0.0, mtp="tree4s1m8")
print("RESULT", json.dumps(res), "tokens", len(ids))
# phase timing: rerun the last captured inputs a few times (state mutates, fine for timing)
d = C.layer_gdn_scratch(torch.empty(1, device="xpu"))[-1].view(64, 16)[:32, :12].cpu().double()
d = d - d[:, :1]
names = ["start", "P1 norm+rq", "P2 proj+z", "P3 conv+ab", None, "P5 delta", "P6 rq", "P7 out", "P8 norm+rq", "P9 gate/up", "P10 rq", "P11 down"]
prev = None
for i, n in enumerate(names):
    if n is None: continue
    col = d[:, i] / 2.8e3
    print(f"PHASE {n:12s} end us med {col.median().item():7.1f} max {col.max().item():7.1f}" + (f"  (+{col.median().item() - prev:.1f})" if prev is not None else ""), flush=True)
    prev = col.median().item()
