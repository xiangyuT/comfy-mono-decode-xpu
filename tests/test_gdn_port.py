"""Run a real in-process generate; for decode GDN calls compare the SIMD16 ports against Kitchen on cloned state."""
import sys, os, json
ARGV = list(sys.argv)
sys.path.insert(0, "/artifacts/scripts"); sys.path.insert(0, "/work")
from load_te import load
clip, tm = load()
import torch, comfy_kitchen
from comfy_mono_decode_xpu import _C as C, patch, mono, layer
patch.install(); import comfy_kitchen as ck; patch.install_prep(); mono.install(); layer.install()
N = int(ARGV[1]) if len(ARGV) > 1 else 400
AT = os.environ.get("AT_LINEAR") == "1"
res = {"conv_calls": 0, "delta_calls": 0}
def acc(k, v): res[k] = res.get(k, 0) + int(v)
def bad(a, b):
    if a.dtype in (torch.bfloat16, torch.float16): return (a.view(torch.int16) != b.view(torch.int16)).sum()
    return (a.view(torch.int32) != b.view(torch.int32)).sum()
oc, od = ck.deltanet_conv_step_deferred, ck.gated_delta_decode_deferred
def conv(proj, state, w, b, pbuf, qbuf, ctl):
    if res["conv_calls"] < N:
        s2, p2, q2 = state.clone(), pbuf.clone(), qbuf.clone()
        C.gdn_conv_port(proj, s2, w, b, p2, q2, ctl)
        oc(proj, state, w, b, pbuf, qbuf, ctl)
        acc("conv_state", bad(state, s2)); acc("conv_pbuf", bad(pbuf, p2)); acc("conv_qkv", bad(qbuf, q2)); res["conv_calls"] += 1
        return None
    return oc(proj, state, w, b, pbuf, qbuf, ctl)
def delta(x, wa, wb, dt, gd, st, kd, hk, sc, z, nw, eps, qkv, gates, sums, ctl):
    if res["delta_calls"] < N:
        st2, g2, s2 = st.clone(), gates.clone(), sums.clone()
        o2 = C.gdn_delta_port(x, wa, wb, dt, gd, st2, kd, hk, sc, z, nw, eps, qkv, g2, s2, ctl, AT)
        o = od(x, wa, wb, dt, gd, st, kd, hk, sc, z, nw, eps, qkv, gates, sums, ctl)
        acc("d_out", bad(o, o2)); acc("d_state", bad(st, st2)); acc("d_gates", bad(gates, g2)); acc("d_sums", bad(sums, s2)); res["delta_calls"] += 1
        return o
    return od(x, wa, wb, dt, gd, st, kd, hk, sc, z, nw, eps, qkv, gates, sums, ctl)
ck.deltanet_conv_step_deferred = conv; ck.gated_delta_decode_deferred = delta
W = json.load(open("/artifacts/scripts/moss-api.json"))
SYS = W["18:14"]["inputs"]["value"]; BASE = W["18:17"]["inputs"]["value"]
tokens = clip.tokenize(BASE.replace("a model", "a giant"), skip_template=False, min_length=1, thinking=False, system_prompt=SYS)
ids = clip.generate(tokens, do_sample=True, max_length=96, temperature=0.7, top_k=64, top_p=0.95, min_p=0.05,
                    repetition_penalty=1.0, seed=0, presence_penalty=0.0, mtp="tree4s1m8")
print("RESULT", json.dumps(res), "tokens", len(ids))
