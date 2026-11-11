# SPDX-License-Identifier: Apache-2.0
"""Dump real apply_partial_rope / KV write inputs+outputs (verify S=8 and draft S=1) for the fused rope_kv kernel."""
import sys, os, json
sys.path.insert(0, "/artifacts/scripts"); sys.path.insert(0, "/work")
from load_te import load
clip, tm = load()
import torch
from comfy_mono_decode_xpu import patch, mono, layer
patch.install(); import comfy_kitchen; patch.install_prep(); mono.install(); layer.install()
from comfy.text_encoders import qwen35 as q35, llama
dumps = []
orig = q35.apply_partial_rope
def rope(xq, xk, f, rd):
    oq, ok = orig(xq, xk, f, rd)
    if len(dumps) < 400 and xq.shape[2] in (1, 8):
        dumps.append(dict(xq=xq.clone(), xk=xk.clone(), f=f.clone(), oq=oq.clone(), ok=ok.clone(), rd=rd,
                          xq_stride=xq.stride(), xk_stride=xk.stride(), f_shape=tuple(f.shape), f_stride=f.stride(), f_dtype=str(f.dtype)))
    return oq, ok
q35.apply_partial_rope = rope
W = json.load(open("/artifacts/scripts/moss-api.json"))
SYS = W["18:14"]["inputs"]["value"]; BASE = W["18:17"]["inputs"]["value"]
tokens = clip.tokenize(BASE.replace("a model", "a giant"), skip_template=False, min_length=1, thinking=False, system_prompt=SYS)
clip.generate(tokens, do_sample=False, max_length=60, temperature=0, seed=0, mtp="tree4s1m8")
d = dumps[0]
for k in ("xq_stride", "xk_stride", "f_shape", "f_stride", "f_dtype", "rd"):
    print("PROBE", k, d[k])
print("PROBE shapes", [(x["xq"].shape[2]) for x in dumps[:20]], len(dumps))
print("PROBE oq", dumps[0]["oq"].shape, dumps[0]["oq"].stride(), "ok", dumps[0]["ok"].stride())
import comfy_kitchen.registry as R
try: print("PROBE backend", R.registry.get_implementation("apply_rope_split_half", kwargs=dict(xq=d["xq"][..., :d["rd"]], xk=d["xk"][..., :d["rd"]], freqs_cis=d["f"])))
except Exception as e: print("PROBE backend?", e)
torch.save([{k: (v.cpu() if torch.is_tensor(v) else v) for k, v in x.items()} for x in dumps], "/dev/shm/rope_dumps.pt")
