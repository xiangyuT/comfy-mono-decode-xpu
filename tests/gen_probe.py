"""In-process TextGenerate-equivalent generate (MOSS official params, fixed tree) + device timeline.
usage: gen_probe.py MAXLEN [MTP] ; env MONO_DECODE_XPU_W4/PREP/MONO as in the plugin."""
import sys, os, json, time, hashlib, collections
sys.path.insert(0, "/artifacts/scripts"); sys.path.insert(0, "/work")
ARGV = list(sys.argv)
from load_te import load
clip, tm = load()
import torch, comfy.model_management as mm
if os.environ.get("MONO_DECODE_XPU_W4") == "1":
    from comfy_mono_decode_xpu import patch; patch.install()
if os.environ.get("MONO_DECODE_XPU_PREP") == "1":
    import comfy_kitchen; from comfy_mono_decode_xpu import patch; patch.install_prep()
if os.environ.get("MONO_DECODE_XPU_MONO") == "1":
    from comfy_mono_decode_xpu import mono; mono.install()
if os.environ.get("MONO_DECODE_XPU_LAYER") == "1":
    from comfy_mono_decode_xpu import layer; layer.install()
W = json.load(open("/artifacts/scripts/moss-api.json"))
SYS = W["18:14"]["inputs"]["value"]; BASE = W["18:17"]["inputs"]["value"]
maxlen = int(ARGV[1]); mtp = ARGV[2] if len(ARGV) > 2 else "tree4s1m8"
word = os.environ.get("WORD", "giant")
prompt = BASE if word == "base" else BASE.replace("a model", "a " + word)
tokens = clip.tokenize(prompt, skip_template=False, min_length=1, thinking=False, system_prompt=SYS)
def gen(n):
    return clip.generate(tokens, do_sample=True, max_length=n, temperature=0.7, top_k=64, top_p=0.95, min_p=0.05,
                         repetition_penalty=1.0, seed=0, presence_penalty=0.0, mtp=mtp)
ids = gen(maxlen); torch.xpu.synchronize()
t0 = time.perf_counter(); ids = gen(maxlen); torch.xpu.synchronize(); dt = time.perf_counter() - t0
print("GEN tokens", len(ids), "s %.3f" % dt, "sha", hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()[:16],
      "prefix64", hashlib.sha256(json.dumps(list(ids[:64])).encode()).hexdigest()[:16], flush=True)
if os.environ.get("PROFILE") != "1":
    sys.exit(0)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.XPU]) as pr:
    ids = gen(maxlen); torch.xpu.synchronize()
ev = sorted([e for e in pr.events() if e.device_type == torch.autograd.DeviceType.XPU], key=lambda e: e.time_range.start)
def cat(n):
    for k, c in (("LayerPre", "layer_pre"), ("LayerPost", "layer_post"), ("MonoMlp", "mono_mlp"), ("W4A8", "w4"), ("RotQuant", "rotq"), ("int8_linear", "lm_head"), ("gemm_kernel", "gemm"),
                 ("DecodeChunks", "attn"), ("DecodeCombine", "attn"), ("DecodeStepMerge", "attn"), ("Deferred", "gdn"), ("rms_norm", "norm"),
                 ("topk", "topk"), ("Sbtopk", "topk"), ("Memcpy", "memcpy")):
        if k in n: return c
    return "other"
# decode portion: drop events before the largest gap after prefill? use the last 70% of the span
monos = [e for e in ev if "MonoMlp" in e.name or "LayerPost" in e.name]
ncyc = len(monos) // 32 - 1
w0, w1 = monos[31].time_range.end, monos[-1].time_range.end  # first verify done .. last verify done
ev = [e for e in ev if w0 < e.time_range.start and e.time_range.end <= w1]
print("cycles", ncyc, "window_us %.0f per_cycle_us %.0f" % (w1 - w0, (w1 - w0) / ncyc))
start, end = w0, w1
busy = collections.Counter(); cnt = collections.Counter()
for e in ev:
    c = cat(e.name); busy[c] += e.time_range.elapsed_us(); cnt[c] += 1
# gaps
gap = 0.0; last = start
for e in ev:
    gap += max(0.0, e.time_range.start - last); last = max(last, e.time_range.end)
span = end - start
print("SPAN_us %.0f busy_us %.0f gap_us %.0f kernels %d" % (span, sum(busy.values()), gap, len(ev)))
for c, v in busy.most_common():
    print(f"  {c:10s} per_cycle_us {v/ncyc:9.1f} n/cycle {cnt[c]/ncyc:7.1f} avg {v/cnt[c]:8.1f}")
print("  gap per_cycle_us %.1f" % (gap / ncyc))
oth = collections.Counter()
for e in ev:
    if cat(e.name) == "other": oth[e.name[:90]] += e.time_range.elapsed_us()
for n, v in oth.most_common(25): print("   other %.1f" % (v / ncyc), n)
