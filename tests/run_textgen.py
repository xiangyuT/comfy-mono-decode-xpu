"""Start ComfyUI via the image entrypoint and submit TextGenerate-only graphs (MOSS official PE params).
usage: run_textgen.py TAG MTP words(comma) [--env K=V ...]"""
import json, os, subprocess, sys, time, urllib.request, signal
tag, mtp, words = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
extra_env = dict(x.split("=", 1) for x in sys.argv[4:])
W = json.load(open("/artifacts/scripts/moss-api.json"))
SYS = W["18:14"]["inputs"]["value"]; BASE = W["18:17"]["inputs"]["value"]
tg = dict(W["18:11"]["inputs"]); tg["mtp"] = mtp
run = f"/artifacts/runs/{tag}"; os.makedirs(run, exist_ok=True)
for d in ("out", "user", "in"): os.makedirs(f"{run}/{d}", exist_ok=True)
env = os.environ.copy(); env.update(MONO_TIMER_OUT=f"{run}/timer.jsonl", MONO_TIMER_TAG=tag, **extra_env)
port = 8288
log = open(f"{run}/server.log", "wb")
srv = subprocess.Popen(["/llm/entrypoints/start_comfyui.sh", "--port", str(port), "--listen", "127.0.0.1",
    "--extra-model-paths-config", "/artifacts/scripts/model-paths.yaml", "--output-directory", f"{run}/out",
    "--user-directory", f"{run}/user", "--input-directory", f"{run}/in", "--disable-auto-launch"] + os.environ.get("CMDX_SERVER_ARGS", extra_env.get("CMDX_SERVER_ARGS", "")).split(),
    cwd="/llm/ComfyUI", env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
def api(path, data=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=None if data is None else json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())
for _ in range(600):
    try: api("/system_stats"); break
    except Exception: time.sleep(1)
    if srv.poll() is not None: sys.exit("server died")
results = []
try:
    for w in words:
        prompt = BASE if w == "base" else BASE.replace("a model", "a " + w)
        g = {"1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3.5_9b_w4a8_gptq.safetensors", "type": "moss_image", "device": "default"}},
             "2": {"class_type": "TextGenerate", "inputs": {**tg, "clip": ["1", 0], "prompt": prompt, "system_prompt": SYS}},
             "3": {"class_type": "PreviewAny", "inputs": {"source": ["2", 0]}}}
        t = time.perf_counter(); pid = api("/prompt", {"prompt": g})["prompt_id"]
        while True:
            h = api(f"/history/{pid}")
            if pid in h and h[pid].get("status", {}).get("completed") is not None or (pid in h and h[pid]["status"].get("status_str") in ("success", "error")): break
            time.sleep(0.2)
        wall = time.perf_counter() - t
        st = h[pid]["status"]; text = json.dumps(h[pid].get("outputs", {}))
        r = {"word": w, "wall_s": wall, "status": st.get("status_str"), "text_chars": len(text)}
        print(json.dumps(r), flush=True); results.append(r)
        with open(f"{run}/history-{w}.json", "w") as f: json.dump(h[pid], f)
finally:
    os.killpg(srv.pid, signal.SIGTERM); srv.wait(timeout=60)
timers = [json.loads(l) for l in open(f"{run}/timer.jsonl")] if os.path.exists(f"{run}/timer.jsonl") else []
for r, tm in zip(results, timers): r.update(tm)
json.dump(results, open(f"{run}/summary.json", "w"), indent=1)
for r in results: print(r["word"], r.get("status"), "tok", r.get("tokens"), "gen %.3f pre %.3f dec %.3f tps %.2f dev %s sha %s" % (r.get("generate_s",0), r.get("prefill_s",0), r.get("decode_s",0), (r.get("tokens",1)-1)/max(r.get("decode_s",1e-9),1e-9), r.get("param_device"), r.get("ids_sha")))
