"""In-process: load the MOSS TE (resident, gpu-only) with OmniXPU, return the Qwen35 model."""
import asyncio, os, sys
def load():
    sys.argv = ["main.py", "--gpu-only", "--disable-dynamic-vram", "--extra-model-paths-config", "/artifacts/scripts/model-paths.yaml"]
    sys.path.insert(0, "/llm/ComfyUI"); os.chdir("/llm/ComfyUI")
    import comfy.options; comfy.options.enable_args_parsing()
    import importlib.util  # OmniXPU prestartup registers the Kitchen XPU backend before comfy imports
    pre = "/llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU/prestartup_script.py"
    spec = importlib.util.spec_from_file_location("omnixpu_prestartup", pre)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    import utils.extra_config
    utils.extra_config.load_extra_path_config("/artifacts/scripts/model-paths.yaml")
    import nodes
    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=True))
    clip = nodes.CLIPLoader().load_clip("qwen3.5_9b_w4a8_gptq.safetensors", type="moss_image", device="default")[0]
    import comfy.model_management as mm
    mm.load_models_gpu([clip.patcher])
    te = clip.cond_stage_model
    inner = getattr(te, te.clip) if hasattr(te, "clip") else te
    tm = inner.transformer
    return clip, tm
if __name__ == "__main__":
    import torch
    clip, tm = load()
    mlp = tm.model.layers[0].mlp
    w = mlp.gate_proj.weight
    print(type(tm).__name__, type(mlp).__name__, type(w).__name__, w.device)
    print([a for a in dir(w) if a.startswith("_") and not a.startswith("__")][:40])
    p = getattr(w, "_params", None); print(type(p).__name__, p)
    print("qdata", getattr(w, "_qdata", None).shape if hasattr(w, "_qdata") else None)
    print("norm", type(tm.model.layers[0].post_attention_layernorm).__name__, tm.model.layers[0].post_attention_layernorm.weight.dtype)
