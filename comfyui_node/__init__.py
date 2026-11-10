import os, sys, logging
NODE_CLASS_MAPPINGS = {}
if any(os.environ.get(k) == "1" for k in ("CMDX_ENABLE", "CMDX_GRAPH", "CMDX_PREP", "CMDX_MONO", "CMDX_LAYER")):
    sys.path.insert(0, os.environ.get("CMDX_REPO", os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
if os.environ.get("CMDX_ENABLE") == "1":
    from comfy_mono_decode_xpu import patch
    logging.info("[cmdx] w4a8 installed=%s", patch.install())
if os.environ.get("CMDX_PREP") == "1":
    import comfy_kitchen  # noqa: F401  (loads the XPU backend modules)
    from comfy_mono_decode_xpu import patch
    logging.info("[cmdx] prep installed=%s", patch.install_prep())
if os.environ.get("CMDX_MONO") == "1":
    from comfy_mono_decode_xpu import mono
    logging.info("[cmdx] mono_mlp installed=%s", mono.install())
if os.environ.get("CMDX_LAYER") == "1":
    from comfy_mono_decode_xpu import layer
    logging.info("[cmdx] layer installed=%s", layer.install())
if os.environ.get("CMDX_GRAPH") == "1":
    from comfy_mono_decode_xpu import graph_step
    logging.info("[cmdx] graph_step installed=%s", graph_step.install())
