# SPDX-License-Identifier: Apache-2.0
"""Generation-owned weight copies and caches, shared by resident and DynamicVRAM calls."""
from __future__ import annotations

from functools import wraps
import logging
import os
import threading

import torch

CLAIM = "__comfy_mono_decode_xpu_session__"
_lock = threading.Lock()
_tiled_ok = os.environ.get("MONO_DECODE_XPU_TILED", "1") == "1"
_margin = int(float(os.environ.get("MONO_DECODE_XPU_TILE_MARGIN_GB", "3")) * (1 << 30))
_session = {"active": False, "ok": False, "tiles": {}, "cache": {}, "plans": {}, "bytes": 0, "model": None,
            "budget": None}
_stats = {"tile_bytes": 0, "tile_oom": 0}
_last_budget = None
_CACHES = ("_mono_decode_xpu_w4", "_mono_decode_xpu_bf16", "_mono_decode_xpu_w1",
           "_mono_decode_xpu_scale", "_mono_decode_xpu_cb_ok", "_mono_decode_xpu_skip")


def stats():
    with _lock:
        return dict(_stats, live_bytes=_session["bytes"], active=_session["active"], budget=_last_budget)


def tile_major(t, block):
    """[N, B] bytes -> [N/16][B/block][16][block], kept as [N, B]."""
    n, b = t.shape
    return t.view(n // 16, 16, b // block, block).permute(0, 2, 1, 3).contiguous().view(n, b)


def _storage_bytes(tensors):
    storage = {}
    for t in tensors:
        if isinstance(t, torch.Tensor):
            s = t.untyped_storage()
            storage[(t.device, s.data_ptr())] = s.nbytes()
    return sum(storage.values())


def _copy_budget(device):
    """Choose after prefill, from actual packed weights and the live allocator pool."""
    import comfy.model_management as mm
    model = _session["model"]
    copies = []
    dynamic = False
    for module in model.modules():
        w = getattr(module, "weight", None)
        dynamic |= hasattr(module, "_v") and w is not None and w.device != device
        q, p = getattr(w, "_qdata", None), getattr(w, "_params", None)
        if (q is not None and getattr(p, "group_size", None) == 16
                and getattr(p, "convrot_groupsize", None) == 256):
            copies.extend((q, getattr(p, "scale", None)))
    model_tensors = []
    if dynamic:  # VBAR model storage is outside Torch's allocator pool.
        for parameter in model.parameters():
            q = getattr(parameter, "_qdata", None)
            if q is None:
                model_tensors.append(parameter)
            else:
                params = parameter._params
                model_tensors.extend((q, *getattr(params, "__dict__", {}).values(),
                                      *(getattr(params, k, None) for k in ("scale", "s_channel", "codebook"))))
    total = torch.xpu.get_device_properties(device).total_memory
    reserved = torch.xpu.memory_reserved(device)
    free = torch.xpu.mem_get_info(device)[0]
    reserve = mm.minimum_inference_memory()
    model_bytes, copy_bytes = _storage_bytes(model_tensors), _storage_bytes(copies)
    available = max(0, min(total - reserved - reserve - model_bytes, free - reserve - _margin))
    return dict(layout="tile" if copy_bytes <= available else "row", copy_bytes=copy_bytes,
                available_bytes=available, capacity_bytes=total, reserved_bytes=reserved,
                inference_reserve_bytes=reserve, model_residency_budget_bytes=model_bytes,
                dynamic_vram=dynamic)


def begin(model):
    _session.update(active=True, ok=_tiled_ok and os.environ.get("MONO_DECODE_XPU_SESSION_TILES", "1") == "1",
                    tiles={}, cache={}, plans={}, bytes=0, model=model, budget=None)


def clear_model_caches(model):
    modules = tuple(model.modules())
    # Graphs hold tensor references: release them before releasing copies.
    for module in modules:
        vars(module).pop("_mono_decode_xpu_graph", None)
    for module in modules:
        for name in _CACHES:
            vars(module).pop(name, None)


def end():
    model = _session["model"]
    if model is not None:
        clear_model_caches(model)
    # Dropping our references is sufficient; leave Torch's reusable pool intact.
    _session.update(active=False, ok=False, tiles={}, cache={}, plans={}, bytes=0, model=None, budget=None)


def tiles_enabled():
    return _session["active"] and _session["ok"]


def row_major(w):
    p = w._params
    return [w._qdata, p.scale, p.s_channel.float().contiguous(), p.codebook.float().contiguous()]


def tiled(lin, w):
    """Reuse one copy per canonical module weight, independent of VBAR cast addresses."""
    global _last_budget
    if not tiles_enabled():
        return None
    if _session["budget"] is None:
        try:
            budget = _copy_budget(w.device)
        except Exception as e:  # an unexpected model/allocator state: keep the row-major weights
            logging.warning("[mono_decode_xpu] weight copy budget unavailable (%s: %s); reading original "
                            "row-major weights for this generate", type(e).__name__, e)
            budget = dict(layout="row", copy_bytes=0, available_bytes=0, error=f"{type(e).__name__}: {e}")
        _session["budget"] = _last_budget = budget
        logging.info("[mono_decode_xpu] weight layout=%s; copies=%d, available=%d",
                     _last_budget["layout"], _last_budget["copy_bytes"], _last_budget["available_bytes"])
        if _last_budget["layout"] != "tile":
            _session["ok"] = False
            return None
    q, p = w._qdata, w._params
    source = lin.weight
    source_key = (source._qdata, source._params.scale)
    hit = _session["tiles"].get(lin)
    if hit is None or hit[0] is not source_key[0] or hit[1] is not source_key[1]:
        try:
            # Every part is owned by this generate: a DynamicVRAM cast's tensors are only valid within
            # its call, and FP32 metadata would otherwise be returned as-is by .float().
            parts = [tile_major(q, 16), tile_major(p.scale.view(torch.uint8), 16).view(torch.float8_e4m3fn),
                     p.s_channel.to(torch.float32, copy=True).contiguous(),
                     p.codebook.to(torch.float32, copy=True).contiguous()]
        except (torch.OutOfMemoryError, RuntimeError) as e:
            if not isinstance(e, torch.OutOfMemoryError) and "out of memory" not in str(e).lower():
                raise
            _session.update(ok=False, tiles={}, plans={}, bytes=0)
            with _lock:
                _stats["tile_oom"] += 1
            logging.warning("[mono_decode_xpu] tile allocation failed; reading original row-major weights")
            return None
        if hit is not None:
            _session["bytes"] -= hit[2]
        size = q.numel() * q.element_size() + p.scale.numel() * p.scale.element_size()
        hit = (source_key[0], source_key[1], size, parts)
        _session["tiles"][lin] = hit
        _session["bytes"] += size
        with _lock:
            _stats["tile_bytes"] += size
    return hit[3]


def plan(owner):
    """This generate's tiled kernel plan for `owner` (a decoder block), if one was recorded."""
    return _session["plans"].get(owner) if _session["active"] and _session["ok"] else None


def set_plan(owner, value):
    if _session["active"] and _session["ok"]:
        _session["plans"][owner] = value


def cached(key, build):
    if not _session["active"]:
        return build()
    if key not in _session["cache"]:
        _session["cache"][key] = build()
    return _session["cache"][key]


def install():
    from comfy.text_encoders import qwen35 as q35, llama
    done = 0
    for cls in (q35.Qwen35, llama.BaseGenerate):
        original = getattr(cls, "generate", None)
        if original is None or getattr(original, CLAIM, False):
            continue

        @wraps(original)
        def wrapped(self, *a, _original=original, **k):
            if _session["active"] or not isinstance(self, q35.Qwen35):
                return _original(self, *a, **k)
            clear_model_caches(self)
            begin(self)
            try:
                return _original(self, *a, **k)
            finally:
                end()

        setattr(wrapped, CLAIM, True)
        setattr(cls, "generate", wrapped)
        done += 1
    return done
