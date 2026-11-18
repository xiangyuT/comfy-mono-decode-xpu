# SPDX-License-Identifier: Apache-2.0
"""Generation-owned weight copies and caches, shared by resident and DynamicVRAM calls."""
from __future__ import annotations

from functools import wraps
from contextlib import contextmanager
import logging
import os
import threading

import torch

from . import guard

CLAIM = "__comfy_mono_decode_xpu_session__"
_lock = threading.Lock()
_tiled_ok = os.environ.get("MONO_DECODE_XPU_TILED", "1") == "1"
_margin = int(float(os.environ.get("MONO_DECODE_XPU_TILE_MARGIN_GB", "3")) * (1 << 30))
_session = {"active": False, "ok": False, "tiles": {}, "cache": {}, "plans": {}, "bytes": 0, "model": None,
            "budget": None}
_stats = {"tile_bytes": 0, "tile_oom": 0}
_last_budget = None
_last_generation = dict(tile_plans=0, prefetch_skips=0, host_rows_calls=0, head_cache=None)
_CACHES = ("_mono_decode_xpu_w4", "_mono_decode_xpu_bf16", "_mono_decode_xpu_w1",
           "_mono_decode_xpu_scale", "_mono_decode_xpu_cb_ok", "_mono_decode_xpu_skip")


def stats():
    with _lock:
        generation = _generation_stats() if _session["active"] else dict(_last_generation)
        head_cache = generation.get("head_cache")
        generation["head_cache"] = dict(head_cache) if head_cache is not None else None
        budget = _session["budget"] if _session["active"] else _last_budget
        return dict(_stats, **generation, live_bytes=_session["bytes"], active=_session["active"],
                    budget=dict(budget) if budget is not None else None)


def _generation_stats():
    head_cache = _session.get("head_cache")
    return dict(tile_plans=len(_session["plans"]), prefetch_skips=_session.get("prefetch_skips", 0),
                host_rows_calls=_session.get("host_rows_calls", 0),
                head_cache=dict(head_cache) if head_cache is not None else None)


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


def _inactive_vram_budget(device):
    """Count inactive VBAR pages only when native Torch allocations can reclaim them."""
    try:
        from comfy_aimdo import control
        mode, stats = control.get_xpu_allocator_mode, control.get_xpu_ur_hook_stats
    except (ImportError, AttributeError):
        return 0
    if mode() != "native_hook" or not stats().get("tracked_alloc_calls", 0):
        return 0
    import comfy.model_management as mm
    active = _session["model"]
    inactive, protected = {}, set()
    for loaded in mm.current_loaded_models:
        patcher = loaded.model
        if loaded.device != device or not patcher.is_dynamic():
            continue
        vbar = getattr(patcher.model, "dynamic_vbars", {}).get(device)
        if vbar is None:
            continue
        if loaded.currently_used or any(module is active for module in patcher.model.modules()):
            protected.add(id(vbar))
        else:
            inactive[id(vbar)] = vbar
    return sum(vbar.loaded_size() for key, vbar in inactive.items() if key not in protected)


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
    allocated = torch.xpu.memory_allocated(device)
    free = torch.xpu.mem_get_info(device)[0]
    reserve = mm.minimum_inference_memory()
    model_bytes, copy_bytes = _storage_bytes(model_tensors), _storage_bytes(copies)
    reclaimable = _inactive_vram_budget(device) if dynamic else 0
    available = max(0, min(total - allocated - reserve - model_bytes, free + reclaimable - reserve - _margin))
    return dict(layout="tile" if copy_bytes <= available else "row", copy_bytes=copy_bytes,
                available_bytes=available, capacity_bytes=total, reserved_bytes=reserved,
                allocated_bytes=allocated,
                inference_reserve_bytes=reserve, model_residency_budget_bytes=model_bytes,
                dynamic_vram=dynamic, inactive_vbar_bytes=reclaimable, physical_free_bytes=free)


@contextmanager
def _reuse_tile_plans(model):
    if not _weight_reuse_enabled():
        yield
        return
    body = model.model
    original = body.forward
    previous = vars(body).get("forward")
    prefetch = body.prefetch_dynamic_vbars

    @wraps(original)
    def forward(*args, **kwargs):
        embeds = kwargs.get("embeds")
        reuse = (prefetch and not guard.off("layer") and embeds is not None
                 and embeds.device.type == "xpu" and embeds.dtype == torch.bfloat16
                 and embeds.ndim == 3 and embeds.shape[0] == 1 and 1 <= embeds.shape[1] <= 8
                 and embeds.shape[2] == 4096 and kwargs.get("past_key_values") is not None
                 and len(body.layers) > 0 and all(plan(block) is not None for block in body.layers))
        body.prefetch_dynamic_vbars = False if reuse else prefetch
        if reuse:
            _session["prefetch_skips"] += 1
        return original(*args, **kwargs)

    body.forward = forward
    try:
        yield
    finally:
        body.prefetch_dynamic_vbars = prefetch
        if previous is None:
            vars(body).pop("forward", None)
        else:
            body.forward = previous


def _weight_reuse_enabled():
    return os.environ.get("MONO_DECODE_XPU_WEIGHT_REUSE", "1") == "1"


def host_rows_enabled(embedding, tokens):
    if not (_weight_reuse_enabled()
            and hasattr(embedding, "_v")
            and getattr(embedding, "quant_format", None) == "int8_tensorwise"
            and session_active() and tokens.device.type == "xpu" and 1 <= tokens.numel() <= 8
            and embedding.weight.device.type == "cpu"
            and not embedding.weight_function and not embedding.bias_function
            and embedding.weight_lowvram_function is None and embedding.max_norm is None):
        return False
    import comfy.ops
    return comfy.ops.vbar_above_watermark(embedding)


@contextmanager
def _host_embedding_rows(model):
    embedding = model.model.embed_tokens
    if (not _weight_reuse_enabled()
            or not hasattr(embedding, "_v")
            or getattr(embedding, "quant_format", None) != "int8_tensorwise"):
        yield
        return
    original = embedding.forward
    previous = vars(embedding).get("forward")

    @wraps(original)
    def forward(tokens, *args, **kwargs):
        if not args and not (kwargs.keys() - {"out_dtype"}) and host_rows_enabled(embedding, tokens):
            result = embedding.host_rows(tokens, out_dtype=kwargs.get("out_dtype"))
            _session["host_rows_calls"] += 1
            return result
        return original(tokens, *args, **kwargs)

    embedding.forward = forward
    try:
        yield
    finally:
        if previous is None:
            vars(embedding).pop("forward", None)
        else:
            embedding.forward = previous


@contextmanager
def _owned_output_head(model):
    import comfy.ops as ops
    body = model.model
    head = getattr(body, "lm_head", None)
    if (not _weight_reuse_enabled() or head is None
            or not hasattr(head, "_v") or getattr(head, "quant_format", None) != "int8_tensorwise"):
        yield
        return
    original = ops.cast_bias_weight
    source = head.weight
    owner_thread = threading.get_ident()
    owned = None
    attempted = False
    status = _session["head_cache"] = dict(bytes=0, hits=0, reason="waiting for decode")

    @wraps(original)
    def cast(*args, **kwargs):
        nonlocal owned, attempted
        module = args[0] if args else kwargs.get("s")
        input = args[1] if len(args) > 1 else kwargs.get("input")
        eligible = (module is head and threading.get_ident() == owner_thread and session_active()
                    and len(args) == 2 and kwargs == {"offloadable": True} and input is not None
                    and input.device.type == "xpu" and input.dtype == source.dtype == torch.bfloat16
                    and input.ndim == 3 and input.shape[0] == 1 and 1 <= input.shape[1] <= 8
                    and head.weight is source and source.device.type == "cpu"
                    and not head.weight_function and not head.bias_function and head.bias is None
                    and head.weight_lowvram_function is None
                    and not body.prefetch_dynamic_vbars and len(body.layers) > 0
                    and all(plan(block) is not None for block in body.layers))
        if not eligible:
            return original(*args, **kwargs)
        if owned is not None:
            status["hits"] += 1
            return owned, None, None
        if attempted:
            return original(*args, **kwargs)
        attempted = True
        qdata, scale = source._qdata, source._params.scale
        required = qdata.numel() * qdata.element_size() + scale.numel() * scale.element_size()
        if required > _copy_budget(input.device)["available_bytes"]:
            status["reason"] = "insufficient memory"
            return original(*args, **kwargs)
        result = original(*args, **kwargs)
        try:
            owned = result[0].clone()
        except torch.OutOfMemoryError:
            status["reason"] = "allocation failed"
            return result
        except BaseException:
            ops.uncast_bias_weight(module, *result)
            raise
        ops.uncast_bias_weight(module, *result)
        status.update(bytes=required, reason="ready")
        return owned, None, None

    ops.cast_bias_weight = cast
    try:
        yield
    finally:
        ops.cast_bias_weight = original
        owned = None


def begin(model):
    _session.update(active=True, ok=_tiled_ok and os.environ.get("MONO_DECODE_XPU_SESSION_TILES", "1") == "1",
                    tiles={}, cache={}, plans={}, bytes=0, model=model, budget=None, prefetch_skips=0,
                    host_rows_calls=0, head_cache=None)


def clear_model_caches(model):
    modules = tuple(model.modules())
    # Graphs hold tensor references: release them before releasing copies.
    for module in modules:
        vars(module).pop("_mono_decode_xpu_graph", None)
    for module in modules:
        for name in _CACHES:
            vars(module).pop(name, None)


def end():
    global _last_generation, _last_budget
    model = _session["model"]
    if model is not None:
        clear_model_caches(model)
    if _session["active"]:
        _last_generation = _generation_stats()
        _last_budget = _session["budget"]
    # Dropping our references is sufficient; leave Torch's reusable pool intact.
    _session.update(active=False, ok=False, tiles={}, cache={}, plans={}, bytes=0, model=None, budget=None)


def session_active():
    return _session["active"]


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
                with (_reuse_tile_plans(self), _host_embedding_rows(self), _owned_output_head(self)):
                    return _original(self, *a, **k)
            finally:
                end()

        setattr(wrapped, CLAIM, True)
        setattr(cls, "generate", wrapped)
        done += 1
    return done
