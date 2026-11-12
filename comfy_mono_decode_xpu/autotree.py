# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 auto MTP on XPU: feed best_widths the device time of the draft and verify parts.

ComfyUI's adaptive MTP tree (qwen35.best_widths) scores each width chain by accepted tokens per
millisecond. On CUDA it times a step with device events; on other devices it falls back to host
clocks, so on XPU the draft time is only the host's launch time and the verify time picks up the
drafts' unfinished device work through the sampling sync. This keeps upstream's shapes, scoring
and smoothing, and only replaces its two inputs with XPU event timings: per step, the device time
from the first draft to the verify forward (divided by the drafts it ran) and from the verify
forward to the next step's first draft (verify, lm_head, sampling and accept). Events are read
when best_widths runs, right after the step's own host sync, so reading them never waits.
Fixed MTP modes never call best_widths and are unaffected.

With the tree timed on the device, the sampling filters no longer need to stop the host: under
_MaskedAssign, processed_probs' `t[mask] = scalar` runs as `t.masked_fill_(mask, scalar)` (the same
values without the nonzero() that waits for the device), so the verify step's sampling queues
behind its lm_head. Installed together with the timing, never alone: upstream's host clocks rely
on that wait.
"""
from __future__ import annotations

import threading

import torch

from . import guard

CLAIM = "__comfy_mono_decode_xpu_autotree__"
_lock = threading.Lock()
_stats = {"steps": 0, "updates": 0, "draft_ms": None, "verify_ms": None}
_current = {"state": None}


def stats():
    with _lock:
        return dict(_stats)


class _Clock:
    def __init__(self):
        self.marks = []      # (kind, event, drafts so far in the step) in issue order
        self.drafts = 0      # drafts issued in the current step
        self.at_step_start = True
        self.steps = 0       # completed steps folded into the averages (upstream: skip the first 8)
        self.draft_ms = 0.0
        self.verify_ms = 0.0

    def mark(self, kind):
        e = torch.xpu.Event(enable_timing=True)
        e.record()
        self.marks.append((kind, e, self.drafts))

    def on_draft(self):
        if self.at_step_start:
            self.mark("d")
            self.at_step_start = False
            self.drafts = 0
        self.drafts += 1

    def on_verify(self):
        self.mark("v")
        self.at_step_start = True

    def fold(self):
        """Fold every complete step (d, v, next d) into the averages with upstream's smoothing."""
        m = self.marks
        i = 0
        while i + 2 < len(m):
            (k0, d0, _), (k1, v0, nd), (k2, d1, _) = m[i], m[i + 1], m[i + 2]
            if (k0, k1, k2) != ("d", "v", "d"):
                i += 1
                continue
            self.steps += 1
            if self.steps > 8 and nd > 0:  # do not learn capture / warm-up cost as steady state
                draft = d0.elapsed_time(v0) / nd
                verify = v0.elapsed_time(d1)
                mix = 1.0 if self.steps == 9 else 0.125
                self.draft_ms += mix * (draft - self.draft_ms)
                self.verify_ms += mix * (verify - self.verify_ms)
            i += 2
        del m[:i]


class _MaskedAssign(torch.overrides.TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is torch.Tensor.__setitem__ and len(args) == 3 and not kwargs:
            t, index, value = args
            if (isinstance(t, torch.Tensor) and isinstance(index, torch.Tensor) and index.dtype == torch.bool
                    and index.shape == t.shape and index.device == t.device
                    and (isinstance(value, (int, float)) or (isinstance(value, torch.Tensor) and value.dim() == 0))):
                t.masked_fill_(index, value)
                return None
        return func(*args, **(kwargs or {}))


def _install_sampling(llama):
    cls = llama.BaseGenerate
    original = getattr(cls, "processed_probs", None)
    if original is None or getattr(original, CLAIM, False):
        return False

    def processed_probs(self, logits, *args, **kwargs):
        if guard.off("autotree") or not isinstance(logits, torch.Tensor) or logits.device.type != "xpu":
            return original(self, logits, *args, **kwargs)
        with _MaskedAssign():
            return original(self, logits, *args, **kwargs)

    setattr(processed_probs, CLAIM, True)
    processed_probs.__wrapped__ = original
    cls.processed_probs = processed_probs
    return True


def install():
    from comfy.text_encoders import llama
    from comfy.text_encoders import qwen35 as q35
    original_best = getattr(q35, "best_widths", None)
    original_gen = getattr(getattr(q35, "Qwen35", None), "_generate_mtp", None)
    if original_best is None or original_gen is None or getattr(original_best, CLAIM, False):
        return False

    def best_widths(widths, hits, draft_ms, verify_ms):
        clock = _current["state"]
        if clock is not None and not guard.off("autotree"):
            try:
                clock.fold()
                if clock.steps > 8:
                    draft_ms, verify_ms = clock.draft_ms, clock.verify_ms
                    with _lock:
                        _stats.update(updates=_stats["updates"] + 1, steps=clock.steps,
                                      draft_ms=round(draft_ms, 3), verify_ms=round(verify_ms, 3))
            except Exception as e:  # timing only: upstream's own inputs remain valid
                guard.failed("autotree", e)
        return original_best(widths, hits, draft_ms, verify_ms)

    def _generate_mtp(self, embeds, *args, **kwargs):
        if guard.off("autotree") or getattr(embeds, "device", None) is None or embeds.device.type != "xpu" \
                or _current["state"] is not None:
            return original_gen(self, embeds, *args, **kwargs)
        clock = _Clock()
        model_forward, mtp_forward = self.model.forward, self.mtp.forward

        def timed_model_forward(*a, **k):
            if k.get("decode_buffers") is not None:  # the verify forward of a step
                clock.on_verify()
            return model_forward(*a, **k)

        def timed_mtp_forward(*a, **k):
            clock.on_draft()
            return mtp_forward(*a, **k)

        saved = [(m, vars(m).get("forward")) for m in (self.model, self.mtp)]  # outer wrappers, if any
        self.model.forward, self.mtp.forward = timed_model_forward, timed_mtp_forward
        _current["state"] = clock
        try:
            return original_gen(self, embeds, *args, **kwargs)
        finally:
            _current["state"] = None
            for module, previous in saved:
                if previous is None:
                    vars(module).pop("forward", None)
                else:
                    module.forward = previous

    for fn in (best_widths, _generate_mtp):
        setattr(fn, CLAIM, True)
    best_widths.__wrapped__ = original_best
    _generate_mtp.__wrapped__ = original_gen
    q35.best_widths = best_widths
    q35.Qwen35._generate_mtp = _generate_mtp
    _install_sampling(llama)
    return True
