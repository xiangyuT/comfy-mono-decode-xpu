# SPDX-License-Identifier: Apache-2.0
"""Failure handling shared by every fast path.

A fast path that raises (a kernel's argument check, a SYCL error at submit, an out-of-memory) is
logged once with its traceback and switched off for the rest of the process; the call that hit the
error runs the original ComfyUI code instead. Callers only fall back when no state has been
modified yet; otherwise they re-raise after switching the path off (see layer.py).
"""
from __future__ import annotations

import logging
import threading

_lock = threading.Lock()
_off = {}  # piece -> reason


def off(piece):
    return piece in _off


def _is_interrupt(exc):
    try:
        import comfy.model_management
        return isinstance(exc, comfy.model_management.InterruptProcessingException)
    except Exception:
        return False


def failed(piece, exc, *, can_fallback=True):
    """Record a failure of `piece` and switch it off. Logs only the first failure per piece.
    A user interrupt (ComfyUI's cancel) is not a failure: it is re-raised unchanged."""
    if _is_interrupt(exc):
        raise exc
    with _lock:
        first = piece not in _off
        _off[piece] = f"{type(exc).__name__}: {exc}"
    if first:
        action = "trying the stock path" if can_fallback else "aborting this call without replaying modified state"
        logging.warning("[mono_decode_xpu] %s path failed and is now disabled; %s",
                        piece, action, exc_info=(type(exc), exc, exc.__traceback__))


def disabled():
    with _lock:
        return dict(_off)
