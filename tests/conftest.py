# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the kernels and the eligibility checks. No model is needed; model-level validation
goes through the ComfyUI UI (see tests/README.md). Run from the repo root: python -m pytest tests"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _xpu():
    return hasattr(torch, "xpu") and torch.xpu.is_available()


@pytest.fixture(scope="session")
def C():
    """The built extension on an XPU device (skips when either is missing)."""
    if not _xpu():
        pytest.skip("no XPU device")
    try:
        from comfy_mono_decode_xpu import _C
    except ImportError as e:
        pytest.skip(f"extension not built ({e}); run python install.py")
    return _C


def bits_equal(a, b):
    """Bitwise equality (NaN payloads and signed zeros included)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.element_size() == 2:
        return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))
    if a.element_size() == 4 and a.is_floating_point():
        return torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))
    return torch.equal(a, b)
