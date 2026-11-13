# SPDX-License-Identifier: Apache-2.0
"""Device MTP loop installation against the ComfyUI it runs in: the cache hooks it relies on (prepare and
advance are no-ops for caches the loop owns, untouched for any other cache) and the upstream source
fingerprint that gates the loop (it reimplements ComfyUI v0.45's _generate_mtp and KV / tree state)."""
import pytest

llama = pytest.importorskip("comfy.text_encoders.llama")
q35 = pytest.importorskip("comfy.text_encoders.qwen35")

from comfy_mono_decode_xpu import mtp_loop  # noqa: E402


def test_advance_is_a_noop_only_for_device_owned_caches():
    mtp_loop._install_device_kv(llama, q35)
    owned, other = object.__new__(llama.FixedKV), object.__new__(llama.FixedKV)  # (a dataclass: no tensors needed)
    owned.index = other.index = 5
    owned._mono_device = True
    owned.advance(8)
    other.advance(8)
    assert owned.index == 5 and other.index == 13
    # rollback is upstream's own: the device loop never calls it
    other.rollback(3)
    assert other.index == 10


def test_upstream_fingerprint_matches_v045():
    try:
        import comfyui_version
    except ImportError:
        pytest.skip("not a ComfyUI checkout")
    if comfyui_version.__version__ != "0.45.0":
        pytest.skip("the device loop is written against ComfyUI 0.45.0; other versions run upstream's loop")
    original = getattr(q35.Qwen35._generate_mtp, "__wrapped__", q35.Qwen35._generate_mtp)
    while getattr(original, "__wrapped__", None) is not None:  # under the plugin: the innermost (upstream)
        original = original.__wrapped__
    assert mtp_loop._fingerprint(llama, q35, original) == mtp_loop._UPSTREAM
