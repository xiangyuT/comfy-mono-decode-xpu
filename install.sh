#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Manual build, same as ComfyUI-Manager's install step: run inside the cloned node directory
# (ComfyUI/custom_nodes/comfy-mono-decode-xpu) with ComfyUI's Python.
#
#   ./install.sh [--force] [--target DEVICE]        (PYTHON=... selects the interpreter)
set -euo pipefail
cd "$(dirname "$0")"
exec "${PYTHON:-python}" install.py "$@"
