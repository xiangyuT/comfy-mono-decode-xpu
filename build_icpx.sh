#!/bin/bash
# Build comfy_mono_decode_xpu/_C in one icpx invocation (compile + device link together), the way
# omni_xpu_kernel is built. torch's SyclExtension path (per-file compile, then -fsycl-link) produced
# different device code for the same source (measured: Kitchen decode attention copied verbatim gave
# 1-ulp lse differences), so the bit-exact ports are validated against this build.
set -euo pipefail
cd "$(dirname "$0")"
TARGET="${MONO_DECODE_XPU_XPU_DEVICE:-bmg}"
PY=${PYTHON:-python}
T=$($PY -c "import torch, os; print(os.path.dirname(torch.__file__))")
PYINC=$($PY -c "import sysconfig; print(sysconfig.get_paths()['include'])")
SUFFIX=$($PY -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
SRCS=(csrc/bindings.cpp csrc/w4a8_decode.sycl csrc/act_prep.sycl csrc/mono_mlp.sycl csrc/exact_probe.sycl
      csrc/mono_layer.sycl csrc/gdn_port.sycl csrc/mono_gdn.sycl csrc/attn_port.sycl csrc/attn_ref.sycl csrc/fast_topk.sycl csrc/rope_kv.sycl)
icpx -fsycl -std=c++20 -O3 -DNDEBUG -fPIC -shared -fsycl-targets=spir64_gen -Xsycl-target-backend "-device ${TARGET}" \
  -I"$T/include" -I"$T/include/torch/csrc/api/include" -I"$PYINC" -Icsrc \
  -DTORCH_EXTENSION_NAME=_C -DTORCH_API_INCLUDE_EXTENSION_H \
  -x c++ "${SRCS[@]}" -o "comfy_mono_decode_xpu/_C${SUFFIX}" \
  -L"$T/lib" -ltorch -ltorch_python -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu -Wl,-rpath,"$T/lib" "$@"
echo "built comfy_mono_decode_xpu/_C${SUFFIX}"
