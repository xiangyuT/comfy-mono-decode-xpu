# Third-party notices

comfy-mono-decode-xpu is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)). The kernels
reproduce the numerics of existing XPU kernels bit for bit, and some files contain code ported from
third-party sources. Those sources and their licenses are listed below.

## Strata (MIT)

- Source: https://github.com/Niko1221/Strata. Received through intel/llm-scaler
  `omni/omni_xpu_kernel/omni_xpu_kernel/csrc/kitchen_decode.cpp` and `kitchen_deferred.cpp` at commit
  `fd4a584a7a4ec94f73f16cb4eb803e9d12f7dd52`. Both files are MIT-licensed and state that they follow
  Strata's SYCL QSA decode and verify kernels.
- Copyright (c) 2026 Niko1221 and the Strata contributors.
- License: MIT, full text in [licenses/Strata-MIT.txt](licenses/Strata-MIT.txt).
- Used in:
  - `csrc/attn_port.sycl`: port of the Kitchen tree-verify decode attention (chunked QK / softmax / PV,
    combine and step merge).
  - `csrc/mono_gdn.sycl`: port of the Kitchen deferred Gated-DeltaNet decode (deferred conv, gates,
    replay / verify delta).

## intel/llm-scaler, omni_xpu_kernel (Apache-2.0)

- Source: https://github.com/intel/llm-scaler, `omni/omni_xpu_kernel`.
- License: Apache License 2.0.
- These kernels reproduce its numerics bit for bit:
  - the packed W4A8 decode contract and FP32 scale-back (`csrc/w4a8_decode.sycl`, `csrc/mono_common.hpp`);
  - the RMSNorm reduction order (`csrc/mono_common.hpp`);
  - the rotary arithmetic (`csrc/rope_kv.sycl`);
  - the Kitchen decode kernels listed above, in `csrc/mono_layer.sycl`, `csrc/mono_mlp.sycl`,
    `csrc/mono_gdn.sycl` and `csrc/attn_port.sycl`.

## Comfy Org comfy-kitchen (Apache-2.0)

- Source: https://github.com/Comfy-Org/comfy-kitchen. Its NOTICE reads:
  "ComfyKitchen. Copyright (c) 2025 Comfy Org. All rights reserved."
- License: Apache License 2.0.
- `csrc/act_prep.sycl` and `csrc/mono_common.hpp` reproduce the semantics of its
  `quantize_and_rotate_rowwise` (ConvRot rotation + row-wise INT8 quantization).
- `comfy_mono_decode_xpu/` reads its quantized tensor layouts.

## Comfy Org workflow_templates (MIT)

- Reference workflows: https://github.com/Comfy-Org/workflow_templates.
- Model validation uses the official MOSS template through the ComfyUI UI. The template JSON and
  model-running development scripts are not distributed with this package.

## PyTorch / torch-xpu-ops (BSD-3-Clause)

- `csrc/fast_topk.sycl` reproduces the result ordering of torch XPU `topk` and `argmax`, including
  ties. This is behavioral compatibility only; no PyTorch code is included.

## Intel torch-xpu-ops (Apache-2.0)

- Source: https://github.com/intel/torch-xpu-ops at
  `9022a83389905509033d02cda8005c21be9c976c`,
  `src/ATen/native/xpu/sycl/LayerNormKernels.cpp`.
- Copyright 2020-2026 Intel Corporation. Licensed under Apache License 2.0.
- `csrc/rope_kv.sycl` preserves its D256 vectorized RMSNorm reduction and output order
  when fusing gated Q normalization with RoPE and KV writes.
- `csrc/sampling.sycl` follows the small-row FP32 softmax reduction and loop-scan
  arithmetic in `src/ATen/native/xpu/sycl/SoftMaxKernels.cpp` and `ScanUtils.h`
  at `7a617719077a9bdc1515661f2dd8b2af5cc26b2f` of the same Intel repository.
  The exponential RNG remains a Torch operation; no generator implementation is
  copied into this project.
