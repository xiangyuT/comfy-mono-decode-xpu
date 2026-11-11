import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, SyclExtension

target = os.environ.get("CMDX_XPU_DEVICE", "bmg")
sycl_flags = ["-O3", "-g0", "-fsycl-targets=spir64_gen", "-Xs", f"-device {target}"]

setup(
    name="comfy_mono_decode_xpu",
    version="0.0.0.dev0",
    packages=["comfy_mono_decode_xpu"],
    ext_modules=[
        SyclExtension(
            "comfy_mono_decode_xpu._C",
            ["csrc/bindings.cpp", "csrc/w4a8_decode.sycl", "csrc/act_prep.sycl", "csrc/mono_mlp.sycl", "csrc/exact_probe.sycl", "csrc/mono_layer.sycl", "csrc/gdn_port.sycl", "csrc/mono_gdn.sycl", "csrc/attn_port.sycl", "csrc/attn_ref.sycl"],
            extra_compile_args={"cxx": ["-O3"], "sycl": sycl_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
