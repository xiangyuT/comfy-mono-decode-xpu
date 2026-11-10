#include <optional>
#include <torch/extension.h>

namespace cmdx {
torch::Tensor w4a8_decode_linear(torch::Tensor x, torch::Tensor x_scale, torch::Tensor packed,
                                 torch::Tensor s_rel, torch::Tensor s_channel,
                                 std::optional<torch::Tensor> codebook, std::optional<torch::Tensor> residual,
                                 int64_t group_size, int64_t out_dtype_code, int64_t cols, bool fast);
torch::Tensor read_checksum(torch::Tensor buf, int64_t wgs);
torch::Tensor mono_mlp_debug();
std::tuple<torch::Tensor, torch::Tensor> rot_quant(torch::Tensor x, torch::Tensor h, int64_t order);
torch::Tensor mono_mlp(torch::Tensor xn, torch::Tensor hadamard, torch::Tensor silu_lut,
                       torch::Tensor gq, torch::Tensor gs, torch::Tensor gsc, torch::Tensor gcb,
                       torch::Tensor uq, torch::Tensor us, torch::Tensor usc, torch::Tensor ucb,
                       torch::Tensor dq, torch::Tensor ds, torch::Tensor dsc, torch::Tensor dcb, int64_t nwg, int64_t stop);
}  // namespace cmdx

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("w4a8_decode_linear", &cmdx::w4a8_decode_linear, py::arg("x"), py::arg("x_scale"), py::arg("packed"),
          py::arg("s_rel"), py::arg("s_channel"), py::arg("codebook") = py::none(),
          py::arg("residual") = py::none(), py::arg("group_size") = 16, py::arg("out_dtype_code") = 2,
          py::arg("cols") = 0, py::arg("fast") = false);
    m.def("read_checksum", &cmdx::read_checksum);
    m.def("mono_mlp_debug", &cmdx::mono_mlp_debug);
    m.def("mono_mlp", &cmdx::mono_mlp, py::arg("xn"), py::arg("hadamard"), py::arg("silu_lut"), py::arg("gq"), py::arg("gs"),
          py::arg("gsc"), py::arg("gcb"), py::arg("uq"), py::arg("us"), py::arg("usc"), py::arg("ucb"), py::arg("dq"),
          py::arg("ds"), py::arg("dsc"), py::arg("dcb"), py::arg("nwg") = 0, py::arg("stop") = 4);
    m.def("rot_quant", &cmdx::rot_quant, py::arg("x"), py::arg("h"), py::arg("order") = 0);
}
