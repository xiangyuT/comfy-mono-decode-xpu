#include <optional>
#include <vector>
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
                       torch::Tensor dq, torch::Tensor ds, torch::Tensor dsc, torch::Tensor dcb, int64_t nwg, int64_t stop, bool tiled);
torch::Tensor rms_norm_probe(torch::Tensor x, torch::Tensor w, double eps, int64_t variant);
torch::Tensor linear_probe(torch::Tensor x, torch::Tensor w, int64_t variant);
torch::Tensor rms_stats();
torch::Tensor linear_ab(torch::Tensor x, torch::Tensor w);
torch::Tensor layer_gdn(torch::Tensor x, torch::Tensor norm_in, double eps_in, torch::Tensor hadamard,
                        std::vector<torch::Tensor> w4, torch::Tensor wa, torch::Tensor wb, torch::Tensor conv_w,
                        torch::Tensor conv_state, torch::Tensor proj_buf, torch::Tensor qkv_buf, torch::Tensor ctl,
                        torch::Tensor dt_bias, torch::Tensor g_decay, torch::Tensor rec_state, torch::Tensor gates_buf,
                        torch::Tensor sums_buf, torch::Tensor gnorm_w, double gnorm_eps, double qscale, int64_t hk,
                        torch::Tensor norm_post, double eps_post, torch::Tensor silu_lut, int64_t nwg, int64_t stop, bool tiled);
std::vector<torch::Tensor> layer_gdn_scratch(torch::Tensor like);
torch::Tensor attn_core(torch::Tensor q, torch::Tensor key, torch::Tensor value, torch::Tensor seqlen,
                        torch::Tensor xk, torch::Tensor xv, torch::Tensor mask, int64_t nwg);
std::vector<torch::Tensor> attn_core_debug();
std::vector<torch::Tensor> attn_ref(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor lengths, torch::Tensor dbg);
void gdn_conv_port(torch::Tensor proj, torch::Tensor state, torch::Tensor weight, std::optional<torch::Tensor> bias,
                   torch::Tensor prev, torch::Tensor qkv, torch::Tensor ctl);
torch::Tensor gdn_delta_port(torch::Tensor x, torch::Tensor wa, torch::Tensor wb, torch::Tensor dt, torch::Tensor gd,
                             torch::Tensor state, int64_t key_dim, int64_t hk, double scale, torch::Tensor z, torch::Tensor nw,
                             double eps, torch::Tensor qkv, torch::Tensor gates, torch::Tensor sums, torch::Tensor ctl,
                             bool at_linear);
std::vector<torch::Tensor> layer_pre(torch::Tensor x, torch::Tensor norm_w, double eps, torch::Tensor hadamard,
                                     std::vector<torch::Tensor> mats, int64_t nwg, bool tiled);
torch::Tensor layer_post(torch::Tensor x, torch::Tensor y, std::optional<torch::Tensor> gate, torch::Tensor sig_lut,
                         torch::Tensor oq, torch::Tensor os, torch::Tensor osc, torch::Tensor ocb,
                         torch::Tensor norm_w, double eps, torch::Tensor hadamard, torch::Tensor silu_lut,
                         torch::Tensor gq, torch::Tensor gs, torch::Tensor gsc, torch::Tensor gcb,
                         torch::Tensor uq, torch::Tensor us, torch::Tensor usc, torch::Tensor ucb,
                         torch::Tensor dq, torch::Tensor ds, torch::Tensor dsc, torch::Tensor dcb, int64_t nwg, int64_t stop, bool tiled);
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
          py::arg("ds"), py::arg("dsc"), py::arg("dcb"), py::arg("nwg") = 0, py::arg("stop") = 4, py::arg("tiled") = true);
    m.def("rms_norm_probe", &cmdx::rms_norm_probe);
    m.def("linear_probe", &cmdx::linear_probe);
    m.def("rms_stats", &cmdx::rms_stats);
    m.def("linear_ab", &cmdx::linear_ab);
    m.def("layer_gdn", &cmdx::layer_gdn);
    m.def("layer_gdn_scratch", &cmdx::layer_gdn_scratch);
    m.def("attn_core", &cmdx::attn_core);
    m.def("attn_core_debug", &cmdx::attn_core_debug);
    m.def("attn_ref", &cmdx::attn_ref);
    m.def("gdn_conv_port", &cmdx::gdn_conv_port);
    m.def("gdn_delta_port", &cmdx::gdn_delta_port);
    m.def("layer_pre", &cmdx::layer_pre, py::arg("x"), py::arg("norm_w"), py::arg("eps"), py::arg("hadamard"), py::arg("mats"),
          py::arg("nwg") = 0, py::arg("tiled") = true);
    m.def("layer_post", &cmdx::layer_post);
    m.def("rot_quant", &cmdx::rot_quant, py::arg("x"), py::arg("h"), py::arg("order") = 0);
}
