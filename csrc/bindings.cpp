// SPDX-License-Identifier: Apache-2.0
#include <optional>
#include <vector>
#include <torch/extension.h>

namespace mono_decode_xpu {
torch::Tensor w4a8_decode_linear(torch::Tensor x, torch::Tensor x_scale, torch::Tensor packed,
                                 torch::Tensor s_rel, torch::Tensor s_channel,
                                 std::optional<torch::Tensor> codebook, std::optional<torch::Tensor> residual,
                                 int64_t group_size, int64_t out_dtype_code, int64_t cols, bool fast);

std::tuple<torch::Tensor, torch::Tensor> rot_quant(torch::Tensor x, torch::Tensor h, int64_t order);
torch::Tensor mono_mlp(torch::Tensor xn, torch::Tensor hadamard, torch::Tensor silu_lut,
                       torch::Tensor gq, torch::Tensor gs, torch::Tensor gsc, torch::Tensor gcb,
                       torch::Tensor uq, torch::Tensor us, torch::Tensor usc, torch::Tensor ucb,
                       torch::Tensor dq, torch::Tensor ds, torch::Tensor dsc, torch::Tensor dcb, int64_t nwg, int64_t stop, bool tiled);

torch::Tensor layer_gdn(torch::Tensor x, torch::Tensor norm_in, double eps_in, torch::Tensor hadamard,
                        std::vector<torch::Tensor> w4, torch::Tensor wa, torch::Tensor wb, torch::Tensor conv_w,
                        torch::Tensor conv_state, torch::Tensor proj_buf, torch::Tensor qkv_buf, torch::Tensor ctl,
                        torch::Tensor dt_bias, torch::Tensor g_decay, torch::Tensor rec_state, torch::Tensor gates_buf,
                        torch::Tensor sums_buf, torch::Tensor gnorm_w, double gnorm_eps, double qscale, int64_t hk,
                        torch::Tensor norm_post, double eps_post, torch::Tensor silu_lut, int64_t nwg, int64_t stop, bool tiled);

torch::Tensor layer_gdn_core(torch::Tensor like, int64_t rows);
torch::Tensor attn_core(torch::Tensor q, torch::Tensor key, torch::Tensor value, torch::Tensor seqlen,
                        torch::Tensor xk, torch::Tensor xv, torch::Tensor mask, int64_t nwg);
torch::Tensor attn_dpas(torch::Tensor q, torch::Tensor key, torch::Tensor value, torch::Tensor seqlen,
                        torch::Tensor xk, torch::Tensor xv, torch::Tensor mask, int64_t chunks);

std::vector<torch::Tensor> fast_topk(torch::Tensor x, int64_t k, bool argmax_order);
std::vector<torch::Tensor> fast_pick(torch::Tensor x, int64_t k);
std::vector<torch::Tensor> rope_kv(torch::Tensor qn, torch::Tensor kn, torch::Tensor vn, torch::Tensor freqs,
                                   torch::Tensor key, torch::Tensor value, torch::Tensor position, int64_t rd);
std::vector<torch::Tensor> qk_norm_rope_kv(torch::Tensor rawq, torch::Tensor rawk, torch::Tensor vn,
    torch::Tensor weightq, torch::Tensor weightk, torch::Tensor freqs, torch::Tensor key,
    torch::Tensor value, torch::Tensor position, double epsq, double epsk, int64_t rd);

std::vector<torch::Tensor> layer_pre(torch::Tensor x, torch::Tensor norm_w, double eps, torch::Tensor hadamard,
                                     std::vector<torch::Tensor> mats, int64_t nwg, bool tiled);
torch::Tensor layer_post(torch::Tensor x, torch::Tensor y, std::optional<torch::Tensor> gate, torch::Tensor sig_lut,
                         torch::Tensor oq, torch::Tensor os, torch::Tensor osc, torch::Tensor ocb,
                         torch::Tensor norm_w, double eps, torch::Tensor hadamard, torch::Tensor silu_lut,
                         torch::Tensor gq, torch::Tensor gs, torch::Tensor gsc, torch::Tensor gcb,
                         torch::Tensor uq, torch::Tensor us, torch::Tensor usc, torch::Tensor ucb,
                         torch::Tensor dq, torch::Tensor ds, torch::Tensor dsc, torch::Tensor dcb, int64_t nwg, int64_t stop, bool tiled);
void mtp_prep_draft(torch::Tensor st, int64_t k, torch::Tensor ftab, torch::Tensor freqs, torch::Tensor mpos,
                    torch::Tensor mseq);
void mtp_prep_verify(torch::Tensor st, torch::Tensor apos, torch::Tensor aseq, torch::Tensor vpos,
                     torch::Tensor tree_pos, torch::Tensor ctl, torch::Tensor tree_ctl, int64_t parity,
                     int64_t rope_offset);
void mtp_accept(torch::Tensor st, torch::Tensor toks, torch::Tensor vtok, torch::Tensor cand, torch::Tensor children,
                int64_t rows, int64_t depth, torch::Tensor x, torch::Tensor nt_buf, torch::Tensor h_buf,
                torch::Tensor ctl, torch::Tensor kv_ptrs, int64_t kv_heads, int64_t kv_cap, int64_t head_dim,
                torch::Tensor result);
void draft_pick(torch::Tensor approx, torch::Tensor qx, torch::Tensor sx, torch::Tensor w, torch::Tensor sw,
                int64_t nout, bool argmax_first, torch::Tensor top, torch::Tensor ranks_row, torch::Tensor token,
                torch::Tensor vtok, std::vector<int64_t> children, torch::Tensor calls);
}  // namespace mono_decode_xpu

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("w4a8_decode_linear", &mono_decode_xpu::w4a8_decode_linear, py::arg("x"), py::arg("x_scale"), py::arg("packed"),
          py::arg("s_rel"), py::arg("s_channel"), py::arg("codebook") = py::none(),
          py::arg("residual") = py::none(), py::arg("group_size") = 16, py::arg("out_dtype_code") = 2,
          py::arg("cols") = 0, py::arg("fast") = false);
    m.def("mono_mlp", &mono_decode_xpu::mono_mlp, py::arg("xn"), py::arg("hadamard"), py::arg("silu_lut"), py::arg("gq"), py::arg("gs"),
          py::arg("gsc"), py::arg("gcb"), py::arg("uq"), py::arg("us"), py::arg("usc"), py::arg("ucb"), py::arg("dq"),
          py::arg("ds"), py::arg("dsc"), py::arg("dcb"), py::arg("nwg") = 0, py::arg("stop") = 4, py::arg("tiled") = true);
    m.def("layer_gdn", &mono_decode_xpu::layer_gdn);
    m.def("layer_gdn_core", &mono_decode_xpu::layer_gdn_core);
    m.def("attn_core", &mono_decode_xpu::attn_core);
    m.def("attn_dpas", &mono_decode_xpu::attn_dpas);
    m.def("mtp_prep_draft", &mono_decode_xpu::mtp_prep_draft);
    m.def("mtp_prep_verify", &mono_decode_xpu::mtp_prep_verify);
    m.def("mtp_accept", &mono_decode_xpu::mtp_accept);
    m.def("draft_pick", &mono_decode_xpu::draft_pick);
    m.def("fast_topk", &mono_decode_xpu::fast_topk);
    m.def("fast_pick", &mono_decode_xpu::fast_pick);
    m.def("rope_kv", &mono_decode_xpu::rope_kv);
    m.def("qk_norm_rope_kv", &mono_decode_xpu::qk_norm_rope_kv);
    m.def("layer_pre", &mono_decode_xpu::layer_pre, py::arg("x"), py::arg("norm_w"), py::arg("eps"), py::arg("hadamard"), py::arg("mats"),
          py::arg("nwg") = 0, py::arg("tiled") = true);
    m.def("layer_post", &mono_decode_xpu::layer_post);
    m.def("rot_quant", &mono_decode_xpu::rot_quant, py::arg("x"), py::arg("h"), py::arg("order") = 0);
}
