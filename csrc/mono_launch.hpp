// SPDX-License-Identifier: Apache-2.0
// Host-side residency and SLM planning for the persistent stream engine.
#pragma once
#include <algorithm>
#include <initializer_list>
#include <ATen/xpu/XPUContext.h>
#include <torch/extension.h>
#include "mono_common.hpp"

namespace mono_decode_xpu::mono {
inline int64_t xe_core_count(const torch::Tensor& like) {
    const auto* p = at::xpu::getDeviceProperties(like.get_device());
    TORCH_CHECK(p->gpu_eu_count_per_subslice > 0 && p->gpu_eu_count > 0 &&
                    p->gpu_eu_count % p->gpu_eu_count_per_subslice == 0,
                "persistent Xe-core geometry is unavailable");
    return p->gpu_eu_count / p->gpu_eu_count_per_subslice;
}

struct LaunchPlan {
    uint32_t workers;
    int64_t accumulator_elements;
};

constexpr size_t STREAM_SLM = TAB * sizeof(uint16_t) + MSG * sizeof(float) + 16 * sizeof(float);
constexpr size_t NORM_SLM = 1056 * sizeof(float);

inline LaunchPlan plan_launch(sycl::queue& queue, const torch::Tensor& like, int64_t requested,
                              std::initializer_list<Stream*> streams, size_t fixed_slm) {
    const int64_t cores = xe_core_count(like);
    TORCH_CHECK(requested >= 0 && requested <= cores,
                "persistent work-groups must be 0 (automatic) or <= Xe cores (", cores, ")");
    const auto workers = static_cast<uint32_t>(requested == 0 ? cores : requested);
    TORCH_CHECK(MWG <= queue.get_device().get_info<sycl::info::device::max_work_group_size>(),
                "persistent work-group exceeds device thread capacity");
    int64_t elements = 1;
    for (auto* stream : streams) {
        stream_finalize(*stream, workers);
        int64_t first, last;
        stream_range(*stream, 0, workers, first, last);
        // Group zero has the largest range; paired streams keep whole gate/up pairs.
        elements = std::max(elements, (last - first) * 8 * 16);
        stream->ksplit = choose_ksplit(last - first, stream->m[0].k);
    }
    const size_t bytes = fixed_slm + size_t(elements) * sizeof(int32_t);
    const auto capacity = queue.get_device().get_info<sycl::info::device::local_mem_size>();
    TORCH_CHECK(bytes <= capacity, "persistent SLM requirement ", bytes,
                " exceeds device capacity ", capacity, "; workers=", workers);
    return {workers, elements};
}
}  // namespace mono_decode_xpu::mono
