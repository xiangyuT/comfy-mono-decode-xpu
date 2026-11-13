// SPDX-License-Identifier: Apache-2.0
// Shared device code for the persistent decode kernels (mono MLP, layer pre/post).
#pragma once
#include <cstdint>

#include <c10/util/Float8_e4m3fn.h>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/clock.hpp>
#include <sycl/ext/oneapi/experimental/prefetch.hpp>

namespace mono_decode_xpu {
namespace mono {
using bf16 = sycl::ext::oneapi::bfloat16;
using fp8 = c10::Float8_e4m3fn;
namespace jm = sycl::ext::oneapi::experimental::matrix;
constexpr int MWG = 1024;                // threads per work-group (64 sub-groups: all hardware threads of an Xe core)
constexpr int MSG = MWG / 16;
constexpr int GS = 256;                  // ConvRot group
constexpr int MAXK = 12288;

// W4 layout used by the persistent kernels ("tile-major", repacked once per linear on the host):
//   q: [N/16][K/32][16 lanes][16 B]  = qdata.view(N/16, 16, K/32, 16).permute(0, 2, 1, 3)
//   s: [N/16][K/256][16 lanes][16 B] = s_rel.view(N/16, 16, K/256, 16).permute(0, 2, 1, 3)
// so every K32 step of a 16-column tile is one contiguous 256 B block (row-major rows were 16
// separate 16 B pieces 2-6 KB apart: 14-34% lower single-kernel bandwidth).
struct W4 {
    const int8_t* q;      // tile-major packed nibbles (N * K/2 bytes)
    const fp8* s;         // tile-major relative group scales (N * K/16 bytes)
    const float* sc;      // [N] channel scales
    const float* cb;      // [16] codebook
    int64_t n, k;
};

template <typename T>
inline auto gptr(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p);
}
template <typename T>
inline auto lptr(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
}

inline void grid_sync(sycl::nd_item<1>& it, uint32_t* counter, uint32_t nwg, uint32_t& epoch) {
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    sycl::group_barrier(it.get_group());
    if (it.get_local_id(0) == 0) {
        sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            a(*counter);
        const uint32_t target = (++epoch) * nwg;
        a.fetch_add(1u);
        // poll with an atomic RMW: a plain atomic load can be served stale and never observe
        // the other work-groups' arrivals on Xe2 (measured hang); RMW polling costs ~0.4 us/barrier
        while (a.fetch_add(0u) < target) {
        }
    } else {
        ++epoch;
    }
    sycl::group_barrier(it.get_group());
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
}

// the EU math.rsqt (what ESIMD rsqrt lowers to); SPMD sycl::native::rsqrt lowering varies by build
inline float hw_rsqrt(float x) {
#if defined(__SYCL_DEVICE_ONLY__)
    float r;
    __asm__ volatile("rsqrt (M1, 16) %0(0,0)<1> %1(0,0)<1;1,0>" : "=rw"(r) : "rw"(x));
    return r;
#else
    return x;
#endif
}

using LPtrU16 = sycl::multi_ptr<uint16_t, sycl::access::address_space::local_space, sycl::access::decorated::no>;
using LPtrF = sycl::multi_ptr<float, sycl::access::address_space::local_space, sycl::access::decorated::no>;
constexpr int TAB = 128 * 256;
constexpr int ACC = 64 * 128;
static_assert(MWG == 1024, "pair table build assumes 1024 threads");

// exact pair table for one codebook: entry (s << 8 | byte) holds the INT8-grid values of the low and
// high nibble codes of `byte` at FP8 scale byte s, computed with the dec4 formula
template <typename CB>
inline void build_pair_table(sycl::nd_item<1>& it, LPtrU16 tab, const CB& cb) {
    // singles [128 scale bytes][16 codes] (the dec4 formula), staged in the upper half of the table
    // region that the pair pass fills last
    const int t = it.get_local_id(0);
    auto single = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
        reinterpret_cast<uint8_t*>(tab.get()) + 2 * TAB - 2048);
    for (int i = t; i < 2048; i += MWG) {
        const float scale = static_cast<float>(sycl::bit_cast<fp8>(uint8_t(i >> 4)));
        const float r = sycl::rint(cb[i & 15] * scale);
        const int32_t q = sycl::isnan(r) ? 0 : static_cast<int32_t>(sycl::clamp(r, -127.f, 127.f));
        single[i] = static_cast<uint8_t>(q & 0xFF);
    }
    sycl::group_barrier(it.get_group());
    // pairs: entries [0, TAB - 1024) never overlap the singles' bytes; the last 1024 entries are
    // computed into registers first, then written after a barrier
    uint16_t tail = 0;
    for (int i = t; i < TAB; i += MWG) {
        const int s = i >> 8, byte = i & 255;
        const uint16_t v = uint16_t(single[s * 16 + (byte & 15)]) | uint16_t(uint16_t(single[s * 16 + (byte >> 4)]) << 8);
        if (i < TAB - 1024) tab[i] = v;
        else tail = v;
    }
    sycl::group_barrier(it.get_group());
    if (TAB - 1024 + t < TAB && t < 1024) tab[TAB - 1024 + t] = tail;
}

// ---- stream GEMV engine ---------------------------------------------------------------------------
// A phase multiplies one INT8 activation block xq [8, K] by up to three packed-W4 matrices that share
// K. The global tile list (16 output columns per tile; matrices concatenated, or gate/up interleaved
// when PAIR) is split evenly over the work-groups; inside a work-group, every (tile, K-chunk) unit
// goes to sub-group u % 64, and each sub-group streams all of its units back to back with a prefetch
// ring that crosses unit boundaries. Split-K partials are added into SLM accumulators with integer
// atomics: INT32 addition is associative, so the result is bit-identical to any fixed order.
struct Stream {
    W4 m[3];
    int nm;            // matrices
    bool pair;         // tile list g0,u0,g1,u1,... (nm == 2)
    int64_t tiles;     // total tiles over all matrices
    int ksplit;        // K chunks per tile (power of two)
    bool tiled;        // tile-major W4 copy (resident weights) or the checkpoint's row-major layout
    int pblock;        // PAIR: tiles per work-group block; a block lists its gate tiles, then its up tiles
};

inline Stream make_stream(const W4* ms, int nm, bool pair) {
    Stream st{};
    st.nm = nm;
    st.pair = pair;
    st.tiles = 0;
    for (int i = 0; i < 3; ++i) st.m[i] = ms[i < nm ? i : 0];
    for (int i = 0; i < nm; ++i) st.tiles += ms[i].n / 16;
    st.ksplit = 1;
    st.tiled = true;
    st.pblock = 0;
    return st;
}

// work-group tile range of a stream
inline void stream_range(const Stream& st, uint32_t wg, uint32_t nw, int64_t& t0, int64_t& t1) {
    int64_t per = (st.tiles + nw - 1) / nw;
    if (st.pair) per += per & 1;  // keep gate/up tile pairs in one work-group
    t0 = sycl::min<int64_t>(st.tiles, int64_t(wg) * per);
    t1 = sycl::min<int64_t>(st.tiles, t0 + per);
}

// global tile -> (matrix index, local tile); selects without dynamic indexing of kernel arguments
inline int stream_which(const Stream& st, int g, int& tile) {
    if (st.pair) { tile = g >> 1; return g & 1; }
    const int a = int(st.m[0].n >> 4), b = a + int(st.m[1].n >> 4);
    if (st.nm < 2 || g < a) { tile = g; return 0; }
    if (st.nm < 3 || g < b) { tile = g - a; return 1; }
    tile = g - b;
    return 2;
}

// W: K32 steps per weight load per lane. Tile-major blocks are already contiguous across the lanes
// (W = 1, 16 B); row-major rows are 2-6 KB apart per lane, so each lane fetches a whole 64 B line
// (W = 4) instead of a quarter line per step.
#ifndef ROW_W
#define ROW_W 4
#endif
template <int VAR, int W, typename ACC>
inline void stream_gemv_impl(sycl::nd_item<1>& it, const int8_t* xq, const Stream& st, int64_t t0_, int64_t t1_,
                             LPtrU16 tab, ACC acc) {
    auto sg = it.get_sub_group();
    const int lane = sg.get_local_id()[0], sgi = sg.get_group_id()[0];
    // 32-bit index math throughout; ksplit is a power of two (shift/mask, no division)
    const int K = int(st.m[0].k), kl2 = sycl::ctz(st.ksplit), kmask = st.ksplit - 1;
    const int klen = K >> kl2, steps = klen >> 5, blocks = steps >> 3;  // block = 8 steps = K256
    const int t0 = int(t0_), units = int(t1_ - t0_) << kl2;
    // sub-group sgi takes a contiguous run of units (consecutive K chunks of the same tiles), so each
    // lane reads long sequential runs of its weight row; INT32 atomics make the split order-free
    const int uq = units / MSG, ur = units % MSG;
    const int mine = uq + (sgi < ur ? 1 : 0);
    const int ubeg = sgi * uq + sycl::min(sgi, ur);
    const int total = mine * steps;
    const int8_t* q0 = st.m[0].q; const int8_t* q1 = st.m[1].q; const int8_t* q2 = st.m[2].q;
    const fp8* s0 = st.m[0].s; const fp8* s1 = st.m[1].s; const fp8* s2 = st.m[2].s;
    const bool tiled = st.tiled;
    // tile-major weights (see the W4 layout note): a sub-group's K32 step is one contiguous 256 B block
    auto tile_of = [&](int u, int& which) {
        int tile;
        which = stream_which(st, t0 + (u >> kl2), tile);
        return int64_t(tile);
    };
    auto wptr = [&](int lu, int ls) {
        const int u = ubeg + lu;
        int which;
        const int64_t tile = tile_of(u, which);
        const int kk = (u & kmask) * klen + ls * 32;
        const int8_t* q = which == 0 ? q0 : (which == 1 ? q1 : q2);
        return reinterpret_cast<const uint32_t*>(tiled ? q + ((tile * (K / 32) + kk / 32) * 16 + lane) * 16
                                                       : q + (tile * 16 + lane) * (K / 2) + kk / 2);
    };
    auto sptr = [&](int su, int sb) {  // 16 B = scale pairs of one block (8 steps)
        const int u = ubeg + su;
        int which;
        const int64_t tile = tile_of(u, which);
        const int kk = (u & kmask) * klen + sb * 256;
        const fp8* s = which == 0 ? s0 : (which == 1 ? s1 : s2);
        return reinterpret_cast<const uint32_t*>(tiled ? s + ((tile * (K / 256) + kk / 256) * 16 + lane) * 16
                                                       : s + (tile * 16 + lane) * (K / 16) + kk / 16);
    };
    constexpr int PF = 4;                 // steps of weights in flight
    constexpr int NS = PF / W > 1 ? PF / W : 2;  // ring slots of W steps each
    sycl::vec<uint32_t, 4 * W> ring[NS];
    sycl::vec<uint32_t, 4> scur, snxt;
    int lu = 0, ls = 0, su = 0, sb = 0;
    auto advance_w = [&]() { ls += W; if (ls == steps) { ls = 0; ++lu; } };
    auto advance_s = [&]() { if (++sb == blocks) { sb = 0; ++su; } };
#pragma unroll
    for (int p = 0; p < NS; ++p)
        if (p * W < total) { ring[p].load(0, gptr(wptr(lu, ls))); advance_w(); }
    if (total > 0) { snxt.load(0, gptr(sptr(su, sb))); advance_s(); }
    jm::joint_matrix<sycl::sub_group, int8_t, jm::use::a, 8, 32, jm::layout::row_major> a;
    jm::joint_matrix<sycl::sub_group, int8_t, jm::use::b, 32, 16, jm::layout::ext_intel_packed> b;
    jm::joint_matrix<sycl::sub_group, int32_t, jm::use::accumulator, 8, 16> c;
    jm::joint_matrix_fill(sg, c, 0);
    int cu = 0, cs = 0;
    for (int i0 = 0; i0 < total; i0 += 8) {
        scur = snxt;
        if (i0 + 8 < total) { snxt.load(0, gptr(sptr(su, sb))); advance_s(); }
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            const int i = i0 + p;
            const int slot = (p / W) % NS, sub = p % W;
            sycl::vec<uint32_t, 4> pk;
#pragma unroll
            for (int w = 0; w < 4; ++w) pk[w] = ring[slot][4 * sub + w];
            if (sub == W - 1 && i + 1 + (NS - 1) * W < total) { ring[slot].load(0, gptr(wptr(lu, ls))); advance_w(); }
            const uint32_t sc = (scur[p >> 1] >> (16 * (p & 1))) & 0xFFFFu;
            const int u = ubeg + cu;
            const int kk = (u & kmask) * klen + cs * 32;
            jm::joint_matrix_load(sg, a, gptr(xq + kk), K);
            const uint32_t ba = (sc & 0xFFu) << 8, bb = (sc >> 8) << 8;
            uint32_t d[8];
            if constexpr (VAR == 1) {
#pragma unroll
                for (int wi = 0; wi < 4; ++wi) { d[2 * wi] = pk[wi] ^ ba; d[2 * wi + 1] = pk[wi] ^ bb; }
            } else
#pragma unroll
            for (int wi = 0; wi < 4; ++wi) {
                const uint32_t word = pk[wi], base = wi < 2 ? ba : bb;
                d[2 * wi] = uint32_t(tab[base | (word & 0xFFu)]) | (uint32_t(tab[base | ((word >> 8) & 0xFFu)]) << 16);
                d[2 * wi + 1] = uint32_t(tab[base | ((word >> 16) & 0xFFu)]) | (uint32_t(tab[base | (word >> 24)]) << 16);
            }
            int e = 0;
            jm::joint_matrix_apply(sg, b, [&](int8_t& v) {
                const int j = e++;
                v = static_cast<int8_t>((d[j >> 2] >> (8 * (j & 3))) & 0xFF);
            });
            jm::joint_matrix_mad(sg, c, a, b, c);
            if (++cs == steps) {
                cs = 0; ++cu;
                // accumulator fragment: element j of a lane is row j, column = lane
                const int base = (u >> kl2) * 128 + lane;
                int j = 0;
                jm::joint_matrix_apply(sg, c, [&](int32_t& v) {
                    sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space> ar(acc[base + 16 * j]);
                    ar.fetch_add(v);
                    ++j;
                });
                jm::joint_matrix_fill(sg, c, 0);
            }
        }
    }
}

template <int VAR = 0, typename ACC>
inline void stream_gemv(sycl::nd_item<1>& it, const int8_t* xq, const Stream& st, int64_t t0_, int64_t t1_,
                        LPtrU16 tab, ACC acc) {
    if (st.tiled) stream_gemv_impl<VAR, 1>(it, xq, st, t0_, t1_, tab, acc);
    else stream_gemv_impl<VAR, ROW_W>(it, xq, st, t0_, t1_, tab, acc);
}

// Warm the L2 with the first `budget` bytes of this work-group's share of a coming GEMV phase, so
// phases that leave the memory system idle (row-wise rot_quant, the GDN core) overlap the next
// phase's weight stream. Tiles are prefetched in the order the engine will consume them.
inline void prefetch_stream(sycl::nd_item<1>& it, const Stream& st, uint32_t wg, uint32_t nw, int64_t budget) {
    namespace syclex = sycl::ext::oneapi::experimental;
    int64_t t0, t1;
    stream_range(st, wg, nw, t0, t1);
    const int t = it.get_local_id(0);
    const int64_t K = st.m[0].k, wbytes = 16 * K / 2, sbytes = 16 * K / 16;
    int64_t used = 0;
    for (int64_t g = t0; g < t1 && used < budget; ++g, used += wbytes + sbytes) {
        int tile;
        const int m = stream_which(st, int(g), tile);
        const int8_t* q = m == 0 ? st.m[0].q : (m == 1 ? st.m[1].q : st.m[2].q);
        const fp8* s = m == 0 ? st.m[0].s : (m == 1 ? st.m[1].s : st.m[2].s);
        char* wp = const_cast<char*>(reinterpret_cast<const char*>(q + int64_t(tile) * 16 * (K / 2)));
        char* sp = const_cast<char*>(reinterpret_cast<const char*>(s + int64_t(tile) * 16 * (K / 16)));
        for (int64_t off = int64_t(t) * 64; off < wbytes; off += int64_t(MWG) * 64)
            syclex::prefetch(wp + off, syclex::properties{syclex::prefetch_hint_L2});
        for (int64_t off = int64_t(t) * 64; off < sbytes; off += int64_t(MWG) * 64)
            syclex::prefetch(sp + off, syclex::properties{syclex::prefetch_hint_L2});
    }
}

// host: fix the work-group split of a stream (PAIR streams need it in stream_which)
inline void stream_finalize(Stream& st, uint32_t nw) {
    int64_t per = (st.tiles + nw - 1) / nw;
    if (st.pair) per += per & 1;
    st.pblock = st.pair ? int(per) : 0;
}

// host: K-split that minimises the busiest sub-group's step count
inline int choose_ksplit(int64_t tiles_per_wg, int64_t K) {
    int best = 1;
    int64_t cost = -1;
    for (int ks = 1; ks <= 16; ks *= 2) {
        if ((K / ks) % 256) break;  // whole K256 scale blocks per unit
        const int64_t c = ((tiles_per_wg * ks + MSG - 1) / MSG) * (K / ks / 32) + ks;  // + small per-unit cost
        if (cost < 0 || c < cost) { cost = c; best = ks; }
    }
    return best;
}

// rot_quant of one row (see act_prep.sycl), done by one work-group into q[row], s[row]
template <typename RED>
inline void rot_quant_row(sycl::nd_item<1>& it, const bf16* x, const bf16* h, int8_t* q, float* s, int64_t K,
                          LPtrF xr, RED& red) {
    auto sg = it.get_sub_group();
    const int t = it.get_local_id(0), sgi = sg.get_group_id()[0];
    const int64_t tiles = (K / GS / 8) * (GS / 16);
    for (int64_t tile = sgi; tile < tiles; tile += MSG) {
        const int64_t gb = tile / (GS / 16);
        const int cb = tile % (GS / 16);
        constexpr int KB = 4;  // k-steps whose loads are in flight together (a load-mad chain of 16 round trips)
        jm::joint_matrix<sycl::sub_group, bf16, jm::use::a, 8, 16, jm::layout::row_major> a[KB];
        jm::joint_matrix<sycl::sub_group, bf16, jm::use::b, 16, 16, jm::layout::row_major> b[KB];
        jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> c;
        jm::joint_matrix_fill(sg, c, 0.f);
        for (int k0 = 0; k0 < GS; k0 += 16 * KB) {
#pragma unroll
            for (int u = 0; u < KB; ++u) {
                jm::joint_matrix_load(sg, a[u], gptr(x + gb * 8 * GS + k0 + 16 * u), GS);
                jm::joint_matrix_load(sg, b[u], gptr(h + (k0 + 16 * u) * GS + cb * 16), GS);
            }
#pragma unroll
            for (int u = 0; u < KB; ++u) jm::joint_matrix_mad(sg, c, a[u], b[u], c);
        }
        jm::joint_matrix_store(
            sg, c,
            xr + (gb * 8 * GS + cb * 16), GS, jm::layout::row_major);
    }
    sycl::group_barrier(it.get_group());
    float amax = 0.f;
    for (int64_t i = t; i < K; i += MWG) {
        const float v = static_cast<float>(static_cast<bf16>(xr[i]));
        xr[i] = v;
        amax = sycl::fmax(amax, sycl::fabs(v));
    }
    const float m = sycl::reduce_over_group(sg, amax, sycl::maximum<float>());
    if (sg.get_local_id()[0] == 0) red[sgi] = m;
    sycl::group_barrier(it.get_group());
    float rmax = 0.f;
#pragma unroll
    for (int i = 0; i < MSG; ++i) rmax = sycl::fmax(rmax, red[i]);
    const float sv = sycl::fmax(rmax / 127.0f, 1e-30f);
    if (t == 0) *s = sv;
    bf16 sb = static_cast<bf16>(sv);
    if (static_cast<float>(sb) == 0.f) sb = sycl::bit_cast<bf16>(uint16_t(0x0080));
    const float sbf = static_cast<float>(sb);
    for (int64_t i = t; i < K; i += MWG) {
        const float d = static_cast<float>(static_cast<bf16>(xr[i] / sbf));
        q[i] = static_cast<int8_t>(sycl::clamp(sycl::rint(d), -128.f, 127.f));
    }
    sycl::group_barrier(it.get_group());
}

// codebook in SLM addressed through the accessor (local address space): a generic pointer
// makes every lookup an address-space-resolved access, which was the decode bottleneck
struct SlmCb {
    sycl::local_accessor<float, 1> acc;
    int off;
    float operator[](int i) const { return acc[off + i]; }
};

// OmniXPU rms_norm_kernel<bf16, GS=32, BS=64> for one H = 4096 row by one work-group, bit-exact:
// ESIMD thread et (32 of them) owns blocks 2et, 2et+1 of 64 lanes: acc = fma(x1, x1, x0 * x0) per lane,
// halving-tree sum over the 64 lanes, * 2^-12, halving-tree sum over the 32 threads, + eps, math.rsqt,
// out = bf16((x * scale) * w). red: >= 1056 local floats.
inline void rms_norm_row(sycl::nd_item<1>& it, const bf16* x, const bf16* w, bf16* out, float eps,
                         sycl::multi_ptr<float, sycl::access::address_space::local_space, sycl::access::decorated::no> red) {
#pragma clang fp contract(off)
    auto sg = it.get_sub_group();
    const int t = it.get_local_id(0), lane = sg.get_local_id()[0], sgi = sg.get_group_id()[0];
    const int et = t >> 5, j0 = t & 31;
    float wv[4096 / MWG];  // the (cold) weight is requested before the reductions
#pragma unroll
    for (int k = 0; k < 4096 / MWG; ++k) wv[k] = static_cast<float>(w[t + k * MWG]);
    float v[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int j = j0 + 32 * h;
        const float x0 = static_cast<float>(x[(2 * et) * 64 + j]);
        const float x1 = static_cast<float>(x[(2 * et + 1) * 64 + j]);
        const float p = x0 * x0;
        v[h] = sycl::fma(x1, x1, p);
    }
    red[et * 32 + j0] = v[0] + v[1];  // 64 -> 32
    sycl::group_barrier(it.get_group());
    if (sgi < 32) {  // sub-group sgi reduces emulated thread et = sgi: 32 -> 16 -> ... -> 1
        float a = red[sgi * 32 + lane] + red[sgi * 32 + lane + 16];
        a = a + sycl::shift_group_left(sg, a, 8);
        a = a + sycl::shift_group_left(sg, a, 4);
        a = a + sycl::shift_group_left(sg, a, 2);
        a = a + sycl::shift_group_left(sg, a, 1);
        if (lane == 0) red[1024 + sgi] = a * 0.000244140625f;  // / 4096 (math.inv(4096) = 2^-12)
    }
    sycl::group_barrier(it.get_group());
    if (sgi == 0) {
        float a = red[1024 + lane] + red[1024 + lane + 16];
        a = a + sycl::shift_group_left(sg, a, 8);
        a = a + sycl::shift_group_left(sg, a, 4);
        a = a + sycl::shift_group_left(sg, a, 2);
        a = a + sycl::shift_group_left(sg, a, 1);
        const float scale = hw_rsqrt(sycl::group_broadcast(sg, a, 0) + eps);
        if (lane == 0) red[1056 - 1] = scale;
    }
    sycl::group_barrier(it.get_group());
    const float scale = red[1056 - 1];
#pragma unroll
    for (int k = 0; k < 4096 / MWG; ++k) {
        const int i = t + k * MWG;
        const float xv = static_cast<float>(x[i]);
        const float s1 = xv * scale;
        out[i] = static_cast<bf16>(s1 * wv[k]);
    }
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::work_group);
    sycl::group_barrier(it.get_group());
}

inline bf16 bf16_add(bf16 a, bf16 b) { return static_cast<bf16>(static_cast<float>(a) + static_cast<float>(b)); }
inline bf16 bf16_mul(bf16 a, bf16 b) { return static_cast<bf16>(static_cast<float>(a) * static_cast<float>(b)); }
// fused_scaleback: float(acc) * x_scale[r] * channel_scale[n], one BF16 cast
inline bf16 scaleback(int32_t acc, float xs, float sc) {
#pragma clang fp contract(off)
    float f = static_cast<float>(acc);
    f = f * xs;
    f = f * sc;
    return static_cast<bf16>(f);
}
}  // namespace mono
}  // namespace mono_decode_xpu
