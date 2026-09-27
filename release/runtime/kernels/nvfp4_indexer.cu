// SPDX-License-Identifier: AGPL-3.0-only
// NVFP4 query x NVFP4 key sparse-indexer logits for DeepSeek V4.1 prefill on SM121.
//
//   logits[i, j] = sum_h weights[i, h] * relu(q[i, h, :] . k[j, :])     starts[i] <= j < ends[i]
//
// Both operands are 128 E2M1 values (64 packed bytes, element 2b in the low nibble of
// byte b) with one unsigned E4M3 scale per 16 values (8 bytes). The block-scaled
// tensor-core MMA consumes the scales directly:
//   mma.sync.m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X  e2m1 x e2m1, ue4m3
// Keys are the M operand (16 per warp tile), query heads the N operand (8 per tile),
// two K steps of 64 cover the head dimension. Like DeepGEMM's MXFP4 kernel (whose
// thread mapping this follows), ReLU, head weights and the 32-head sum are fused;
// the four lanes of a quad reduce with two shuffles.
//
// Work splits over query rows (blockIdx.x, two per CTA) and contiguous key slices
// (blockIdx.y, `split` keys, a multiple of 128) so a few decode rows still fill the GPU.
// Output contract: clean=0 matches DeepGEMM clean_logits=False (every key a CTA
// visits gets its logit or -inf outside that row's range; unvisited keys untouched;
// callers read only [starts[i], ends[i])). clean=1 writes -inf to every other column
// of its rows as well (graph decode contract), without computing those keys.
#include <cstdint>
#include <cuda_runtime.h>

namespace {

constexpr int kHeads = 32;
constexpr int kVectorBytes = 64;
constexpr int kScaleBytes = 8;
constexpr int kWarps = 8;
constexpr int kKeysPerWarp = 16;
constexpr int kKeysPerTile = kWarps * kKeysPerWarp;
#ifndef DS41_NVFP4_QUERIES
#define DS41_NVFP4_QUERIES 2
#endif
// Two query rows per CTA (128 registers, two CTAs per SM) matched DeepGEMM MXFP4 on GB10;
// four needed 206 registers and ran 5-10% slower, eight spilled.
constexpr int kQueries = DS41_NVFP4_QUERIES;  // each warp keeps these rows' fragments in registers

__device__ __forceinline__ uint32_t load_word(const uint8_t* pointer) {
    return __ldg(reinterpret_cast<const uint32_t*>(pointer));
}

__device__ __forceinline__ void mma_nvf4(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                         uint32_t scale_a, uint32_t scale_b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
        "{%10}, {%11, %12}, {%13}, {%14, %15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
          "r"(scale_a), "n"(0), "n"(0), "r"(scale_b), "n"(0), "n"(0));
#endif
}

__global__ void __launch_bounds__(kWarps * 32, 1)
nvfp4_logits_kernel(const uint8_t* __restrict__ q_values, const uint8_t* __restrict__ q_scales,
                    const float* __restrict__ weights, const uint8_t* __restrict__ k_values,
                    const uint8_t* __restrict__ k_scales, const int* __restrict__ starts,
                    const int* __restrict__ ends, float* __restrict__ out, int m, int n, long long ld,
                    int split, int clean) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;  // groupID: M row / N column within the MMA tile
    const int t = lane & 3;   // thread in group: K slice
    const int first = blockIdx.x * kQueries;
    const int slice_start = static_cast<int>(blockIdx.y) * split;
    const int slice_end = min(n, slice_start + split);

    int row_start[kQueries], row_end[kQueries];
    int span_start = n, span_end = 0;
#pragma unroll
    for (int qi = 0; qi < kQueries; ++qi) {
        const int row = first + qi;
        int s = 0, e = 0;
        if (row < m) {
            s = min(max(starts[row], 0), n);
            e = min(max(ends[row], 0), n);
        }
        row_start[qi] = s;
        row_end[qi] = e;
        if (s < e) {
            span_start = min(span_start, s);
            span_end = max(span_end, e);
        }
    }
    int lo = max(span_start, slice_start);
    const int hi = min(span_end, slice_end);
    int visited_lo = slice_start, visited_hi = slice_start;
    if (lo < hi) {
        lo &= ~(kKeysPerWarp - 1);  // slice_start is 16-aligned, so lo stays inside the slice
        visited_lo = lo;
        visited_hi = min(slice_end, (hi + kKeysPerWarp - 1) & ~(kKeysPerWarp - 1));

        // Query fragments (B operand, "col": each head's 64 bytes are one column).
        uint32_t b[kQueries][4][2][2];
        uint32_t scale_b[kQueries][4][2];
        float weight[kQueries][4][2];
    #pragma unroll
        for (int qi = 0; qi < kQueries; ++qi) {
            const int row = min(first + qi, m - 1);
    #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int head = nt * 8 + g;
                const uint8_t* vector = q_values + (static_cast<size_t>(row) * kHeads + head) * kVectorBytes;
                const uint8_t* scale = q_scales + (static_cast<size_t>(row) * kHeads + head) * kScaleBytes;
    #pragma unroll
                for (int k = 0; k < 2; ++k) {
                    b[qi][nt][k][0] = load_word(vector + k * 32 + 4 * t);
                    b[qi][nt][k][1] = load_word(vector + k * 32 + 16 + 4 * t);
                    scale_b[qi][nt][k] = load_word(scale + 4 * k);
                }
                const float* head_weights = weights + static_cast<size_t>(row) * kHeads + nt * 8 + 2 * t;
                weight[qi][nt][0] = __ldg(head_weights);
                weight[qi][nt][1] = __ldg(head_weights + 1);
            }
        }

        // Key fragments (A operand, row-major 16 x 64 bytes); out-of-range keys load zeros. The
        // next tile's fragments are fetched before the current tile's MMAs to hide L2 latency.
        uint32_t a[2][4], scale_a[2];
        const auto load_keys = [&](int base, uint32_t (&frag)[2][4], uint32_t (&scale)[2]) {
            const int key0 = base + g, key1 = base + g + 8, scale_row = base + g + (t & 1) * 8;
    #pragma unroll
            for (int k = 0; k < 2; ++k) {
                const int offset = k * 32 + 4 * t;
                frag[k][0] = key0 < n ? load_word(k_values + static_cast<size_t>(key0) * kVectorBytes + offset) : 0u;
                frag[k][1] = key1 < n ? load_word(k_values + static_cast<size_t>(key1) * kVectorBytes + offset) : 0u;
                frag[k][2] = key0 < n ? load_word(k_values + static_cast<size_t>(key0) * kVectorBytes + offset + 16) : 0u;
                frag[k][3] = key1 < n ? load_word(k_values + static_cast<size_t>(key1) * kVectorBytes + offset + 16) : 0u;
                scale[k] = scale_row < n ? load_word(k_scales + static_cast<size_t>(scale_row) * kScaleBytes + 4 * k) : 0u;
            }
        };
        load_keys(lo + warp * kKeysPerWarp, a, scale_a);
        for (int base = lo + warp * kKeysPerWarp; base < hi; base += kKeysPerTile) {
            const int key0 = base + g, key1 = base + g + 8;
            uint32_t a_next[2][4], scale_next[2];
            load_keys(base + kKeysPerTile, a_next, scale_next);
    #pragma unroll
            for (int qi = 0; qi < kQueries; ++qi) {
                float partial0 = 0.f, partial1 = 0.f;
    #pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    float d[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
                    for (int k = 0; k < 2; ++k)
                        mma_nvf4(d, a[k], b[qi][nt][k][0], b[qi][nt][k][1], scale_a[k], scale_b[qi][nt][k]);
                    partial0 += fmaxf(d[0], 0.f) * weight[qi][nt][0] + fmaxf(d[1], 0.f) * weight[qi][nt][1];
                    partial1 += fmaxf(d[2], 0.f) * weight[qi][nt][0] + fmaxf(d[3], 0.f) * weight[qi][nt][1];
                }
                partial0 += __shfl_xor_sync(0xffffffffu, partial0, 1);
                partial0 += __shfl_xor_sync(0xffffffffu, partial0, 2);
                partial1 += __shfl_xor_sync(0xffffffffu, partial1, 1);
                partial1 += __shfl_xor_sync(0xffffffffu, partial1, 2);
                const int row = first + qi;
                if (row < m && t < 2) {
                    const int key = t == 0 ? key0 : key1;
                    const float value = t == 0 ? partial0 : partial1;
                    if (key < n)
                        out[static_cast<size_t>(row) * ld + key] =
                            (key >= row_start[qi] && key < row_end[qi]) ? value : -INFINITY;
                }
            }
    #pragma unroll
            for (int k = 0; k < 2; ++k) {
    #pragma unroll
                for (int r = 0; r < 4; ++r)
                    a[k][r] = a_next[k][r];
                scale_a[k] = scale_next[k];
            }
        }
    }
    if (clean) {
        // Columns of this slice the MMA loop did not visit are outside every row's range.
        for (int qi = 0; qi < kQueries; ++qi) {
            const int row = first + qi;
            if (row >= m)
                break;
            float* destination = out + static_cast<size_t>(row) * ld;
            for (int key = slice_start + static_cast<int>(threadIdx.x); key < visited_lo; key += kWarps * 32)
                destination[key] = -INFINITY;
            for (int key = visited_hi + static_cast<int>(threadIdx.x); key < slice_end; key += kWarps * 32)
                destination[key] = -INFINITY;
        }
    }
}

}  // namespace

extern "C" int ds41_nvfp4_indexer_abi() { return 2; }

// Returns 0 on success, 1 for invalid arguments, 2 for a launch error.
extern "C" int ds41_nvfp4_logits(const void* q_values, const void* q_scales, const void* weights,
                                 const void* k_values, const void* k_scales, const void* starts,
                                 const void* ends, void* out, int m, int n, long long ld, void* stream,
                                 int split, int clean) {
    if (m < 0 || n < 0 || ld < n || !q_values || !q_scales || !weights || !k_values || !k_scales
            || !starts || !ends || !out || split <= 0 || split % kKeysPerTile || (clean != 0 && clean != 1))
        return 1;
    if (m == 0 || n == 0)
        return 0;
    const dim3 grid((m + kQueries - 1) / kQueries, (n + split - 1) / split);
    nvfp4_logits_kernel<<<grid, kWarps * 32, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t*>(q_values), static_cast<const uint8_t*>(q_scales),
        static_cast<const float*>(weights), static_cast<const uint8_t*>(k_values),
        static_cast<const uint8_t*>(k_scales), static_cast<const int*>(starts),
        static_cast<const int*>(ends), static_cast<float*>(out), m, n, ld, split, clean);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
