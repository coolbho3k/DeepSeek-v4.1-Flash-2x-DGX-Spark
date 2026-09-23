// SPDX-License-Identifier: AGPL-3.0-only
// Derived from MiaAI Lab / Wesley Young grouped MoE and Turboderp ExLlamaV3.
// Local change: fuse gate/up input gather; bound the grid by routed batch size.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "util.cuh"
#include "quant/hadamard_inner.cuh"
namespace {

constexpr int FM_THREADS = 256;                 // 8 warps
constexpr int FM_WARPS = FM_THREADS / 32;
constexpr int FM_TILE_N = 128;                  // one 16-col block per warp
constexpr int FM_TILE_K = 32;                   // per pipeline stage
constexpr int FM_STAGES = 4;
constexpr float FM_HAD_SCALE = 0.088388347648f; // 1/sqrt(128)

constexpr int FM_MB_GATEUP = 4;                 // 64-row tiles, 2 B streams
constexpr int FM_MB_DOWN = 4;                   // 64-row tiles, 2 B streams (256 cols)

template <int BITS>
struct FmPack
{
    static constexpr int words = BITS * 16;                     // int16 words per 16x16 tile
    static constexpr int chunks = words / 8;                    // 16 B chunks per tile
    static constexpr int b_stage_words = 2 * FM_WARPS * words;  // 2 k16 sub-steps x 8 warps
};

template <int MB, int NS, int BITS, int NA>
constexpr int fm_smem_bytes()
{
    constexpr int a_stage = NA * MB * 16 * FM_TILE_K * 2;
    constexpr int b_stage = NS * FmPack<BITS>::b_stage_words * 2;
    constexpr int pipe = FM_STAGES * (a_stage + b_stage);
    constexpr int epi = 16 * NS * FM_TILE_N * 4;
    return pipe > epi ? pipe : epi;
}

// 128-element Hadamard over one row held as float4 per lane, then optional
// per-column scale. Same arithmetic as fat_had_ff_128 / had_ff_r_128_inner.
__device__ __forceinline__ void fm_had_row(float4& v, int lane)
{
    float s0 = v.x + v.y;
    float d0 = v.x - v.y;
    float s1 = v.z + v.w;
    float d1 = v.z - v.w;
    v.x = s0 + s1;
    v.y = d0 + d1;
    v.z = s0 - s1;
    v.w = d0 - d1;
    shuffle_had_f2x32(v.x, v.y, lane);
    shuffle_had_f2x32(v.z, v.w, lane);
    v.x *= FM_HAD_SCALE;
    v.y *= FM_HAD_SCALE;
    v.z *= FM_HAD_SCALE;
    v.w *= FM_HAD_SCALE;
}

__device__ __forceinline__ float4 fm_load_half4(const half* p)
{
    half4 h = *reinterpret_cast<const half4*>(p);
    return make_float4(__low2float(h.x), __high2float(h.x), __low2float(h.y), __high2float(h.y));
}

__device__ __forceinline__ void fm_mul_half4(float4& v, const half* p)
{
    float4 s = fm_load_half4(p);
    v.x *= s.x; v.y *= s.y; v.z *= s.z; v.w *= s.w;
}

__device__ __forceinline__ void fm_store_half4(half* p, const float4& v)
{
    half4 h(__floats2half2_rn(v.x, v.y), __floats2half2_rn(v.z, v.w));
    *reinterpret_cast<half4*>(p) = h;
}

__device__ __forceinline__ void fm_ds41_round_half4(float4& v)
{
    v.x = __half2float(__float2half_rn(v.x));
    v.y = __half2float(__float2half_rn(v.y));
    v.z = __half2float(__float2half_rn(v.z));
    v.w = __half2float(__float2half_rn(v.w));
}

// XOR swizzle of the 16-byte chunk column inside a 32-wide (64 B) A row so
// ldmatrix phases (8 consecutive rows, one chunk) hit 8 distinct bank groups.
__device__ __forceinline__ int fm_swz(int row, int chunk)
{
    return chunk ^ ((row >> 1) & 3);
}

__global__ __launch_bounds__(FM_THREADS)
void ds41_dual_gather_kernel(
    const half* __restrict__ x,
    const int64_t* __restrict__ row_token,
    const int* __restrict__ row_expert,
    const half* const* __restrict__ gate_suh_ptrs,
    const half* const* __restrict__ up_suh_ptrs,
    half* __restrict__ h13g,
    half* __restrict__ h13u,
    const int* __restrict__ num_rows_ptr,
    int size_k)
{
    const int num_rows = *num_rows_ptr;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int blk = blockIdx.y;             // 128-wide K block
    for (int row = blockIdx.x * FM_WARPS + warp; row < num_rows; row += gridDim.x * FM_WARPS)
    {
        const int64_t token = row_token[row];
        const int expert = row_expert[row];
        const half* gate_suh = gate_suh_ptrs[expert] + blk * 128;
        const half* up_suh = up_suh_ptrs[expert] + blk * 128;
        const half* src = x + token * (int64_t) size_k + blk * 128;
        half* dstg = h13g + (int64_t) row * size_k + blk * 128;
        half* dstu = h13u + (int64_t) row * size_k + blk * 128;
        // E2 boundary (had_hf_r_128_inner<pre_scale>): the input scale is a
        // fp16 x fp16 multiply, rounded, BEFORE the fp32 Hadamard.
        const half4 input = *reinterpret_cast<const half4*>(src + lane * 4);
        {
        half4 hv = input;
        half4 hs = *reinterpret_cast<const half4*>(gate_suh + lane * 4);
        hv.x = __hmul2(hv.x, hs.x);
        hv.y = __hmul2(hv.y, hs.y);
        float4 v = make_float4(__low2float(hv.x), __high2float(hv.x),
                               __low2float(hv.y), __high2float(hv.y));
        fm_had_row(v, lane);
        fm_store_half4(dstg + lane * 4, v);
        }
        {
        half4 hv = input;
        half4 hs = *reinterpret_cast<const half4*>(up_suh + lane * 4);
        hv.x = __hmul2(hv.x, hs.x);
        hv.y = __hmul2(hv.y, hs.y);
        float4 v = make_float4(__low2float(hv.x), __high2float(hv.x),
                               __low2float(hv.y), __high2float(hv.y));
        fm_had_row(v, lane);
        fm_store_half4(dstu + lane * 4, v);
        }
    }
}

} // namespace
extern "C" int ds41_dual_gather_abi() { return 1; }
extern "C" int ds41_dual_gather_info(int* info) {
    if (!info) return int(cudaErrorInvalidValue);
    cudaFuncAttributes attr;
    auto err = cudaFuncGetAttributes(&attr, ds41_dual_gather_kernel);
    if (err != cudaSuccess) return int(err);
    int blocks;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, ds41_dual_gather_kernel, FM_THREADS, 0);
    if (err != cudaSuccess) return int(err);
    info[0] = FM_THREADS; info[1] = attr.numRegs; info[2] = attr.localSizeBytes; info[3] = blocks;
    return 0;
}
extern "C" int ds41_dual_gather(void** pointers, int rows_bound, int width, void* raw_stream) {
    if (!pointers || rows_bound < 1 || rows_bound > 18432 || width != 5120) return int(cudaErrorInvalidValue);
    for (int i=0; i<8; ++i) if (!pointers[i]) return int(cudaErrorInvalidValue);
    int gx = (rows_bound + FM_WARPS - 1) / FM_WARPS;
    if (gx > 1024) gx = 1024;
    ds41_dual_gather_kernel<<<dim3(gx, width / 128), FM_THREADS, 0, (cudaStream_t)raw_stream>>>(
        (const half*)pointers[0], (const int64_t*)pointers[1], (const int*)pointers[2],
        (const half* const*)pointers[3], (const half* const*)pointers[4],
        (half*)pointers[5], (half*)pointers[6], (const int*)pointers[7], width);
    return int(cudaGetLastError());
}
