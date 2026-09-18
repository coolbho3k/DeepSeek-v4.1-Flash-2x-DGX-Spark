#pragma once
// DS4.1-specific epilogues, used with the pinned EXL3 MUL1 GEMM core.
// Preserve FP32 SwiGLU and route weighting BEFORE the down projection.
// Each warp owns one complete 128-value row fragment; blockIdx.y must be 0.

__device__ __forceinline__ void ds41_guad(
    half* gate, half* up, const half* gate_scale, const half* up_scale,
    const half* down_scale, const float route_weight)
{
    constexpr float norm = 0.088388347648f;
    had_hf_r_128_inner<false, true>(gate, gate, gate_scale, norm);
    had_hf_r_128_inner<false, true>(up, up, up_scale, norm);
    int lane = threadIdx.x & 31;
    #pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        const int i = lane * 4 + j;
        float g = fminf(__half2float(gate[i]), 10.0f);
        float u = fmaxf(-10.0f, fminf(__half2float(up[i]), 10.0f));
        float activated = g / (1.0f + expf(-g));
        gate[i] = __float2half_rn((activated * u) * route_weight);
    }
    had_hf_r_128_inner<true, false>(gate, gate, down_scale, norm);
}

__device__ __forceinline__ void ds41_down_out(
    const half* down, half* scratch, const half* scale, float* output)
{
    // Match the existing FP16 expert output boundary before FP32 accumulation.
    had_hf_r_128_inner<false, true>(down, scratch, scale, 0.088388347648f);
    int lane = threadIdx.x & 31;
    #pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        const int i = lane * 4 + j;
        atomicAdd(output + i, __half2float(scratch[i]));
    }
}
