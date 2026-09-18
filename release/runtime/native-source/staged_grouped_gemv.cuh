// SPDX-License-Identifier: AGPL-3.0-only
// Reuses the retained MIT EXL3 register decoder.
#pragma once
#include "staged_register_gemv.cuh"

template<int WK,int WNT,int PF>
__device__ __forceinline__ void ds41_grouped_gemv_inner(
    const half* __restrict__ A,const uint16_t* __restrict__ B,
    half* __restrict__ C,const int size_k,const int size_n,
    const int expert_block,const int expert_blocks,const int size_m)
{
    constexpr int CFG=0,MMODE=1,bits=3,cb=2;
    constexpr bool SMEM_STAGE=false,c_fp32=false;
    constexpr int THREADS = WK * 32;
    constexpr int ROWS = 4;
    constexpr int COLS = WNT * 16;

    constexpr int TWORDS = 8 * bits;                        // uint32 per 16x16 tile
    constexpr int LOADS = bits == 2 ? WNT / 2 : WNT;        // warp loads per k-slice
    constexpr int LSTRIDE = bits == 3 ? 24 : 32;            // uint32 per load
    static_assert(bits != 2 || WNT % 2 == 0, "2 bpw packs two tiles per warp load");

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int ntiles = size_n / 16;
    const int kslices = size_k / 16;
    const int num_groups = size_n / COLS;

    const int chunk = CEIL_DIVIDE(kslices, WK);
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));

    const uint32_t* B32 = (const uint32_t*) B;
    const size_t slice_stride = (size_t) ntiles * TWORDS;   // uint32 per k-slice row
    const half2* A2 = (const half2*) A;
    const half2 hzero = __half2half2(__ushort_as_half(0));

    // A fragment row indices for this lane
    const int r0 = lane >> 2;
    const size_t a_row0 = (size_t) r0 * (size_k / 2);
    const bool r0_ok = MMODE == 0 ? lane < 4 : r0 < size_m;

    // Per-lane extraction constants (see dq8_aligned_2bits / dq8<3, cb, 4> in exl3_dq.cuh)
    [[maybe_unused]] int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    if constexpr (bits == 2)
    {
        int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    }
    if constexpr (bits == 3)
    {
        int t_offset = lane << 3;
        int b1 = (t_offset + 257) * 3;
        int b2 = b1 + 21;
        int i0 = (b1 - 16) / 32;
        int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }

    // Independently launched CTA: exactly WK*4*WNT*16 FP32 reduction slots.
    extern __shared__ float ds41_gemv_scratch[];
    auto& sh_red=*reinterpret_cast<float (*)[WK][ROWS][COLS]>(ds41_gemv_scratch);
    [[maybe_unused]] __shared__ uint32_t sh_stage[SMEM_STAGE ? WK : 1][SMEM_STAGE ? LOADS * LSTRIDE : 1];

    for (int group = expert_block; group < num_groups; group += expert_blocks)
    {
        const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;

        // Prefetch ring (indices must be compile-time or pf lands in local memory)
        auto ld_b = [&] (int i, int l) -> uint32_t
        {
            if constexpr (bits == 3)
                return lane < 24 ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
            else
                return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
        };

        uint32_t pf[PF][LOADS];
        #pragma unroll
        for (int d = 0; d < PF; ++d)
            if (d < myn)
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(d, l);

        FragC ch[WNT][2] = {};


        for (int ib = 0; ib < myn; ib += PF)
        {
        #pragma unroll
        for (int d = 0; d < PF; ++d)
        {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                bw[l] = pf[d][l];

            if (i + PF < myn)
            {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(i + PF, l);
            }

            if constexpr (SMEM_STAGE)
            {
                __syncwarp();
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    if (bits != 3 || lane < 24)
                        sh_stage[warp][l * LSTRIDE + lane] = bw[l];
                __syncwarp();
            }

            // A fragment: lane covers row lane/4, k pairs (2(lane%4), +1) and (+8, +9)
            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
            a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;

            #pragma unroll
            for (int t = 0; t < WNT; ++t)
            {
                FragB f0, f1;
                if constexpr (SMEM_STAGE)
                {
                    const uint32_t* tp = &sh_stage[warp][t * TWORDS];
                    if constexpr (bits == 4)
                        exl3_gemv_ns::dq8_regs_4bits<cb>(tp[(lane + 31) & 31], tp[lane], f0, f1);
                    else if constexpr (bits == 2)
                        exl3_gemv_ns::dq8_regs_2bits<cb>(tp[x_src_a], tp[x_src_b], lane << 3, f0, f1);
                    else
                        exl3_gemv_ns::dq8_regs_3bits<cb>(tp[x_src_a], tp[x_src_b], x_s2, f0, f1);
                }
                else if constexpr (bits == 4)
                {
                    uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                    exl3_gemv_ns::dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
                }
                else if constexpr (bits == 2)
                {
                    // Two tiles per loaded word group: tile t lives in lanes (t&1)*16 .. +15
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
                }
                else  // bits == 3
                {
                    uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
                }

                ds41_mma_ab_f(a01, a23, f0, ch[t][0]);
                ds41_mma_ab_f(a01, a23, f1, ch[t][1]);
            }


        }
        }

        // Cross-warp reduction over the k splits. Lane l holds row l/4, cols
        // tile*16 + frag*8 + 2*(l%4) (+1)
        {
            const int c0 = 2 * (lane & 3);
            const bool store0 = MMODE == 0 ? lane < 4 : r0 < ROWS;
            const int sr0 = MMODE == 0 ? 0 : r0;
            if (store0)
            {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f)
                    {
                        const int col = t * 16 + f * 8 + c0;
                        sh_red[warp][sr0][col + 0] = ch[t][f][0];
                        sh_red[warp][sr0][col + 1] = ch[t][f][1];
                    }
            }
        }
        __syncthreads();

        const int rows_out = MMODE == 0 ? 1 : min(size_m, ROWS);
        for (int idx = threadIdx.x; idx < COLS * rows_out; idx += THREADS)
        {
            const int r = idx / COLS;
            const int c = idx % COLS;
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < WK; ++j)
                sum += sh_red[j][r][c];
            const int col = group * COLS + c;
            if constexpr (c_fp32) ((float*) C)[(size_t) r * size_n + col] = sum;
            else                  ((half*)  C)[(size_t) r * size_n + col] = __float2half_rn(sum);
        }
        __syncthreads();
    }

}
