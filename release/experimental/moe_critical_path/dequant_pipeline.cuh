// SPDX-License-Identifier: AGPL-3.0-only
// Local software-pipeline experiment in MiaAI Lab / Wesley Young cooperative
// MoE; exact EXL3 MUL1 and MMA primitives by Turboderp. Original notices retained.
// Included inside gemv_tile, only for its register-dequantized 3-bit branch.
// The existing PF=4 compressed-weight ring is retained. Two decoded register
// banks additionally move the next K tile's decode ahead of this tile's MMA.
static_assert(PF == 4 && FOLD == 4, "Preserve the qualified fold schedule");
FragB decoded[2][WNT][2];
if (myn > 0)
{
    #pragma unroll
    for (int t = 0; t < WNT; ++t)
    {
        const uint32_t awv = __shfl_sync(0xffffffffu, pf[0][t], x_src_a);
        const uint32_t bwv = __shfl_sync(0xffffffffu, pf[0][t], x_src_b);
        exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, decoded[0][t][0], decoded[0][t][1]);
    }
}
for (int ib = 0; ib < myn; ib += PF)
{
    #pragma unroll
    for (int d = 0; d < PF; ++d)
    {
        const int i = ib + d;
        if (i >= myn) break;
        if (i + PF < myn)
        {
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                pf[d][l] = ld_b(i + PF, l);
        }
        const size_t a_col = (size_t)(ks0 + i) * 8 + (lane & 3);
        FragB a01, a23;
        a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
        a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
        a01[1] = hzero;
        a23[1] = hzero;

        #pragma unroll
        for (int t = 0; t < WNT; ++t)
        {
            if (i + 1 < myn)
            {
                const uint32_t awv = __shfl_sync(0xffffffffu, pf[(d + 1) % PF][t], x_src_a);
                const uint32_t bwv = __shfl_sync(0xffffffffu, pf[(d + 1) % PF][t], x_src_b);
                exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2,
                    decoded[(d + 1) % 2][t][0], decoded[(d + 1) % 2][t][1]);
                // Compiler-only dependency: materialize the next fragments
                // before consuming the current ones. No GPU instruction,
                // new arithmetic, memory fence, or cross-warp synchronization.
                uint32_t* current = reinterpret_cast<uint32_t*>(&decoded[d % 2][t][0]);
                const uint32_t* next = reinterpret_cast<const uint32_t*>(&decoded[(d + 1) % 2][t][0]);
                asm volatile("" : "+r"(current[0]), "+r"(current[1]), "+r"(current[2]), "+r"(current[3])
                    : "r"(next[0]), "r"(next[1]), "r"(next[2]), "r"(next[3]));
            }
            exl3_gemv_ns::mma_ab_h(a01, a23, decoded[d % 2][t][0], ch[t][0]);
            exl3_gemv_ns::mma_ab_h(a01, a23, decoded[d % 2][t][1], ch[t][1]);
        }
        if ((d + 1) % FOLD == 0 || i + 1 == myn)
        {
            #pragma unroll
            for (int t = 0; t < WNT; ++t)
                #pragma unroll
                for (int f = 0; f < 2; ++f)
                {
                    acc0[t][f].x += __low2float(ch[t][f][0]);
                    acc0[t][f].y += __high2float(ch[t][f][0]);
                    ch[t][f][0] = hzero;
                }
        }
    }
}
