// SPDX-License-Identifier: AGPL-3.0-only
// Derived from MiaAI Lab / Wesley Young cooperative MoE and Turboderp's
// ExLlamaV3 primitives. Retained licenses accompany the prepared sources.
// Local experiment: two 512-thread logical groups inside one 1024-thread
// block run gate/up concurrently, with block-local FP16 projection outputs.

constexpr int fused_a_smem_bytes()
{
    return 2 * (smem_red_bytes() + smem_stage_bytes<3>()) + 2 * ROWS * 128 * sizeof(half);
}

__global__ __launch_bounds__(2 * THREADS)
void exl3_moe_coop_fused_a_kernel(const MoeCoopParams p_in)
{
    const MoeCoopParams p = goal50_fixed_params(p_in);
    extern __shared__ uint32_t smem_dyn[];
    const int projection = threadIdx.x / THREADS;
    const int slice_words = (smem_red_bytes() + smem_stage_bytes<3>()) / sizeof(uint32_t);
    float* sh_red = reinterpret_cast<float*>(smem_dyn + projection * slice_words);
    uint32_t* sh_stage = smem_dyn + projection * slice_words + smem_red_bytes() / sizeof(uint32_t);
    half* gate = reinterpret_cast<half*>(smem_dyn + 2 * slice_words);
    half* up = gate + ROWS * 128;
    __shared__ int sh_res[4];
    __shared__ int sh_rows[ROWS];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int ng = p.I / 128;
    for (int i = blockIdx.x * (2 * THREADS) + threadIdx.x; i < p.ctr_b_len; i += gridDim.x * (2 * THREADS))
        p.ctr_b[i] = 0;
    const int run_idx = blockIdx.x / ng;
    const int group = blockIdx.x % ng;
    int nrows = 0;
    if (!read_run(p, run_idx, sh_rows, nrows, sh_res)) return;
    const int local = slot_info(p, sh_rows[0]).local;
    const bool is_gate = projection == 0;

    // All 1024 threads take the same helper call and every block barrier.
    // The two logical groups have identical K extent and live-row count,
    // disjoint reduction scratch, and the parent's exact 16-warp geometry.
    gemv_tile<3, 2, true, true>(
        reinterpret_cast<const uint32_t*>((is_gate ? p.g_trellis : p.u_trellis)[local]),
        reinterpret_cast<const half2*>(is_gate ? p.had_g : p.had_u), (size_t)p.Hi / 2,
        sh_rows, nrows, is_gate ? gate : up, 128, false, 0, p.Hi / 16,
        p.I / 16, group, sh_red, sh_stage);

    if (warp < nrows)
    {
        const int s = sh_rows[warp];
        const int col = group * 128 + lane * 4;
        const size_t off = (size_t)s * p.I + col;
        float u0, u1, u2, u3;
        load_h4(up + warp * 128 + lane * 4, u0, u1, u2, u3);
        had128(u0, u1, u2, u3, lane);
        scale_h4(((const half*)p.u_svh[local]) + col, u0, u1, u2, u3);
        float g0, g1, g2, g3;
        load_h4(gate + warp * 128 + lane * 4, g0, g1, g2, g3);
        had128(g0, g1, g2, g3, lane);
        scale_h4(((const half*)p.g_svh[local]) + col, g0, g1, g2, g3);
        float a0 = act_gate(p.act, p.gated, g0, u0, p.act_limit);
        float a1 = act_gate(p.act, p.gated, g1, u1, p.act_limit);
        float a2 = act_gate(p.act, p.gated, g2, u2, p.act_limit);
        float a3 = act_gate(p.act, p.gated, g3, u3, p.act_limit);
        scale_h4(((const half*)p.d_suh[local]) + col, a0, a1, a2, a3);
        had128(a0, a1, a2, a3, lane);
        store_h4(p.act_out + off, a0, a1, a2, a3);
    }
}
