// SPDX-License-Identifier: AGPL-3.0-only
// Derived from MiaAI Lab / Wesley Young cooperative MoE, with ExLlamaV3
// primitives by Turboderp. Original notices accompany the prepared source.
// Local experiment: co-locate wide gate/up tiles and their activation;
// preserve the parent GEMV arithmetic, input rotations and down kernel.
// Included INSIDE goal50_fixed_coop_ns after the original A kernel.

constexpr int fused_a_smem_bytes()
{
    return smem_red_bytes() + smem_stage_bytes<3>() + 2 * ROWS * 128 * sizeof(half);
}

__global__ __launch_bounds__(THREADS)
void exl3_moe_coop_fused_a_kernel(const MoeCoopParams p_in)
{
    const MoeCoopParams p = goal50_fixed_params(p_in);
    extern __shared__ uint32_t smem_dyn[];
    float* sh_red = reinterpret_cast<float*>(smem_dyn);
    uint32_t* sh_stage = smem_dyn + smem_red_bytes() / sizeof(uint32_t);
    half* gate = reinterpret_cast<half*>(sh_stage + smem_stage_bytes<3>() / sizeof(uint32_t));
    half* up = gate + ROWS * 128;
    __shared__ int sh_res[4];
    __shared__ int sh_rows[ROWS];

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    // Host selects this kernel ONLY for the fixed 3-bit, geometry-1,
    // multi-row contract. The unchanged rotation kernel built the run table
    // and initialized empty-token outputs on this same stream.
    const int ng = p.I / 128;
    for (int i = blockIdx.x * THREADS + threadIdx.x; i < p.ctr_b_len; i += gridDim.x * THREADS)
        p.ctr_b[i] = 0;
    const int run_idx = blockIdx.x / ng;
    const int group = blockIdx.x % ng;
    int nrows = 0;
    if (!read_run(p, run_idx, sh_rows, nrows, sh_res)) return;
    const int local = slot_info(p, sh_rows[0]).local;

    // LOCAL_C changes only the store address: [local run row, local column].
    // Each call preserves WKK=4, prefetch/fold cadence, FP16 accumulators,
    // ordered FP32 fold/reduction and the final FP16 projection boundary.
    gemv_tile<3, 2, true, true>(
        reinterpret_cast<const uint32_t*>(p.g_trellis[local]),
        reinterpret_cast<const half2*>(p.had_g), (size_t)p.Hi / 2,
        sh_rows, nrows, gate, 128, false, 0, p.Hi / 16,
        p.I / 16, group, sh_red, sh_stage);
    gemv_tile<3, 2, true, true>(
        reinterpret_cast<const uint32_t*>(p.u_trellis[local]),
        reinterpret_cast<const half2*>(p.had_u), (size_t)p.Hi / 2,
        sh_rows, nrows, up, 128, false, 0, p.Hi / 16,
        p.I / 16, group, sh_red, sh_stage);

    // gemv_tile ends with a block barrier. One warp per live row now owns
    // both projections; no inter-block gate/up completion counter is needed.
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
