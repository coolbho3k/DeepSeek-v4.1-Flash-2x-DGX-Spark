// SPDX-License-Identifier: AGPL-3.0-only
// Local scheduler around MiaAI Lab / Wesley Young cooperative MoE tasks;
// ExLlamaV3 primitives by Turboderp. Retained licenses accompany this file.
// No whole-grid barrier: each resident block can always claim remaining A
// work, consume a published B task, or retire after all B tasks complete.

constexpr int PIPE_RUNS = 144;
constexpr int PIPE_A_NEXT = 0, PIPE_READY_TAIL = 1, PIPE_B_DONE = 2;
constexpr int PIPE_A_DONE = 16;
constexpr int PIPE_READY = PIPE_A_DONE + PIPE_RUNS;
constexpr int PIPE_B_NEXT = PIPE_READY + PIPE_RUNS;
static_assert(PIPE_B_NEXT + PIPE_RUNS <= 1296, "Pipeline exceeds existing A-counter scratch");
constexpr int persistent_smem_bytes()
{
    return smem_b_bytes<3>() > fused_a_smem_bytes() ? smem_b_bytes<3>() : fused_a_smem_bytes();
}

__device__ __forceinline__ cuda::atomic_ref<int, cuda::thread_scope_device> pipe_atomic(int* address)
{
    return cuda::atomic_ref<int, cuda::thread_scope_device>(*address);
}

__global__ __launch_bounds__(THREADS)
void exl3_moe_coop_persistent_kernel(const MoeCoopParams p_in)
{
    const MoeCoopParams p = goal50_fixed_params(p_in);
    extern __shared__ uint32_t smem_dyn[];
    __shared__ int task[3]; // kind: -1 retire, 0 retry, 1 gate/up, 2 down; run; column
    const int runs = p.runs[0];
    const int a_columns = p.I / 128;
    const int b_columns = p.Ho / 128;
    int cursor = blockIdx.x;
    unsigned idle_iterations = 0;
    for (;;)
    {
        if (threadIdx.x == 0)
        {
            task[0] = 0; task[1] = 0; task[2] = 0;
            // The tail reserves queue positions. The per-position release
            // store below publishes a fully ready run; zero means unpublished.
            const int ready = pipe_atomic(p.ctr_a + PIPE_READY_TAIL).load(cuda::memory_order_acquire);
            for (int offset = 0; offset < ready; ++offset)
            {
                const int slot = (cursor + offset) % ready;
                const int encoded = pipe_atomic(p.ctr_a + PIPE_READY + slot).load(cuda::memory_order_acquire);
                if (!encoded) continue;
                const int run = encoded - 1;
                auto next = pipe_atomic(p.ctr_a + PIPE_B_NEXT + run);
                if (next.load(cuda::memory_order_relaxed) >= b_columns) continue;
                const int column = next.fetch_add(1, cuda::memory_order_relaxed);
                if (column >= b_columns) continue;
                task[0] = 2; task[1] = run; task[2] = column;
                cursor = slot + 1;
                break;
            }
            if (!task[0])
            {
                auto next = pipe_atomic(p.ctr_a + PIPE_A_NEXT);
                if (next.load(cuda::memory_order_relaxed) < runs * a_columns)
                {
                    const int item = next.fetch_add(1, cuda::memory_order_relaxed);
                    if (item < runs * a_columns)
                    {
                        task[0] = 1; task[1] = item / a_columns; task[2] = item % a_columns;
                    }
                }
            }
            if (!task[0] && pipe_atomic(p.ctr_a + PIPE_B_DONE).load(cuda::memory_order_acquire) == runs * b_columns)
                task[0] = -1;
        }
        __syncthreads();
        const int kind = task[0], run = task[1], column = task[2];
        if (kind < 0) return;
        if (!kind)
        {
            // Fail closed instead of silently hanging a faulty experimental
            // scheduler. This is a very generous no-work retry bound, not a
            // serving timeout or a relaxation of memory safety limits.
            if (++idle_iterations > (1u << 20)) asm volatile("trap;");
            __nanosleep(128);
            // The broadcaster must not overwrite the task until every warp
            // has consumed this iteration's shared descriptor, even on retry.
            __syncthreads();
            continue;
        }
        idle_iterations = 0;
        if (kind == 1)
        {
            persistent_a_task(p, run, column, smem_dyn);
            // All activation writers publish their stores before thread zero
            // participates in the release/acquire completion chain for a run.
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0)
            {
                const int before = pipe_atomic(p.ctr_a + PIPE_A_DONE + run).fetch_add(1, cuda::memory_order_acq_rel);
                if (before == a_columns - 1)
                {
                    const int slot = pipe_atomic(p.ctr_a + PIPE_READY_TAIL).fetch_add(1, cuda::memory_order_relaxed);
                    pipe_atomic(p.ctr_a + PIPE_READY + slot).store(run + 1, cuda::memory_order_release);
                }
            }
        }
        else
        {
            persistent_b_task<3, 2, true>(p, run, column, smem_dyn);
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0)
                pipe_atomic(p.ctr_a + PIPE_B_DONE).fetch_add(1, cuda::memory_order_release);
        }
        // No block may reuse its shared task/scratch before all its members
        // finish the selected operation. This is block-local, not grid-wide.
        __syncthreads();
    }
}
