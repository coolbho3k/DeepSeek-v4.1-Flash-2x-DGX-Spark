# SPDX-License-Identifier: AGPL-3.0-only
"""Derive a bounded persistent pipeline from the qualified native baseline.

Uses experiment one's exact serial gate/up helper, with the original down
arithmetic; neither prior fusion variant is enabled globally in serving.
"""
from pathlib import Path
from fused_gateup import once,transform as fused_transform,fragment as fused_fragment


def transform(wrapper,kernel,resident_blocks=1):
    if resident_blocks not in (1,2):raise ValueError('Unsupported occupancy experiment')
    wrapper,kernel=fused_transform(wrapper,kernel)
    w=wrapper.decode();k=kernel.decode();a=fused_fragment().decode()
    a=once(a,'__global__ __launch_bounds__(THREADS)\nvoid exl3_moe_coop_fused_a_kernel(const MoeCoopParams p_in)',
        '__device__ __forceinline__ void persistent_a_task(const MoeCoopParams p_in, int run_idx, int group, uint32_t* smem_dyn)')
    a=once(a,'    extern __shared__ uint32_t smem_dyn[];\n','')
    start=a.index('    const int ng = p.I / 128;')
    end=a.index('    int nrows = 0;',start)
    a=a[:start]+a[end:]
    # Copy the original B body without changing its arithmetic. Omit its
    # next-call A-counter reset, since that array now owns scheduler state.
    start=k.index('template <int bits, int cb, bool WIDE>\n__global__ __launch_bounds__(THREADS)\nvoid exl3_moe_coop_b_kernel')
    end=k.index('\n}  // namespace goal50_fixed_coop_ns',start)
    body=k[start:end]
    body=once(body,'__global__ __launch_bounds__(THREADS)\nvoid exl3_moe_coop_b_kernel(const MoeCoopParams p_in)',
        '__device__ __forceinline__ void persistent_b_task(const MoeCoopParams p_in, int run_idx, int group, uint32_t* smem_dyn)')
    body=once(body,'    extern __shared__ uint32_t smem_dyn[];\n','')
    old_start=body.index('    // Reset the gate/up stage\'s counters for the next call')
    old_end=body.index('    int nrows = 0;',old_start)
    body=body[:old_start]+'    const int ks = 0;\n'+body[old_end:]
    k=k[:end]+'\n'+body+'\n#include "persistent_pipeline.cuh"\n'+k[end:]
    w=once(w,'#include <cuda_runtime.h>','#include <cuda_runtime.h>\n#include <cuda/atomic>')
    w=once(w,'static bool prepared[3][3] = {};','static bool prepared[3][3] = {};\nstatic int persistent_grid = 0;')
    w=once(w,'goal50_coop_experiment() { return 101; }','goal50_coop_experiment() { return 201; }')
    w=w.replace('ns::exl3_moe_coop_fused_a_kernel','ns::exl3_moe_coop_persistent_kernel')
    w=w.replace('ns::fused_a_smem_bytes()','ns::persistent_smem_bytes()')
    w=once(w,'    prepared[bits-2][geometry]=true;',
        '    if(bits==3 && geometry==1) persistent_grid=prop.multiProcessorCount*info[4];\n    prepared[bits-2][geometry]=true;')
    w=once(w,'    int ga=rows*TOPK*(fuse_a?1:2)*(I/(s.wa?128:MOE_COOP_COLS));',
        '    if(fuse_a) {\n        if(persistent_grid<1) return int(cudaErrorInvalidConfiguration);\n'
        '        ns::exl3_moe_coop_persistent_kernel<<<persistent_grid,MOE_COOP_THREADS,ns::persistent_smem_bytes(),stream>>>(p);\n'
        '        return int(cudaGetLastError());\n    }\n'
        '    int ga=rows*TOPK*2*(I/(s.wa?128:MOE_COOP_COLS));')
    queue=Path(__file__).with_suffix('.cuh').read_text()
    if resident_blocks==2:
        queue=once(queue,'__launch_bounds__(THREADS)','__launch_bounds__(THREADS, 2)')
    return w.encode(),k.encode(),a.encode(),queue.encode()
