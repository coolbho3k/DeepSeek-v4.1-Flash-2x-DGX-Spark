# SPDX-License-Identifier: AGPL-3.0-only
"""Pure, pinned source transforms for the first native decode experiment."""
import hashlib
from pathlib import Path

WRAPPER_SHA='8b94fe324029dceea5cbd1685ec02ad32d3839a5a975e1a1601acc8ba877f10f'
KERNEL_SHA='7b6eaf25dd48d77a22a6a7d7e7c12ddde0d3b8ae81a5fb85c36c714e58ba97f4'


def once(text,before,after):
    if text.count(before)!=1:raise ValueError('Ambiguous native source anchor: '+before[:80])
    return text.replace(before,after)


def transform(wrapper,kernel,parallel=False):
    if hashlib.sha256(wrapper).hexdigest()!=WRAPPER_SHA or hashlib.sha256(kernel).hexdigest()!=KERNEL_SHA:
        raise ValueError('Changed qualified cooperative parent')
    w=wrapper.decode();k=kernel.decode()
    k=once(k,'template <int bits, int cb, bool WIDE>\n__device__ __forceinline__ void gemv_tile',
        'template <int bits, int cb, bool WIDE, bool LOCAL_C = false>\n__device__ __forceinline__ void gemv_tile')
    k=once(k,'const size_t idx = (size_t) rows[r] * c_stride + group * TCOLS + c;',
        'const size_t idx = LOCAL_C ? (size_t)r * TCOLS + c : (size_t)rows[r] * c_stride + group * TCOLS + c;')
    anchor='// Kernel B: down GEMV per (expert run, group);'
    k=once(k,anchor,'#include "fused_gateup.cuh"\n\n'+anchor)
    w=once(w,'extern "C" int goal50_coop_abi() { return 2; }',
        'extern "C" int goal50_coop_abi() { return 2; }\nextern "C" int goal50_coop_experiment() { return 101; }')
    w=once(w,'Selected s=select(bits,geometry);\n    void* funcs[3]',
        'Selected s=select(bits,geometry);\n    if(bits==3 && geometry==1) {\n        s.a=(void*)ns::exl3_moe_coop_fused_a_kernel; s.sa=ns::fused_a_smem_bytes();\n    }\n    void* funcs[3]')
    w=once(w,'Selected s=select(bits,geometry);\n    cudaStream_t stream=',
        'Selected s=select(bits,geometry);\n    const bool fuse_a=bits==3 && geometry==1 && rows>1;\n    if(fuse_a) {\n        s.a=(void*)ns::exl3_moe_coop_fused_a_kernel; s.sa=ns::fused_a_smem_bytes();\n    }\n    cudaStream_t stream=')
    w=once(w,'int ga=rows*TOPK*2*(I/(s.wa?128:MOE_COOP_COLS));',
        'int ga=rows*TOPK*(fuse_a?1:2)*(I/(s.wa?128:MOE_COOP_COLS));')
    if parallel:
        begin=k.index('__device__ __forceinline__ void gemv_tile')
        end=k.index('// Completion counter:',begin)
        body=k[begin:end]
        body=once(body,'const int warp = threadIdx.x / 32;',
            'const int logical_thread = LOCAL_C ? threadIdx.x % THREADS : threadIdx.x;\n    const int warp = logical_thread / 32;')
        body=once(body,'for (int o = threadIdx.x; o < TCOLS * nrows; o += THREADS)',
            'for (int o = logical_thread; o < TCOLS * nrows; o += THREADS)')
        k=k[:begin]+body+k[end:]
        w=once(w,'goal50_coop_experiment() { return 101; }','goal50_coop_experiment() { return 102; }')
        w=once(w,'int smems[3]={s.sa,s.sb,0};',
            'int smems[3]={s.sa,s.sb,0};\n    int threads[3]={(bits==3 && geometry==1)?1024:MOE_COOP_THREADS,MOE_COOP_THREADS,MOE_COOP_THREADS};')
        w=once(w,'cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,funcs[n],MOE_COOP_THREADS,smems[n])',
            'cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,funcs[n],threads[n],smems[n])')
        w=once(w,'info[n*5+0]=MOE_COOP_THREADS;','info[n*5+0]=threads[n];')
        w=once(w,'cudaLaunchKernel(s.a,dim3(ga),dim3(MOE_COOP_THREADS),args,s.sa,stream)',
            'cudaLaunchKernel(s.a,dim3(ga),dim3(fuse_a?1024:MOE_COOP_THREADS),args,s.sa,stream)')
    prefix='// DS41 experimental fused gate/up handoff; not serving-qualified.\n'
    return (prefix+w).encode(),(prefix+k).encode()


def fragment(parallel=False):
    name='fused_gateup_parallel.cuh' if parallel else 'fused_gateup.cuh'
    return Path(__file__).with_name(name).read_bytes()
