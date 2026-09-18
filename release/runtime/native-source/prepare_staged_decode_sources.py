# SPDX-License-Identifier: AGPL-3.0-only
"""Add an unselected non-cooperative one-token expert pipeline.

Reuse the already numerically qualified FP32 register-decoder source and
the original DS41 Hadamard/SwiGLU/output epilogues. Original forward and
its resource/ownership contract are untouched. Generated source and original
MIT notices accompany the additive AGPL build.
"""
import hashlib
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
HEADER='artifacts/exl3-moe-combined-build-v5/gemv-include/quant/ds41_register_gemv.cuh'
HEADER_SHA='684279f5aafa377622ffe6526b28029020b3c4e41691c24fae04f6ca82c71b03'


def prepare(build):
    from build_exl3_moe_mul1 import once
    build=Path(build)
    assert not (build/'staged-launch.cu').exists()
    raw=(ROOT/HEADER).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==HEADER_SHA
    text=raw.decode()
    text=once(text,'#include "exl3_gemv_kernel.cuh"','#include "quant/exl3_gemv_kernel.cuh"')
    text=once(text,'__device__ __forceinline__ void ds41_register_gemv_inner(',
        'template<int WK,int WNT,int PF>\n__device__ __forceinline__ void ds41_staged_gemv_inner(')
    for line in (
        '    constexpr int WK   = CFG == 0 ? 16 : 8;     // k-split (warps per block)\n',
        '    constexpr int WNT  = CFG == 0 ? 2 : 4;      // adjacent n-tiles per warp\n',
        '    constexpr int PF   = CFG == 0 ? 4 : 2;      // prefetch ring depth\n',
        '    constexpr int FOLD = CFG == 0 ? 4 : 2;      // fp16->fp32 fold cadence (divides PF)\n',
    ):text=once(text,line,'')
    text=once(text,"    // Reuse the original cooperative launch's90KiB dynamic scratch.\n    // Only2KiB is needed here; the original GEMM fallback is unchanged.",
        '    // Independently launched CTA: exactly WK*WNT*16 FP32 reduction slots.')
    assert 'grid.sync' not in text and 'FragC_h ch' not in text and 'mma_ab_h(' not in text
    (build/'staged_register_gemv.cuh').write_text(text)
    shutil.copyfile(ROOT/'kernels/exl3_moe_staged1/staged.cu',build/'staged-launch.cu')
    bindings=build/'bindings.cpp'
    text=once(bindings.read_text(),'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {',
        '''void ds41_staged1_forward(const at::Tensor&,const at::Tensor&,const at::Tensor&,
    const at::Tensor&,const at::Tensor&,const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,const at::Tensor&,int64_t,int64_t);
std::vector<int64_t> ds41_staged1_resources();
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_staged1",&ds41_staged1_forward);
    m.def("staged1_resources",&ds41_staged1_resources);''')
    bindings.write_text(text)
    shutil.copyfile(Path(__file__),build/Path(__file__).name)
    return dict(enabled=False,one_token_only=True,original_forward_preserved=True,
        source_register_header_sha256=HEADER_SHA,cooperative_launch=False,
        fp32_mma_accumulation_preserved=True,original_epilogues_preserved=True,
        all_scratch_caller_owned=True,variants=[[4,2,4],[8,2,4],[8,4,2],[16,2,4]],
        parallel_route_prefix=True,
        numerical_and_gpu_qualification_pending=True)
