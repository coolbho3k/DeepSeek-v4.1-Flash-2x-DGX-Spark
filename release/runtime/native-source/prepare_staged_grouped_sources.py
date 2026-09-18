# SPDX-License-Identifier: AGPL-3.0-only
"""Share EXL3 weight decoding across up to four same-expert assignments.

The existing small-batch source remains available as an independent baseline.
Only the active MMA rows and their FP32 reduction storage change. Hadamard,
SwiGLU, scales, accumulation order within each row and rounding stay intact.
"""
from pathlib import Path
import shutil


def prepare(build):
    from build_exl3_moe_mul1 import once
    build = Path(build)
    assert not (build / 'staged-grouped-launch.cu').exists()
    original = (build / 'staged_register_gemv.cuh').read_text()
    start = original.index('template<int WK,int WNT,int PF>')
    header = ('// SPDX-License-Identifier: AGPL-3.0-only\n'
              '// Reuses the retained MIT EXL3 register decoder.\n'
              '#pragma once\n#include "staged_register_gemv.cuh"\n\n'
              + original[start:])
    header = once(header, 'ds41_staged_gemv_inner(', 'ds41_grouped_gemv_inner(')
    header = once(header, 'const int expert_block,const int expert_blocks)',
                  'const int expert_block,const int expert_blocks,const int size_m)')
    header = once(header, 'CFG=0,MMODE=0,bits=3,cb=2', 'CFG=0,MMODE=1,bits=3,cb=2')
    header = once(header, '    constexpr int size_m=1;\n', '')
    header = once(header, 'ROWS = MMODE == 0 ? 1 : EXL3_GEMV_MAX_M', 'ROWS = 4')
    header = once(header, 'exactly WK*WNT*16 FP32 reduction slots',
                  'exactly WK*4*WNT*16 FP32 reduction slots')
    (build / 'staged_grouped_gemv.cuh').write_text(header)

    source = (build / 'staged-small-launch.cu').read_text()
    source = source.replace('ds41_small_staged', 'ds41_grouped_staged')
    source = once(source, '#include "staged_register_gemv.cuh"',
                  '#include "staged_grouped_gemv.cuh"')
    source = once(source,
        'half* g,half* u,half* ig,half* iu,const int64_t* meta,Tables t,int rows) {',
        'half* g,half* u,half* ig,half* iu,const int64_t* meta,Tables t,int rows,int slots) {')
    source = once(source, '    if(expert<0)return; // Uniform across the complete CTA.',
        '''    if(expert<0)return; // Uniform across the complete CTA.
    // The router emits contiguous equal-expert assignments. Partition each
    // run into groups of four, including duplicate routes within a token.
    // Every CTA agrees on the same leader and active row count.
    int begin=slot;
    while(begin>0 && meta[begin-1]==expert)--begin;
    if((slot-begin)%4!=0)return;
    int active_rows=1;
    while(active_rows<4 && slot+active_rows<slots
          && meta[slot+active_rows]==expert)++active_rows;''')
    source = once(source,
        'ds41_staged_gemv_inner<WK,WNT,PF>(x,w,y,UP?H:I,UP?I:H,blockIdx.x,gridDim.x);',
        'ds41_grouped_gemv_inner<WK,WNT,PF>(x,w,y,UP?H:I,UP?I:H,blockIdx.x,gridDim.x,active_rows);')
    assert source.count('WK*WNT*16*sizeof(float)') == 2
    source = source.replace('WK*WNT*16*sizeof(float)', 'WK*4*WNT*16*sizeof(float)')
    source = once(source, 'stream>>>(g,u,ig,iu,meta,t,rows);',
                  'stream>>>(g,u,ig,iu,meta,t,rows,slots);')
    (build / 'staged-grouped-launch.cu').write_text(source)
    bindings = build / 'bindings.cpp'
    text = once(bindings.read_text(), 'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {',
        '''void ds41_grouped_staged_forward(const at::Tensor&,const at::Tensor&,const at::Tensor&,
    const at::Tensor&,const at::Tensor&,const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,const at::Tensor&,int64_t,int64_t);
std::vector<int64_t> ds41_grouped_staged_resources();
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_staged_grouped_small",&ds41_grouped_staged_forward);
    m.def("staged_grouped_small_resources",&ds41_grouped_staged_resources);''')
    bindings.write_text(text)
    shutil.copyfile(Path(__file__), build / Path(__file__).name)
    return dict(enabled=False, rows=[2,3,4], maximum_group_rows=4,
                original_small_source_preserved=True, extra_gpu_allocation_bytes=0,
                original_precision_epilogues_preserved=True, gpu_qualification_pending=True)
