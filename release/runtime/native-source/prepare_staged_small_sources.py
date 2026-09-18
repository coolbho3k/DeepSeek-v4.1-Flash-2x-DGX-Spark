# SPDX-License-Identifier: AGPL-3.0-only
"""Add a 2..4-token expert pipeline while preserving the one-token source.

Each sorted assignment gets a separate scratch row. Its token index selects
the input/output row; the existing FP32 register decoder and all precision
epilogues are unchanged. No GPU allocation or cooperative launch is added.
"""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def prepare(build):
    from build_exl3_moe_mul1 import once
    build = Path(build)
    original = build / 'staged-launch.cu'
    text = original.read_text()
    text = text.replace('ds41_staged1_', 'ds41_small_staged_')
    text = text.replace('namespace ds41_staged', 'namespace ds41_small_staged')
    text = text.replace('ds41_staged::', 'ds41_small_staged::')
    text = once(text, '// One token only.', '// Two through four tokens.')
    text = once(text, 'int experts,int assignments) {',
                'int experts,int assignments,int slots) {')
    text = once(text, 'if(expert<TOP)meta[expert]=-1;',
                'if(expert<slots)meta[expert]=-1;')
    text = once(text, 'void gather(const half* x,half* g,half* u,const int64_t* meta,Tables t,int rows)',
                'void gather(const half* x,half* g,half* u,const int64_t* meta,const int64_t* tokens,Tables t,int rows)')
    text = once(text, '    const int off=chunk*128, dst=slot*rows*H+off;',
                '    const int off=chunk*128, dst=slot*rows*H+off;\n'
                '    x += tokens[slot]*H;')
    text = once(text, 'float* out,const int64_t* meta,Tables t,int rows) {',
                'float* out,const int64_t* meta,const int64_t* tokens,Tables t,int rows) {')
    text = once(text, 'out+off);', 'out+tokens[slot]*H+off);')
    text = text.replace('Tables t,int rows,cudaStream_t stream)',
                        'Tables t,int rows,int slots,cudaStream_t stream)')
    text = once(text, '1,UP?TOP*2:TOP)', '1,UP?slots*2:slots)')
    text = text.replace('(g,u,ig,iu,meta,t,rows,stream);',
                        '(g,u,ig,iu,meta,t,rows,slots,stream);')
    text = once(text, 'x.size(0)==1', '(x.size(0)>=2 && x.size(0)<=4)')
    text = once(text, '"Only one5120-wide token is supported"',
                '"Only two through four5120-wide tokens are supported"')
    text = once(text, 'tokens.numel()<=TOP', 'tokens.numel()<=TOP*x.size(0)')
    text = once(text, '"Invalid one-token routes"', '"Invalid small-batch routes"')
    text = once(text, 'meta.numel()==TOP', 'meta.numel()==TOP*x.size(0)')
    text = once(text, '"Six owned route slots required"', '"Six route slots per token required"')
    # All24 assignments fit inside the first24 scratch rows, while metadata
    # is aliased in the last original workspace row by the Python binding.
    text = once(text, 'm,experts,tokens.numel());',
                'm,experts,tokens.numel(),TOP*x.size(0));')
    text = once(text, 'g,u,m,tables,rows);',
                'g,u,m,tokens.data_ptr<int64_t>(),tables,1);')
    text = once(text, 'up_variant,g,u,ig,iu,m,tables,rows,stream);',
                'up_variant,g,u,ig,iu,m,tables,1,TOP*x.size(0),stream);')
    text = once(text, 'weights.data_ptr<float>(),tables,rows);',
                'weights.data_ptr<float>(),tables,1);')
    text = once(text, 'down_variant,g,u,ig,iu,m,tables,rows,stream);',
                'down_variant,g,u,ig,iu,m,tables,1,TOP*x.size(0),stream);')
    text = once(text, 'out.data_ptr<float>(),m,tables,rows);',
                'out.data_ptr<float>(),m,tokens.data_ptr<int64_t>(),tables,1);')
    text = text.replace('dim3(10,1,TOP)', 'dim3(10,1,TOP*x.size(0))')
    text = text.replace('dim3(3,1,TOP)', 'dim3(3,1,TOP*x.size(0))')
    (build / 'staged-small-launch.cu').write_text(text)
    bindings = build / 'bindings.cpp'
    text = once(bindings.read_text(), 'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {',
        '''void ds41_small_staged_forward(const at::Tensor&,const at::Tensor&,const at::Tensor&,
    const at::Tensor&,const at::Tensor&,const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,const at::Tensor&,int64_t,int64_t);
std::vector<int64_t> ds41_small_staged_resources();
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_staged_small",&ds41_small_staged_forward);
    m.def("staged_small_resources",&ds41_small_staged_resources);''')
    bindings.write_text(text)
    shutil.copyfile(Path(__file__), build / Path(__file__).name)
    return dict(enabled=False, rows=[2,3,4], maximum_assignments=24,
                original_one_token_source_preserved=True, extra_gpu_allocation_bytes=0,
                original_precision_epilogues_preserved=True, gpu_qualification_pending=True)
