# SPDX-License-Identifier: AGPL-3.0-only
# Derives grouped expert kernels from MiaAI-Lab; retained notices and source
# are under vendor/miaai-dsv41-agpl. Generated outputs have the same license.
"""Generate, but do not compile or activate, DS41-compatible grouped kernels.

The original vendored source stays immutable. Explicitly preserve the current
FP16 GEMM/output boundaries, FP32 SiLU, and routing BEFORE down-input rounding.
The generated extension is additive, K3/MUL1-only, and not serving-qualified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/'vendor/miaai-dsv41-agpl'
PINS={
    'overlay/e3v2/exl3_fat_moe.cu':'5108f1fbf3ba0af259273798dc0eeba99a98d188e0a3cb09440474fb9b43f504',
    'overlay/e3v2/exl3_fat_moe.cuh':'bcb86658a06c010522c1a5e7dcecfddeb7a8a510c0c29e643b77dbafe46f7ac2'}


def sha(raw):return hashlib.sha256(raw).hexdigest()


def once(source,before,after):
    assert source.count(before)==1,('Changed upstream anchor',before)
    return source.replace(before,after,1)


def adapt(source,header):
    note='// DS41 adaptation: K3/MUL1; FP16 GEMM/output boundaries; FP32 pre-down routing.\n'
    marker='// Local change: this provenance/license prefix only; original body follows.\n'
    source=once(source,marker,note)
    header=once(header,marker,note)
    helper='''__device__ __forceinline__ void fm_ds41_round_half4(float4& v)
{
    v.x = __half2float(__float2half_rn(v.x));
    v.y = __half2float(__float2half_rn(v.y));
    v.z = __half2float(__float2half_rn(v.z));
    v.w = __half2float(__float2half_rn(v.w));
}

'''
    source=once(source,'// XOR swizzle of the 16-byte chunk column',helper+'// XOR swizzle of the 16-byte chunk column')
    source=once(source,'    half* __restrict__ h2,\n','    half* __restrict__ h2,\n    const float* __restrict__ row_weight,\n')
    source=once(source,'    at::Tensor& h2,\n','    at::Tensor& h2,\n    const at::Tensor& row_weight,\n')
    source=once(source,'    at::Tensor h2,\n    at::Tensor seg_expert,',
        '    at::Tensor h2,\n    at::Tensor row_weight,\n    at::Tensor seg_expert,')
    source=once(source,'        reinterpret_cast<half*>(h2.data_ptr()),\n',
        '        reinterpret_cast<half*>(h2.data_ptr()),\n        reinterpret_cast<const float*>(row_weight.data_ptr()),\n')
    header=once(header,'    at::Tensor h2,           // [rows_cap, N] half (out)\n',
        '    at::Tensor h2,           // [rows_cap, N] half (out)\n    at::Tensor row_weight,   // [rows_cap] float, applied before down-input FP16 rounding\n')
    for value,scale in (('g','svh_g'),('u','svh_u')):
        old=f'                    fm_had_row({value}, lane);\n                    fm_mul_half4({value}, {scale} + lane * 4);'
        middle=old.replace(f'                    fm_mul_half4({value},',
            f'                    fm_ds41_round_half4({value});\n                    fm_mul_half4({value},')
        new=f'                    fm_ds41_round_half4({value});\n'+middle+f'\n                    fm_ds41_round_half4({value});'
        source=once(source,old,new)
    begin=source.index('                    // E2 boundaries, in order:')
    end=source.index('                    half4 ha(',begin)
    source=source[:begin]+'''                    // Match PackedExpert and the qualified thin-expert kernel:
                    // FP32 SiLU and route weight, then FP16 down-input boundary.
                    float weight = row_weight[row0 + mb * 16 + r];
                    float4 act;
                    act.x = ((g.x / (1.0f + expf(-g.x))) * u.x) * weight;
                    act.y = ((g.y / (1.0f + expf(-g.y))) * u.y) * weight;
                    act.z = ((g.z / (1.0f + expf(-g.z))) * u.z) * weight;
                    act.w = ((g.w / (1.0f + expf(-g.w))) * u.w) * weight;
'''+source[end:]
    source=once(source,'    const half* __restrict__ row_weight,','    const float* __restrict__ row_weight,')
    source=once(source,'        reinterpret_cast<const half*>(row_weight.data_ptr()),',
        '        reinterpret_cast<const float*>(row_weight.data_ptr()),')
    source=once(source,'                    float w = __half2float(row_weight[frow]);\n',
        '                    // Routing was already applied in the gate/up epilogue.\n')
    source=once(source,'                        fm_had_row(v, lane);',
        '                        fm_ds41_round_half4(v);\n                        fm_had_row(v, lane);\n                        fm_ds41_round_half4(v);')
    source=once(source,'                        v.x *= w; v.y *= w; v.z *= w; v.w *= w;',
        '                        fm_ds41_round_half4(v); // FP16 expert output, then FP32 scatter')
    source=once(source,'row_weight.scalar_type() == at::kHalf','row_weight.scalar_type() == at::kFloat')
    source=once(source,'"row_weight must be half[rows_cap]"','"row_weight must be float[rows_cap]"')
    source=once(source,'    float lim = (float) act_limit;',
        '''    TORCH_CHECK(row_weight.is_cuda() && row_weight.device() == h2.device()
                && row_weight.scalar_type() == at::kFloat && row_weight.is_contiguous()
                && row_weight.dim() == 1 && row_weight.size(0) >= h2.size(0),
                "row_weight must be contiguous float[rows_cap] on the same device");
    TORCH_CHECK(act_limit == 10.0, "DS41 activation limit must remain 10");
    float lim = (float) act_limit;''')
    source=once(source,'return (bits == 4 && (cb == 1 || cb == 2)) || ((bits == 3 || bits == 2) && cb == 2);',
        'return bits == 3 && cb == 2;')
    source=once(source,'int64_t exl3_fat_moe_abi() { return 2; }',
        'int64_t exl3_fat_moe_abi() { return 1003; } // Private DS41 rounding/routing ABI')
    for kind in ('gateup','down'):
        function=source.index('void exl3_fat_moe_'+kind+'(')
        begin=source.index('    if (bits == 4 && cb == 1)',function)
        end=source.index('    cuda_check(cudaPeekAtLastError());',begin)
        calls=[line for line in source[begin:end].splitlines() if f'launch_{kind}<3, 2>' in line]
        assert len(calls)==1
        call=calls[0].strip()
        if kind=='gateup':call=once(call,'down_suh_ptrs, h2, seg_expert','down_suh_ptrs, h2, row_weight, seg_expert')
        source=source[:begin]+'    '+call+'\n'+source[end:]
    source=source.replace('compiled for (4,mcg) (4,mul1) (3,mul1) (2,mul1)','compiled for DS41 (3,mul1) only')
    header=once(header,'    at::Tensor row_weight,   // [rows_cap] half',
        '    at::Tensor row_weight,   // [rows_cap] float; validated but NOT applied again')
    header+='\n// Private ABI 1003 supersedes upstream ABI 2: DS41 FP16 boundaries and pre-down routing.\n'
    assert 'float w = __half2float(row_weight' not in source and 'v.x *= w;' not in source
    assert source.count('fm_ds41_round_half4(')==10
    assert source.count('launch_gateup<3, 2>')==source.count('launch_down<3, 2>')==1
    return source,header


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.absolute()
    assert output.resolve()==output and output.parent==ROOT/'artifacts'
    assert output.name.startswith('miaai-grouped-prefill-source-v') and not output.exists()
    inputs={name:(VENDOR/name).read_bytes() for name in PINS}
    assert {name:sha(raw) for name,raw in inputs.items()}==PINS
    source,header=adapt(*(inputs[name].decode() for name in PINS))
    entries={'include/quant/exl3_fat_moe.cu':source.encode(),
        'include/quant/exl3_fat_moe.cuh':header.encode()}
    ext=ROOT/'vendor/exllamav3/exllamav3/exllamav3_ext'
    dependencies={}
    for path in sorted(ext.rglob('*')):
        if path.is_file() and path.suffix in ('.h','.cuh'):
            assert path.resolve()==path
            raw=path.read_bytes();dependencies[str(path.relative_to(ROOT))]=sha(raw)
            name='include/'+str(path.relative_to(ext))
            assert name not in entries
            entries[name]=raw
    assert dependencies
    for name in ('LICENSE','LICENSE.MIT','UPSTREAM.json'):
        entries[name]=(VENDOR/name).read_bytes()
    entries['LICENSE.exllamav3.MIT']=(ROOT/'vendor/exllamav3/LICENSE').read_bytes()
    entries['bindings.cpp']=b'''// SPDX-License-Identifier: AGPL-3.0-only
#include "include/quant/exl3_fat_moe.cuh"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather", &exl3_fat_moe_gather);
    m.def("gateup", &exl3_fat_moe_gateup);
    m.def("down", &exl3_fat_moe_down);
    m.def("tile_rows_gateup", &exl3_fat_moe_tile_rows_gateup);
    m.def("tile_rows_down", &exl3_fat_moe_tile_rows_down);
    m.def("abi", &exl3_fat_moe_abi);
}
'''
    receipt=dict(status='grouped_prefill_sources_prepared_not_compiled',upstream_sha256=PINS,
        dependency_sha256=dependencies,generated_sha256={n:sha(b) for n,b in entries.items()},
        generator_sha256=sha(Path(__file__).read_bytes()),abi=1003,bits=3,codebook='MUL1',
        routing_before_down_fp16=True,fp16_gemm_and_output_boundaries=True,
        required_cuda_flags=['-O3','-lineinfo','--fmad=false'],
        gpu_used=False,compiled=False,serving_qualified=False,live_serving_modified=False)
    assert shutil.disk_usage(ROOT).free>=32*2**30
    output.mkdir()
    for name,raw in entries.items():
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(raw)
    with (output/'prepared.json').open('x') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps(dict(status=receipt['status'],output=str(output),files=len(entries),abi=1003)),flush=True)


if __name__=='__main__':main()
