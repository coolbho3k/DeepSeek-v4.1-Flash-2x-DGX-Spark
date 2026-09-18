"""CPU-only, offline additive kernel build in /build; no original source edits.

Code-generate two narrowly adapted EXL3 headers into the disposable build
tree. Pin every header/source and retain generated sources, MIT license,
compiler output and binary digest. Never replace the installed extension.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path('/work')
BUILD = Path('/build')
EXT = ROOT/'vendor/exllamav3/exllamav3/exllamav3_ext'
OWN = ROOT/'kernels/exl3_moe_mul1'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def exclusive(name, value):
    with (BUILD/name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def once(text, before, after):
    assert text.count(before) == 1, ('Unreviewed kernel anchor', before)
    return text.replace(before, after, 1)


def replace_block(text, begin, end, replacement):
    assert text.count(begin) == text.count(end) == 1
    start = text.index(begin); stop = text.index(end, start)
    return text[:start] + replacement + text[stop:]


def generate():
    source_files = sorted(p for p in EXT.rglob('*') if p.suffix in ('.h', '.cuh'))
    source_files += sorted(OWN.iterdir()) + [Path(__file__), ROOT/'vendor/exllamav3/LICENSE']
    pins = {str(p.relative_to(ROOT)): digest(p) for p in source_files}
    include = BUILD/'include'
    include.mkdir()
    for source in source_files:
        if source.is_relative_to(EXT):
            target = include/source.relative_to(EXT)
        elif source.is_relative_to(OWN):
            target = BUILD/source.name
        else:
            target = BUILD/source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    # Preserve all other scheduling, GEMM, group barriers and codebook logic.
    common_path = include/'quant/exl3_moe_common.cuh'
    common = once(common_path.read_text(), 'const half* __restrict__ weight_sorted,',
                  'const float* __restrict__ weight_sorted,')
    kernel_path = include/'quant/exl3_moe_kernel.cuh'
    kernel = once(kernel_path.read_text(), 'void exl3_moe_kernel(EXL3_MOE_KERNEL_ARGS)',
                  'void ds41_moe_mul1_kernel(EXL3_MOE_KERNEL_ARGS)')
    kernel = once(kernel, '#include "hadamard_inner.cuh"',
                  '#include "hadamard_inner.cuh"\n#include "semantics.cuh"')
    kernel = replace_block(kernel,
        '        // Output hadamard for g, u + activation+gate + input hadamard for d',
        '        // d GEMM', '''        // DS4.1: clamp BEFORE FP32 SiLU, route in FP32 BEFORE down GEMM.
        const int gu_warps_per_token = intermediate_dim / 128;
        for (int warp_idx = warp_idx0; warp_idx < token_count * gu_warps_per_token; warp_idx += warps_per_group)
        {
            int token_off = warp_idx % gu_warps_per_token;
            ds41_guad(temp_intermediate_g + 128 * warp_idx,
                temp_intermediate_u + 128 * warp_idx,
                exp_gate_svh + 128 * token_off, exp_up_svh + 128 * token_off,
                exp_down_suh + 128 * token_off,
                weight_sorted[start + warp_idx / gu_warps_per_token]);
        }
        group_barrier(group_idx, group_size, barrier_counters_sense);

''')
    kernel = replace_block(kernel,
        '        // Output hadamard for d + scatter add',
        '        // Draw the next ticket', '''        // DS4.1: FP16 expert-output boundary, then FP32 scatter sum.
        const int d_warps_per_token = hidden_dim / 128;
        for (int warp_idx = warp_idx0; warp_idx < token_count * d_warps_per_token; warp_idx += warps_per_group)
        {
            int token_idx = token_sorted[start + warp_idx / d_warps_per_token];
            int token_off = warp_idx % d_warps_per_token;
            ds41_down_out(temp_state_g + 128 * warp_idx,
                temp_state_u + 128 * warp_idx, exp_down_svh + 128 * token_off,
                output_state + token_idx * hidden_dim + token_off * 128);
        }

''')
    # Generated build outputs only; the read-only vendored source is untouched.
    common_path.write_text(common)
    kernel_path.write_text(kernel)
    assert pins == {relative: digest(ROOT/relative) for relative in pins}
    return pins, {str(p.relative_to(BUILD)): digest(p) for p in BUILD.rglob('*') if p.is_file()}


def main():
    assert not any((BUILD/name).exists() for name in ('attempt.json','complete.json','failed.json','include'))
    memory = {line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
              if line.split(':')[0] in ('MemFree','MemAvailable')}
    assert memory['MemAvailable'] >= 96*2**30 and memory['MemFree'] >= 32*2**30
    assert shutil.disk_usage(BUILD).free >= 32*2**30
    limits = {name:(Path('/sys/fs/cgroup')/name).read_text().strip()
              for name in ('memory.max','memory.swap.max','cpu.max')}
    assert limits['memory.max'] == str(16*2**30) and limits['memory.swap.max'] == '0'
    quota, period = map(int, limits['cpu.max'].split()); assert quota <= 4*period
    assert os.environ['MAX_JOBS'] == '1' and os.environ['TORCH_CUDA_ARCH_LIST'] == '12.1a'
    pins, generated = generate()
    exclusive('attempt.json', dict(source_sha256=pins, generated_sha256=generated,
        memory=memory, cgroup_limits=limits, cuda_visible=False, replaces_baked_extension=False,
        specialized_bits=3, specialized_codebook='MUL1', hidden=5120, tp_intermediate=1152))
    start = time.monotonic()
    try:
        import torch
        assert not torch.cuda.is_initialized() and not torch.cuda.is_available()
        from torch.utils.cpp_extension import load
        module = load(name='ds41_moe_mul1_v1', sources=[str(BUILD/'bindings.cpp'),str(BUILD/'launch.cu')],
            extra_include_paths=[str(BUILD/'include'),str(BUILD)],
            extra_cflags=['-O3'], extra_cuda_cflags=['-O3','-lineinfo','--fmad=false'],
            build_directory=str(BUILD), with_cuda=True, verbose=True)
        assert module.contract_version() == 1 and not torch.cuda.is_initialized()
        assert pins == {relative:digest(ROOT/relative) for relative in pins}
        binary = Path(module.__file__)
        exclusive('complete.json', dict(status='additive_mul1_kernel_built_cpu_only',
            elapsed_seconds=time.monotonic()-start, source_sha256=pins, generated_sha256=generated,
            binary_name=binary.name, binary_sha256=digest(binary), binary_bytes=binary.stat().st_size,
            torch_version=torch.__version__, cuda_version=torch.version.cuda, cuda_initialized=False,
            actual_gpu_qualified=False, serving_modified=False))
        print(json.dumps(dict(status='additive_mul1_kernel_built_cpu_only', binary=str(binary),
                              elapsed_seconds=time.monotonic()-start)), flush=True)
    except Exception as exc:
        exclusive('failed.json', dict(status='failed', error=repr(exc), elapsed_seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    main()
