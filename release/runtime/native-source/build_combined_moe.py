# SPDX-License-Identifier: AGPL-3.0-only
# The reused EXL3 headers retain their original MIT license in the artifact.
"""CPU-only native launcher rebuild for the combined graphs/1536-row path."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import build_exl3_moe_mul1 as original

BUILD = Path('/build')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    candidates=parser.add_mutually_exclusive_group()
    candidates.add_argument('--decode-tuning',action='store_true')
    candidates.add_argument('--register-gemv',action='store_true')
    candidates.add_argument('--staged-decode',action='store_true')
    candidates.add_argument('--staged-small',action='store_true')
    candidates.add_argument('--staged-grouped',action='store_true')
    args=parser.parse_args()
    assert not any(BUILD.iterdir())
    limits = {n: (Path('/sys/fs/cgroup') / n).read_text().strip()
              for n in ('memory.max', 'memory.swap.max', 'cpu.max')}
    assert limits == {'memory.max': str(16 * 2**30), 'memory.swap.max': '0', 'cpu.max': '400000 100000'}
    memory = {r.split(':')[0]: int(r.split()[1]) * 1024
              for r in Path('/proc/meminfo').read_text().splitlines()
              if r.split(':')[0] in ('MemAvailable', 'MemFree')}
    assert memory['MemAvailable'] >= 96 * 2**30 and memory['MemFree'] >= 32 * 2**30
    assert shutil.disk_usage(BUILD).free >= 32 * 2**30
    assert os.environ['MAX_JOBS'] == '1' and os.environ['TORCH_CUDA_ARCH_LIST'] == '12.1a'
    import torch
    assert not torch.cuda.is_initialized() and not torch.cuda.is_available()
    pins, _ = original.generate()
    # Reuse the existing, qualified mathematical header transformations.
    # Only the host launch/admission code changes below; private scratch,
    # cooperative grid, precision boundaries and all device math stay exact.
    path = BUILD / 'launch.cu'
    source = path.read_text()
    source = original.once(source, '#include <vector>', '#include <vector>\n#include <mutex>')
    source = original.once(source, 'x.size(0) <= 1056', 'x.size(0) <= 2048')
    source = original.once(source, '    cudaDeviceProp properties;', '''    // Bounded single-device resource cache. Prewarm before graph capture;
    // repeated forwards never mutate function attributes inside a graph.
    static std::mutex resource_mutex;
    static int resource_device = -1;
    static std::vector<int64_t> resource_values;
    const std::lock_guard<std::mutex> lock(resource_mutex);
    if (!resource_values.empty()) {
        TORCH_CHECK(resource_device == device, "One visible device per combined worker");
        return resource_values;
    }
    cudaStreamCaptureStatus capture;
    C10_CUDA_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream().stream(), &capture));
    TORCH_CHECK(capture == cudaStreamCaptureStatusNone, "Prewarm combined MoE resources before capture");
    cudaDeviceProp properties;''')
    source = original.once(source, '    return {properties.multiProcessorCount,',
        '    resource_device = device;\n    resource_values = {properties.multiProcessorCount,')
    source = original.once(source, '            DS41_LOCKS, MOE_SMS_PER_EXPERT};',
        '            DS41_LOCKS, MOE_SMS_PER_EXPERT};\n    return resource_values;')
    source = original.once(source, '''    cudaStreamCaptureStatus capture;
    C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    TORCH_CHECK(capture == cudaStreamCaptureStatusNone, "Diagnostic requires eager execution");''',
        '''    // The owned Python dispatcher fences the caller-provided scratch.
    // Native CUDA graph capture preserves this cooperative kernel launch.''')
    path.write_text('// SPDX-License-Identifier: AGPL-3.0-only\n' + source)
    path = BUILD / 'bindings.cpp'
    source = original.once(path.read_text(), 'return 1;', 'return 2;')
    source = original.once(source, '    m.def("resources", &ds41_mul1_resources);',
        '    m.def("resources", &ds41_mul1_resources);\n'
        '    m.def("max_rows", []() { return 2048; });\n'
        '    m.def("graph_capture_supported", []() { return true; });')
    path.write_text('// SPDX-License-Identifier: AGPL-3.0-only\n' + source)
    shutil.copyfile(Path(__file__), BUILD / Path(__file__).name)
    decode_candidate=None
    if args.decode_tuning:
        from prepare_decode_occupancy_sources import prepare
        decode_candidate=prepare(BUILD)
    register_candidate=None
    if args.register_gemv:
        from prepare_register_gemv_sources import prepare
        register_candidate=prepare(BUILD)
    staged_candidate=None
    if args.staged_decode or args.staged_small or args.staged_grouped:
        from prepare_staged_decode_sources import prepare
        staged_candidate=prepare(BUILD)
    small_candidate=None
    if args.staged_small or args.staged_grouped:
        from prepare_staged_small_sources import prepare
        small_candidate=prepare(BUILD)
    grouped_candidate=None
    if args.staged_grouped:
        from prepare_staged_grouped_sources import prepare
        grouped_candidate=prepare(BUILD)
    shutil.copyfile(original.ROOT / 'vendor/miaai-serving-stack-agpl/LICENSE', BUILD / 'LICENSE.AGPL-3.0')
    generated = {str(p.relative_to(BUILD)): original.digest(p) for p in BUILD.rglob('*') if p.is_file()}
    original.exclusive('attempt.json', dict(status='combined_moe_build_started', source_sha256=pins,
        generated_sha256=generated, memory=memory, cgroup_limits=limits, cuda_visible=False))
    start = time.monotonic()
    try:
        from torch.utils.cpp_extension import load
        module = load(name='ds41_moe_mul1_v1',
            sources=[str(BUILD / 'bindings.cpp'), str(BUILD / 'launch.cu')]
                + ([str(BUILD/'decode-launch.cu')] if args.decode_tuning else [])
                + ([str(BUILD/'gemv-launch.cu')] if args.register_gemv else [])
                + ([str(BUILD/'staged-launch.cu')] if args.staged_decode or args.staged_small or args.staged_grouped else [])
                + ([str(BUILD/'staged-small-launch.cu')] if args.staged_small or args.staged_grouped else [])
                + ([str(BUILD/'staged-grouped-launch.cu')] if args.staged_grouped else []),
            extra_include_paths=[str(BUILD / 'include'), str(BUILD)],
            extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '-lineinfo', '--fmad=false'],
            build_directory=str(BUILD), with_cuda=True, verbose=True)
        assert module.contract_version() == 2 and module.max_rows() == 2048 and module.graph_capture_supported()
        assert not torch.cuda.is_initialized()
        assert all(original.digest(original.ROOT / n) == p for n, p in pins.items())
        assert all(original.digest(BUILD / n) == p for n, p in generated.items())
        binary = Path(module.__file__)
        result = dict(status='combined_moe_built_cpu_only', elapsed_seconds=time.monotonic() - start,
            decode_candidate=decode_candidate,
            register_gemv_candidate=register_candidate,
            staged_decode_candidate=staged_candidate,
            staged_small_candidate=small_candidate,
            staged_grouped_candidate=grouped_candidate,
            source_sha256=pins, generated_sha256=generated, binary_name=binary.name,
            binary_sha256=original.digest(binary), binary_bytes=binary.stat().st_size,
            contract_version=2, max_rows=2048, graph_capture_supported=True,
            cuda_initialized=False, gpu_qualified=False, torch_version=torch.__version__)
        original.exclusive('complete.json', result)
        print(json.dumps({k: result[k] for k in ('status', 'elapsed_seconds', 'binary_sha256')}), flush=True)
    except BaseException as error:
        original.exclusive('failed.json', dict(error=repr(error)))
        raise


if __name__ == '__main__': main()
