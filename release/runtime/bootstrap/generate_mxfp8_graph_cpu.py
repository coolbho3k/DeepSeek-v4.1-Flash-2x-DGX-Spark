"""Generate the pinned FlashInfer MXFP8 graph in an EMPTY CPU-only cache.

Run only in the pinned image, with both serving workers stopped. This writes
sources/build.ninja, never invokes Ninja, loads a library, or loads a model.
Native empty-cache execution is required before this is a qualified bootstrap.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys


GIB = 2**30
PACKAGE = Path('/usr/local/lib/python3.12/dist-packages/flashinfer')
CACHE = Path('/cache/flashinfer')
WORKSPACE = CACHE/'.cache/flashinfer/0.6.18.dev20260819/121a'
GRAPH = WORKSPACE/'cached_ops/mxfp8_gemm_cutlass_sm120/build.ninja'
GRAPH_SHA256 = '3aeab12adc90d94d85ca58e94c450a79eb82976d082e5cbbf60ac0c08da031dc'
SOURCE_SHA256 = {
    'jit/core.py': '33e49ddda268b3df5b57e3abd592ab761ece732785514aff3155dd37c6088b75',
    'jit/gemm/core.py': '292a23c295fa2174b156fc9e757fbceddd3504c38bf4498af960c57066c9fe81',
    'jit/cpp_ext.py': 'c2ebe90fa1ddc3896553c51a4799e43ee8e2cd506740d3e3b1493280bfd7262b',
    'compilation_context.py': '65d3afe9c069f9d685a9f99a15e307f02d966a7112681e2d1d5ca70c054336cd',
    'jit/env.py': 'c3955fd0b83154356942840c1217e7c64a3602766feddbd701f814bae25ff47e',
    'data/csrc/mxfp8_gemm_cutlass_sm120.cu': '896676c7f8f8238958f3c918c611963a0630729a664f82511ed509e57bb6c962',
    'data/csrc/mxfp8_gemm_cutlass_sm120.jinja': '692ef56546cb9a50120b0138c5949e90e62c238a578c7232ae2cf7ea7d8ee1ff',
}
EXPECTED_ENV = {
    'NVIDIA_VISIBLE_DEVICES': 'void',
    'CUDA_VISIBLE_DEVICES': '',
    'FLASHINFER_CUDA_ARCH_LIST': '12.1a',
    'FLASHINFER_WORKSPACE_BASE': str(CACHE),
    'MAX_JOBS': '2',
}


def seeded_graph(native):
    """Change only NVCC's internal-name seed, uniquely for each input file.

    Ninja expands $in at the compilation edge, not once for the whole graph.
    Keep the native graph separately; this is a new build, not the old binary.
    """
    if hashlib.sha256(native).hexdigest() != GRAPH_SHA256:
        raise ValueError('Deterministic transform requires the exact native graph')
    before = b'command = $nvcc_launcher $nvcc --generate-dependencies-with-compile'
    after = b'command = $nvcc_launcher $nvcc --frandom-seed=$in --generate-dependencies-with-compile'
    if native.count(before) != 1 or b'frandom-seed' in native:
        raise ValueError('Unexpected CUDA compiler command')
    return native.replace(before, after)


def validate_environment(env, available, limits, devices, isolated):
    if not isolated or devices:
        raise ValueError('Isolated Python and a driver-free CPU container are required')
    if any(env.get(k) != v for k, v in EXPECTED_ENV.items()):
        raise ValueError('Explicit CPU-only device, architecture and cache settings required')
    if type(available) is not int or available < 48*GIB:
        raise ValueError('Wait for idle serving workers and at least48GiB available RAM')
    memory, swap = limits['memory.max'], limits['memory.swap.max']
    quota, period = limits['cpu.max'].split()
    if (memory == 'max' or not 0 < int(memory) <= 4*GIB or swap != '0'
            or quota == 'max' or not 0 < int(quota) <= 2*int(period)
            or int(period) <= 0):
        raise ValueError('Use at most4GiB/no-swap/twoCPU for graph generation')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_imported_sources(modules):
    for relative in SOURCE_SHA256:
        if not relative.endswith('.py'):
            continue
        name = 'flashinfer.' + relative[:-3].replace('/', '.')
        module = modules.get(name)
        if (module is None or not getattr(module, '__file__', None)
                or Path(module.__file__).resolve() != (PACKAGE/relative).resolve()):
            raise ValueError('Imported FlashInfer modules differ from pinned source paths')


def validate_graph(spec):
    if (spec.name != 'mxfp8_gemm_cutlass_sm120' or spec.ninja_path != GRAPH
            or spec.needs_device_linking or len(spec.sources) != 11
            or digest(GRAPH) != GRAPH_SHA256):
        raise ValueError('Generated graph differs from the tested pinned graph')
    generated = WORKSPACE/'generated/gen_gemm_sm120_cutlass_mxfp8'
    expected = {
        generated/f'mxfp8_gemm_cutlass_sm120_{dtype}_{m}_{n}_{k}.cu'
        for dtype in ('__nv_bfloat16', 'half')
        for m, n, k in ((128,32,128),(128,64,128),(128,128,128),
                        (256,128,128),(128,256,128))
    } | {PACKAGE/'data/csrc/mxfp8_gemm_cutlass_sm120.cu'}
    if set(spec.sources) != expected or any(p.is_symlink() for p in expected):
        raise ValueError('Unexpected generated source inventory')
    if list(CACHE.rglob('*.so')) or list(CACHE.rglob('*.o')):
        raise ValueError('Graph-only preparation unexpectedly produced compiled artifacts')
    return {str(p): digest(p) for p in sorted(expected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()
    limits = {name: (Path('/sys/fs/cgroup')/name).read_text().strip()
              for name in ('memory.max', 'memory.swap.max', 'cpu.max')}
    mem = {p[0][:-1]: int(p[1])*1024
           for line in Path('/proc/meminfo').read_text().splitlines()
           if (p := line.split())[0] == 'MemAvailable:'}
    devices = list(Path('/dev').glob('nvidia*')) + list(Path('/dev').glob('dri/render*'))
    validate_environment(os.environ, mem['MemAvailable'], limits, devices, sys.flags.isolated)
    if (Path('/cache').resolve() != Path('/cache') or not Path('/cache').is_dir()
            or CACHE.exists() or CACHE.is_symlink()
            or Path('/cache/graph-receipt.json').exists()
            or Path('/cache/graph-receipt.json').is_symlink()):
        raise ValueError('Use a fresh cache mount; existing caches must not be modified')
    actual = {name: digest(PACKAGE/name) for name in SOURCE_SHA256}
    if actual != SOURCE_SHA256:
        raise ValueError('Installed generator source does not match the inspected runtime')
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    # Import only after all preconditions. The container must have no GPU
    # devices; explicit12.1a avoids hardware-based architecture discovery.
    import torch
    from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_mxfp8
    validate_imported_sources(sys.modules)
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA unexpectedly initialized during import')
    spec = gen_gemm_sm120_module_cutlass_mxfp8()
    spec.write_ninja()
    sources = validate_graph(spec)
    if args.deterministic:
        native = GRAPH.read_bytes()
        with GRAPH.with_name('build.native.ninja').open('xb') as stream:
            stream.write(native)
        GRAPH.write_bytes(seeded_graph(native))
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA unexpectedly initialized during graph generation')
    result = dict(status=('clean_mxfp8_seeded_graph_generated' if args.deterministic
                         else 'clean_mxfp8_graph_matches_tested_graph'),
        graph_sha256=digest(GRAPH), generator_source_sha256=actual,
        native_graph_sha256=GRAPH_SHA256, deterministic_per_source_seed=args.deterministic,
        generated_and_static_source_sha256=sources, limits=limits,
        initial_available_bytes=mem['MemAvailable'], cuda_initialized=False,
        compilation_performed=False, gpu_kernel_execution_tested=False,
        probe_sha256=digest(Path(__file__)))
    with Path('/cache/graph-receipt.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
