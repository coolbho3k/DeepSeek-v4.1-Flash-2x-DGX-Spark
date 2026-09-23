# SPDX-License-Identifier: AGPL-3.0-only
"""Prepare a dual-projection gather retaining the qualified native math."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VENDOR = ROOT / 'release/runtime/vendor/miaai-grouped-prefill-ds41-v2'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Changed native source anchor: ' + before[:60])
    return text.replace(before, after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.absolute()
    if out.resolve() != out or out.exists():
        raise ValueError('Fresh, unredirected source directory required')
    source_path = VENDOR / 'include/quant/exl3_fat_moe.cu'
    manifest = json.loads((ROOT / 'release/runtime/bundle-manifest.json').read_bytes())
    entry = manifest['files'][str(source_path.relative_to(ROOT / 'release/runtime'))]
    source = source_path.read_bytes()
    if sha(source) != entry['sha256']:
        raise ValueError('Changed qualified grouped MoE source')
    text = source.decode()
    helpers = text[text.index('namespace {'):text.index('// ---------------------------------------------------------------------------\n// Gather')]
    start = text.index('__global__ __launch_bounds__(FM_THREADS)\nvoid fm_gather_kernel(')
    stop = text.index('\n// ---------------------------------------------------------------------------', start)
    kernel = text[start:stop]
    kernel = once(kernel, 'void fm_gather_kernel(', 'void ds41_dual_gather_kernel(')
    kernel = once(kernel, 'const half* const* __restrict__ suh_ptrs,',
                  'const half* const* __restrict__ gate_suh_ptrs,\n    const half* const* __restrict__ up_suh_ptrs,')
    kernel = once(kernel, 'half* __restrict__ h13,',
                  'half* __restrict__ h13g,\n    half* __restrict__ h13u,')
    kernel = once(kernel, 'const half* suh = suh_ptrs[row_expert[row]] + blk * 128;',
                  'const int expert = row_expert[row];\n        const half* gate_suh = gate_suh_ptrs[expert] + blk * 128;\n        const half* up_suh = up_suh_ptrs[expert] + blk * 128;')
    kernel = once(kernel, 'half* dst = h13 + (int64_t) row * size_k + blk * 128;',
                  'half* dstg = h13g + (int64_t) row * size_k + blk * 128;\n        half* dstu = h13u + (int64_t) row * size_k + blk * 128;')
    begin = kernel.index('        half4 hv = ')
    end = kernel.index('\n    }', begin)
    old = kernel[begin:end]
    # Compute each transform with exactly the original arithmetic. The input
    # half4 is loaded once; each scale multiply still rounds to FP16 first.
    gate = old.replace('half4 hv = *reinterpret_cast<const half4*>(src + lane * 4);', 'half4 hv = input;').replace('suh + lane', 'gate_suh + lane').replace('dst + lane', 'dstg + lane')
    up = gate.replace('gate_suh + lane', 'up_suh + lane').replace('dstg + lane', 'dstu + lane')
    kernel = kernel[:begin] + '        const half4 input = *reinterpret_cast<const half4*>(src + lane * 4);\n        {\n' + gate + '\n        }\n        {\n' + up + '\n        }' + kernel[end:]
    wrapper = r'''
} // namespace
extern "C" int ds41_dual_gather_abi() { return 1; }
extern "C" int ds41_dual_gather_info(int* info) {
    if (!info) return int(cudaErrorInvalidValue);
    cudaFuncAttributes attr;
    auto err = cudaFuncGetAttributes(&attr, ds41_dual_gather_kernel);
    if (err != cudaSuccess) return int(err);
    int blocks;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, ds41_dual_gather_kernel, FM_THREADS, 0);
    if (err != cudaSuccess) return int(err);
    info[0] = FM_THREADS; info[1] = attr.numRegs; info[2] = attr.localSizeBytes; info[3] = blocks;
    return 0;
}
extern "C" int ds41_dual_gather(void** pointers, int rows_bound, int width, void* raw_stream) {
    if (!pointers || rows_bound < 1 || rows_bound > 18432 || width != 5120) return int(cudaErrorInvalidValue);
    for (int i=0; i<8; ++i) if (!pointers[i]) return int(cudaErrorInvalidValue);
    int gx = (rows_bound + FM_WARPS - 1) / FM_WARPS;
    if (gx > 1024) gx = 1024;
    ds41_dual_gather_kernel<<<dim3(gx, width / 128), FM_THREADS, 0, (cudaStream_t)raw_stream>>>(
        (const half*)pointers[0], (const int64_t*)pointers[1], (const int*)pointers[2],
        (const half* const*)pointers[3], (const half* const*)pointers[4],
        (half*)pointers[5], (half*)pointers[6], (const int*)pointers[7], width);
    return int(cudaGetLastError());
}
'''
    prefix = '// SPDX-License-Identifier: AGPL-3.0-only\n// Derived from MiaAI Lab / Wesley Young grouped MoE and Turboderp ExLlamaV3.\n// Local change: fuse gate/up input gather; bound the grid by routed batch size.\n#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n#include "util.cuh"\n#include "quant/hadamard_inner.cuh"\n'
    files = {'dual_gather.cu': (prefix + helpers + kernel + wrapper).encode()}
    for path in (VENDOR / 'include').rglob('*'):
        if path.is_file():
            files['include/' + str(path.relative_to(VENDOR / 'include'))] = path.read_bytes()
    for name in ('LICENSE', 'LICENSE.exllamav3.MIT', 'UPSTREAM.json'):
        files[name] = (VENDOR / name).read_bytes()
    out.mkdir(parents=True)
    for name, data in files.items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    receipt = dict(status='dual_gather_prepared_not_gpu_qualified',
                   parent_source_sha256=sha(source),
                   files={name: sha(data) for name, data in sorted(files.items())},
                   additional_persistent_gpu_bytes=0,
                   retained_input_and_output_rounding='fp16',
                   generator_sha256=sha(Path(__file__).read_bytes()))
    (out / 'prepared.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(status=receipt['status'], path=str(out), source_sha256=sha(files['dual_gather.cu']))))


if __name__ == '__main__':
    main()
