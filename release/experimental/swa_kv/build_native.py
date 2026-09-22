# SPDX-License-Identifier: Apache-2.0
"""Build the group-32 SWA writer from the pinned Apache-2.0 vLLM source.

Run inside the pinned serving image. This does not load a model or touch CUDA.
The original source and license are shipped in vendor/vllm-swa32-apache.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(data):
    return hashlib.sha256(data).hexdigest()


def transform(source):
    # Keep the exact paged kernel and its existing decode/prefill dispatch.
    # Drop unrelated full-cache kernels and their operator wrappers.
    begin = source.index('torch::stable::Tensor fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(')
    end = source.index('// FlashInfer full-cache torch ops', begin)
    wrapper = source[begin:end].rsplit('// ─', 1)[0]
    end_kernel = source.index('// FlashInfer full-cache kernel')
    kernel = source[:end_kernel].rsplit('// ─', 1)[0]
    result = kernel + '\n}  // namespace deepseek_v4_fused_ops\n}  // namespace vllm\n\n' + wrapper
    edits = [
        ('QUANT_BLOCK = 64 (UE8M0 FP8 quant block)', 'QUANT_BLOCK = 32 (UE8M0 FP8 quant block)'),
        ('[bs*576,       bs*576 + bs*8):   UE8M0 scales, 7 real + 1 pad per token',
         '[bs*576,       bs*576 + bs*16):  UE8M0 scales, 14 real + 2 pad per token'),
        ('constexpr int kQuantBlock = 64;', 'constexpr int kQuantBlock = 32;'),
        ('// 7\nconstexpr int kScaleBytesPerToken = kNumQuantBlocks + 1;  // 8 (7 real + 1 pad)',
         '// 14\nconstexpr int kScaleBytesPerToken = 16;  // 14 real + 2 pad'),
        ('  // Reduce absolute max across 4 consecutive lanes (lane id & 3 group).',
         '  // Reduce absolute max across 2 lanes, 16 values per lane.'),
        ('  peer = __shfl_xor_sync(FINAL_MASK, val, 2);\n  val = fmaxf(val, peer);\n', ''),
        # log2f with --use_fast_math can round exact powers of two upward
        # before ceil (e.g. BF16 amax=1.75); derive the exponent exactly.
        ('float const exponent = ceilf(log2f(absmax / kFp8Max));',
         'int const bits = __float_as_int(absmax);\n'
         '      float const exponent = ((bits >> 23) & 255) - 135 +\n'
         '          ((bits & 0x7fffff) > 0x600000);'),
        ('if ((laneId & 3) == 0)', 'if ((laneId & 1) == 0)'),
        ('int const q_block_idx = laneId >> 2;', 'int const q_block_idx = laneId >> 1;'),
        ('token_scale_ptr[kNumQuantBlocks] = 0;',
         'token_scale_ptr[kNumQuantBlocks] = 0;\n          token_scale_ptr[kNumQuantBlocks + 1] = 0;'),
    ]
    for old, new in edits:
        if result.count(old) != 1:
            raise ValueError('Unreviewed native anchor: ' + old)
        result = result.replace(old, new)
    result = result.replace('warp4MaxAbs', 'warp2MaxAbs')
    result = result.replace('deepseek_v4_fused_ops', 'ds41_swa32_ops')
    result = result.replace('fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert', 'ds41_swa32_insert')
    anchor = '  int const kv_block_stride = static_cast<int>(k_cache.stride(0));\n'
    validation = '''  STD_TORCH_CHECK(q_in.scalar_type() == torch::headeronly::ScalarType::BFloat16,
                  "SWA32 requires BF16 inputs");
  STD_TORCH_CHECK(k_cache.dim() == 2 && k_cache.stride(1) == 1 &&
                  cache_block_size > 0 && cache_block_size % 32 == 0 &&
                  k_cache.size(1) >= cache_block_size * 592 &&
                  k_cache.stride(0) >= k_cache.size(1) &&
                  k_cache.stride(0) <= 2147483647,
                  "SWA32 requires 592-byte states in strided packed pages");
  STD_TORCH_CHECK(slot_mapping.dim() == 1 && slot_mapping.is_contiguous() &&
                  position_ids.dim() == 1 && position_ids.is_contiguous() &&
                  cos_sin_cache.is_contiguous() &&
                  kv.device() == q_in.device() && k_cache.device() == q_in.device() &&
                  slot_mapping.device() == q_in.device() && position_ids.device() == q_in.device() &&
                  cos_sin_cache.device() == q_in.device(),
                  "SWA32 expects contiguous slots/positions/phases on the input device");
'''
    assert result.count(anchor) == 1
    result = result.replace(anchor, validation + anchor)
    result = result.replace('  VLLM_STABLE_DISPATCH_HALF_TYPES(', '  if (num_tokens_full == 0) return q_out;\n\n  VLLM_STABLE_DISPATCH_HALF_TYPES(')
    result += '''
#include <torch/csrc/stable/library.h>
STABLE_TORCH_LIBRARY_FRAGMENT(ds41_swa32, ops) {
  ops.def("insert(Tensor q_in, Tensor kv, Tensor! k_cache, Tensor slot_mapping, "
          "Tensor position_ids, Tensor cos_sin_cache, int q_head_padded, float eps, "
          "int cache_block_size, bool apply_q_norm=True) -> Tensor");
}
STABLE_TORCH_LIBRARY_IMPL(ds41_swa32, CUDA, ops) {
  ops.impl("insert", TORCH_BOX(&ds41_swa32_insert));
}
'''
    return '// DS41 adaptation: group-32 FP8 NoPE, unchanged BF16 Q/RoPE math.\n' + result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    vendor = args.runtime / 'vendor/vllm-swa32-apache'
    upstream = json.loads((vendor / 'UPSTREAM.json').read_bytes())
    for name, row in upstream['files'].items():
        if sha((vendor/name).read_bytes()) != row['sha256']:
            raise ValueError('Changed native source: ' + name)
    source = transform((vendor/'csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    generated = args.output/'swa32.cu'; generated.write_text(source)
    torch_root = Path('/usr/local/lib/python3.12/dist-packages/torch')
    binary = args.output/'libds41_swa32.so'
    command = ['nvcc', '-std=c++17', '-O3', '--use_fast_math', '-lineinfo',
               '-gencode=arch=compute_121a,code=sm_121a', '--expt-relaxed-constexpr',
               '-shared', '-DUSE_CUDA', '-Xcompiler=-fPIC,-fvisibility=hidden', '--threads=1',
               '-I'+str(vendor/'csrc/libtorch_stable'), '-I'+str(vendor/'csrc'),
               '-I'+str(torch_root/'include'), '-I'+str(torch_root/'include/torch/csrc/api/include'),
               str(generated), '-L'+str(torch_root/'lib'), '-ltorch', '-ltorch_cpu',
               '-ltorch_cuda', '-lc10', '-lc10_cuda', '-o', str(binary)]
    subprocess.run(command, check=True)
    result = dict(status='swa32_native_built', group_size=32, scale_bytes=16,
                  state_bytes=592, rope_dtype='bfloat16', source_sha256=sha(source.encode()),
                  binary_sha256=sha(binary.read_bytes()), upstream_manifest_sha256=sha((vendor/'UPSTREAM.json').read_bytes()),
                  command=command, gpu_work_performed=False)
    (args.output/'build.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
