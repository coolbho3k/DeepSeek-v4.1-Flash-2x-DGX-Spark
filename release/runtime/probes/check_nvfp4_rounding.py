# SPDX-License-Identifier: AGPL-3.0-only
"""Exhaustive supported BF16/E4M3 rounding checks for the native FP4 encoder."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import triton
import triton.language as tl
from bench_nvfp4_kv import load_package


@triton.jit
def check_kernel(values, scale_bytes, output, ENCODE: tl.constexpr, COUNT: tl.constexpr):
    channel = tl.program_id(1) * 512 + tl.arange(0, 512)
    scale = tl.load(scale_bytes + tl.program_id(0)).to(tl.float8e4nv, bitcast=True)
    groups = tl.reshape(tl.load(values + channel, channel < COUNT, other=0).to(tl.float32), (32, 16))
    scales = tl.full((32,), 0, tl.float32) + scale.to(tl.float32)
    codes = ENCODE(groups, scales.to(tl.float8e4nv))
    tl.store(output + tl.program_id(0) * COUNT + channel, tl.reshape(codes, (512,)), channel < COUNT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.002)
    codec, _ = load_package(args.runtime, 'rounding_codec', 'nvfp4_4over6')
    # Every finite nonnegative BF16 bit pattern from zero through 2688,
    # the maximum finite value representable with a 448 E4M3 scale.
    positive = torch.arange(0x4528 + 1, dtype=torch.int16).view(torch.bfloat16)
    assert positive[-1].item() == 2688
    values = torch.cat((positive, -positive)).cuda()
    scales = torch.arange(1, 127, dtype=torch.uint8).cuda()
    output = torch.empty(126, len(values), device='cuda', dtype=torch.uint8)
    check_kernel[(126, triton.cdiv(len(values), 512))](values, scales, output,
        ENCODE=codec._encode_e2m1, COUNT=len(values), num_warps=4)
    actual = output.cpu()
    x = values.cpu().double()
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float64)
    for i, scale in enumerate(scales.cpu().view(torch.float8_e4m3fn).double()):
        # Clamp before distance calculation so tiny subnormals cannot
        # obscure differences between distant saturated codebook levels.
        normalized = (x.abs() / scale).clamp_max(6)
        expected = order[(normalized[:, None] - levels[order]).abs().argmin(-1)]
        expected |= torch.signbit(x).long() << 3
        assert torch.equal(actual[i], expected.byte()), (i, 'native rounding differs')
    result = dict(status='pass', values=len(values), scales=126, combinations=actual.numel(),
        signed_zero_checked=True, subnormals_checked=True, all_midpoints_checked=True,
        finite_bf16_abs_max=2688, codec_sha256=hashlib.sha256(Path(codec.__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
