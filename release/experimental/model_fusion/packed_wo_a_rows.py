# SPDX-License-Identifier: AGPL-3.0-only
"""Share packed wo_a weight loads across the four speculative rows.

Derived from the recipe's spark_packed_wo_a GEMV. Keep all 16 split-K partials,
the BF16 reconstructed-weight boundary, FP32 multiply/reduction schedule and
original final reducer. Reuse each decoded tile across rows inside one CTA.
No additional persistent storage or changed quantization. Not serving-enabled.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _shared_rows(X, W, S, P, M: tl.constexpr, XS0: tl.constexpr,
                 XS1: tl.constexpr, XS2: tl.constexpr,
                 BN: tl.constexpr, BK: tl.constexpr):
    ns = tl.program_id(0) * BN + tl.arange(0, BN)
    group = tl.program_id(1) // 16
    part = tl.program_id(1) % 16
    ks = part * BK + tl.arange(0, BK)
    # This specialization retains the original one-iteration K loop:
    # K=4096, split=16, BK=256. Load and reconstruct once for all rows.
    w = tl.load(W + (group * 1024 + ns[:, None]) * 4096 + ks[None, :],
                (ns[:, None] < 1024) & (ks[None, :] < 4096), other=0.).to(tl.float32)
    scale = tl.load(S + (group * 1024 + ns[:, None]) * 128 + ks[None, :] // 32,
                    (ns[:, None] < 1024) & (ks[None, :] < 4096), other=127).to(tl.float32)
    restored = (w * tl.exp2(scale - 127.)).to(tl.bfloat16).to(tl.float32)
    for row in tl.static_range(M):
        x = tl.load(X + row * XS0 + group * XS1 + ks * XS2, ks < 4096, other=0.).to(tl.float32)
        acc = tl.full((BN, BK), 0, tl.float32)
        acc += restored * x[None, :]
        value = tl.sum(acc, 1)
        tl.store(P + ((part * M + row) * 4 + group) * 1024 + ns, value, ns < 1024)


def forward(x, weight, scale, *, tile_n=16, warps=4):
    from spark_packed_wo_a import kernels
    if (tile_n not in (8,16,32,64) or warps not in (4,8) or x.ndim != 3 or x.shape[1:] != (4, 4096) or not 1 <= len(x) <= 4
            or x.dtype != torch.bfloat16 or weight.shape != (4096, 4096)
            or weight.dtype != torch.float8_e4m3fn or scale.shape != (4096, 128)
            or scale.dtype != torch.uint8 or not weight.is_contiguous()
            or not scale.is_contiguous() or any(s <= 0 for s in x.stride())
            or any(not t.is_cuda or t.device != x.device or t.requires_grad for t in (x, weight, scale))):
        raise ValueError('Expected bounded packed wo_a CUDA inference inputs')
    out = torch.empty((len(x), 4, 1024), device=x.device, dtype=x.dtype)
    partials = torch.empty((16, len(x), 4, 1024), device=x.device, dtype=torch.float32)
    _shared_rows[(tr.cdiv(1024, tile_n), 64)](x, weight, scale, partials, len(x), *x.stride(),
                           tile_n, 256, num_warps=warps, enable_fp_fusion=False)
    _, _, _, finish, _ = kernels()
    finish[(tr.cdiv(out.numel(), 256),)](partials, out, out.numel(), 16, 256, num_warps=4)
    return out
