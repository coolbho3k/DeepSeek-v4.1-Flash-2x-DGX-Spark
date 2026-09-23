# SPDX-License-Identifier: AGPL-3.0-only
"""Candidate: fuse mHC postmix with the next small-row prenorm projection.

Retain the BF16 residual boundary BEFORE the next projection and squared norm,
all 24 FP32 projections, and the qualified split/reduction geometry. The native
Sinkhorn, carried pre-mix and normalized-input consumer remain separate. This
module is an experiment and is not installed in serving.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _post_prenorm(X, R, Post, Comb, F, ResidualOut, Mixes, Squares,
                  ROWS: tl.constexpr, SPLITS: tl.constexpr,
                  BN: tl.constexpr, BK: tl.constexpr, FUSED_POST: tl.constexpr):
    part = tl.program_id(0)
    row = tl.program_id(2)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    local_k = tl.arange(0, BK)
    span = 20480 // SPLITS
    k = part * span + local_k
    valid = local_k < span
    stream = k // 5120
    col = k % 5120
    x = tl.load(X + row * 5120 + col, valid, other=0.).to(tl.float32)
    scale = tl.load(Post + row * 4 + stream, valid, other=0.)
    value = x * scale
    for i in tl.static_range(4):
        residual = tl.load(R + (row * 4 + i) * 5120 + col, valid, other=0.).to(tl.float32)
        weight = tl.load(Comb + row * 16 + i * 4 + stream, valid, other=0.)
        if FUSED_POST:
            value = tl.fma(weight, residual, value)
        else:
            value = value + weight * residual
    # This rounding is part of the existing model, even when the intermediate
    # residual is kept in registers for the following computation.
    stored = value.to(tl.bfloat16)
    current = stored.to(tl.float32)
    if tl.program_id(1) == 0:
        tl.store(ResidualOut + row * 20480 + k, stored, valid)
    f = tl.load(F + n[:, None] * 20480 + k[None, :],
                (n[:, None] < 24) & valid[None, :], other=0.)
    products = tl.sum(f * current[None, :], 1)
    tl.store(Mixes + (part * ROWS + row) * 24 + n, products, n < 24)
    if tl.program_id(1) == 0:
        tl.store(Squares + part * ROWS + row, tl.sum(current * current, 0))


def forward(x, residual, post, comb, fn, splits, *, fused_post=True):
    rows = len(x)
    if (not 1 <= rows <= 4 or x.shape != (rows, 5120)
            or residual.shape != (rows, 4, 5120) or post.shape != (rows, 4, 1)
            or comb.shape != (rows, 4, 4) or fn.shape != (24, 20480)
            or type(splits) is not int or splits not in (4, 16)
            or type(fused_post) is not bool):
        raise ValueError('Expected the fixed small-row mHC contract')
    for tensor, dtype in ((x, torch.bfloat16), (residual, torch.bfloat16),
                          (post, torch.float32), (comb, torch.float32), (fn, torch.float32)):
        if (not tensor.is_cuda or tensor.device != x.device or tensor.dtype != dtype
                or not tensor.is_contiguous() or tensor.requires_grad):
            raise ValueError('Expected contiguous inference CUDA tensors')
    output = torch.empty_like(residual)
    mixes = torch.empty((splits, rows, 24), device=x.device, dtype=torch.float32)
    squares = torch.empty((splits, rows), device=x.device, dtype=torch.float32)
    _post_prenorm[(splits, 6, rows)](x, residual, post, comb, fn, output, mixes,
                                     squares, rows, splits, 4,
                                     tr.next_power_of_2(20480 // splits), fused_post,
                                     num_warps=8, enable_fp_fusion=False)
    return output, mixes, squares
