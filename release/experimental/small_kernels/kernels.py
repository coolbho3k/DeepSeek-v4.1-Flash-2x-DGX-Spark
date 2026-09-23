# SPDX-License-Identifier: AGPL-3.0-only
"""Decode-row fusions for two small, frequent kernels.

prenorm: the mHC projection/squared-norm producer previously launched one
program per (split, 24-column tile, row) and therefore streamed its FP32
[24, 20480] weight once per row. This variant loops over rows inside the
program, loading each weight tile once. Every row keeps the identical
per-program expression and reduction shape, so outputs are bit-identical.

router: the BF16 [E, 5120] router projection with FP32 output. cuBLAS runs
it as split-K plus a separate reduction kernel at small row counts. This is
one kernel: BF16 inputs, FP32 accumulation, FP32 output. Only FP32
summation order differs.
"""
import triton as tr
import triton.language as tl
import torch


@tr.jit
def _prenorm_rows(X, F, O, S, K: tl.constexpr, SPLITS: tl.constexpr, ROWS: tl.constexpr,
                  BN: tl.constexpr, BK: tl.constexpr):
    part = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    local_k = tl.arange(0, BK)
    span = K // SPLITS
    k = part * span + local_k
    f = tl.load(F + n[:, None] * K + k[None, :],
                (n[:, None] < 24) & (local_k[None, :] < span), other=0.).to(tl.float32)
    for row in tl.static_range(ROWS):
        x = tl.load(X + row * K + k, local_k < span, other=0.).to(tl.float32)
        value = tl.sum(f * x[None, :], 1)
        tl.store(O + (part * ROWS + row) * 24 + n, value, n < 24)
        if tl.program_id(1) == 0:
            tl.store(S + part * ROWS + row, tl.sum(x * x, 0))


def prenorm(x, fn, out, sqrsum, splits, *, tile_n=4, warps=4):
    rows, k = x.shape
    if (not 1 <= rows <= 4 or k not in (5120, 20480) or x.dtype != torch.bfloat16
            or fn.shape != (24, k) or out.shape != (splits, rows, 24) or sqrsum.shape != (splits, rows)):
        raise ValueError('Expected the bounded native small-row mHC producer contract')
    _prenorm_rows[(splits, tr.cdiv(24, tile_n))](x, fn, out, sqrsum, k, splits, rows,
        tile_n, tr.next_power_of_2(k // splits), num_warps=warps, enable_fp_fusion=False)


@tr.jit
def _router(X, W, Y, R, N, K: tl.constexpr, RP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    r = tl.arange(0, RP)
    acc = tl.zeros((RP, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        x = tl.load(X + r[:, None] * K + k[None, :], (r[:, None] < R) & (k[None, :] < K), other=0.)
        w = tl.load(W + n[None, :] * K + k[:, None], (n[None, :] < N) & (k[:, None] < K), other=0.)
        acc += tl.dot(x, w, out_dtype=tl.float32)
    tl.store(Y + r[:, None] * N + n[None, :], acc, (r[:, None] < R) & (n[None, :] < N))


def router(x, weight, *, bn=16, bk=128, warps=4, stages=3):
    rows, k = x.shape
    n = weight.shape[0]
    if (not 1 <= rows <= 32 or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or weight.shape[1] != k or not x.is_contiguous() or not weight.is_contiguous()):
        raise ValueError('Unsupported router projection layout')
    y = torch.empty((rows, n), device=x.device, dtype=torch.float32)
    _router[(tr.cdiv(n, bn),)](x, weight, y, rows, n, k, 16 if rows <= 16 else 32, bn, bk,
                               num_warps=warps, num_stages=stages)
    return y


@tr.jit
def _router_fma(X, W, Y, R, N, K: tl.constexpr, RP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    r = tl.arange(0, RP)
    acc = tl.zeros((RP, BN, BK), dtype=tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        x = tl.load(X + r[:, None] * K + k[None, :], (r[:, None] < R) & (k[None, :] < K), other=0.).to(tl.float32)
        w = tl.load(W + n[:, None] * K + k[None, :], (n[:, None] < N) & (k[None, :] < K), other=0.).to(tl.float32)
        acc += x[:, None, :] * w[None, :, :]
    tl.store(Y + r[:, None] * N + n[None, :], tl.sum(acc, 2), (r[:, None] < R) & (n[None, :] < N))


def router_fma(x, weight, *, bn=4, bk=256, warps=4, stages=2):
    rows, k = x.shape
    n = weight.shape[0]
    if (not 1 <= rows <= 8 or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or weight.shape[1] != k or not x.is_contiguous() or not weight.is_contiguous()):
        raise ValueError('Unsupported router projection layout')
    y = torch.empty((rows, n), device=x.device, dtype=torch.float32)
    rp = 1 if rows == 1 else 2 if rows == 2 else 4 if rows <= 4 else 8
    _router_fma[(tr.cdiv(n, bn),)](x, weight, y, rows, n, k, rp, bn, bk,
                                   num_warps=warps, num_stages=stages)
    return y
