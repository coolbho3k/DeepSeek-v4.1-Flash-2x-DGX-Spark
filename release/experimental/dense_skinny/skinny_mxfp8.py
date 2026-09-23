# SPDX-License-Identifier: AGPL-3.0-only
"""Experimental skinny MXFP8 linear for speculative-decode row counts.

Reads the native packed MXFP8 weight values and UE8M0 row scales in place
(no converted copy) and the activation quantized by the same native B12X
MXFP8 quantizer. Each 32-wide scale block is an exact FP8 tensor-core dot
product in FP32; block results are scaled by the two power-of-two scales and
accumulated in FP32. Split-K partials are reduced in FP32 and rounded once
to BF16. Only FP32 summation order differs from the native B12X kernel.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _e8m0(code):
    return tl.exp2(code.to(tl.float32) - 127.0)


@triton.jit
def _skinny(XQ, XS, W, WS, OUT, M, N, K, KB, SPLIT_SPAN,
            MP: tl.constexpr, BN: tl.constexpr, BLOCKS: tl.constexpr, FINAL: tl.constexpr):
    pid_n = tl.program_id(0)
    split = tl.program_id(1)
    rows = pid_n * BN + tl.arange(0, BN)
    live = rows < N
    m = tl.arange(0, MP)
    mlive = m < M
    kk = tl.arange(0, 32)
    acc = tl.zeros((MP, BN), dtype=tl.float32)
    first = split * SPLIT_SPAN
    last = tl.minimum(first + SPLIT_SPAN, KB)
    for kb0 in range(first, last, BLOCKS):
        for j in tl.static_range(BLOCKS):
            kb = kb0 + j
            ok = kb < last
            k = kb * 32 + kk
            x = tl.load(XQ + m[:, None] * K + k[None, :], mask=mlive[:, None] & ok, other=0.)
            w = tl.load(W + rows[None, :] * K + k[:, None], mask=live[None, :] & ok, other=0.)
            xs = _e8m0(tl.load(XS + m * KB + kb, mask=mlive & ok, other=127))
            ws = _e8m0(tl.load(WS + rows * KB + kb, mask=live & ok, other=127))
            d = tl.dot(x, w, out_dtype=tl.float32)
            acc += d * xs[:, None] * ws[None, :]
    if FINAL:
        tl.store(OUT + m[:, None] * N + rows[None, :], acc.to(tl.bfloat16),
                 mask=mlive[:, None] & live[None, :])
    else:
        base = OUT + split * M * N
        tl.store(base + m[:, None] * N + rows[None, :], acc, mask=mlive[:, None] & live[None, :])


@triton.jit
def _reduce(P, OUT, MN, S: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    live = i < MN
    total = tl.zeros((B,), dtype=tl.float32)
    for s in tl.static_range(S):
        total += tl.load(P + s * MN + i, mask=live, other=0.)
    tl.store(OUT + i, total.to(tl.bfloat16), mask=live)


def forward(q_values, q_scales, w_values, w_scales, *, bn=64, blocks=4, splits=1, warps=4, stages=3):
    m, k = q_values.shape
    n = w_values.shape[0]
    kb = k // 32
    if (k % 32 or w_values.shape != (n, k) or q_scales.numel() != m * kb
            or w_scales.numel() != n * kb or not 1 <= m <= 32):
        raise ValueError('Unsupported skinny MXFP8 layout')
    mp = 16 if m <= 16 else 32
    span = triton.cdiv(triton.cdiv(kb, splits), blocks) * blocks
    grid = (triton.cdiv(n, bn), splits)
    out = torch.empty((m, n), device=q_values.device, dtype=torch.bfloat16)
    if splits == 1:
        _skinny[grid](q_values, q_scales, w_values, w_scales, out, m, n, k, kb, span,
                      MP=mp, BN=bn, BLOCKS=blocks, FINAL=True, num_warps=warps, num_stages=stages)
        return out
    partial = torch.empty((splits, m, n), device=q_values.device, dtype=torch.float32)
    _skinny[grid](q_values, q_scales, w_values, w_scales, partial, m, n, k, kb, span,
                  MP=mp, BN=bn, BLOCKS=blocks, FINAL=False, num_warps=warps, num_stages=stages)
    _reduce[(triton.cdiv(m * n, 1024),)](partial, out, m * n, S=splits, B=1024, num_warps=4)
    return out
