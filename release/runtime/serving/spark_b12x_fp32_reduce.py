"""Private correction for native B12X's four-slice non-atomic reduction.

The installed policy may select four K slices with TURBO=0, but its named
two-slice reducer silently sums only slices0/1. Retain native FP32 partials
and sum *all* slices in a deterministic tree before one BF16/FP16 rounding.
This module is not enabled in any serving kit yet.
"""
import functools
import hashlib
import importlib
import os
from pathlib import Path

import torch

NATIVE_SHA = 'f92d4e1e73a20dd801db200aaa6d89e7463c8b319ac304a19d2d13b194efec4e'
_installed = None


@functools.cache
def reduction_kernel():
    from vllm.triton_utils import triton, tl

    @triton.jit
    def reduce(P, O, M: tl.constexpr, N: tl.constexpr, S: tl.constexpr,
               PS0: tl.constexpr, PS1: tl.constexpr, PS2: tl.constexpr,
               OS0: tl.constexpr, OS1: tl.constexpr, B: tl.constexpr):
        offsets = tl.program_id(0) * B + tl.arange(0, B)
        rows, cols = offsets // N, offsets % N
        parts = tl.arange(0, S)
        values = tl.load(P + rows[None, :] * PS0 + cols[None, :] * PS1 + parts[:, None] * PS2,
                         mask=rows[None, :] < M, other=0.)
        tl.store(O + rows * OS0 + cols * OS1, tl.sum(values, axis=0), mask=rows < M)

    return triton, reduce


def reduce_partials(partials, out, *, m, n):
    if (partials.ndim != 3 or partials.shape[:2] != (m, n) or partials.shape[2] not in (2, 4)
            or partials.dtype != torch.float32 or out.shape != (m, n, 1)
            or out.dtype not in (torch.bfloat16, torch.float16)
            or partials.device.type != 'cuda' or out.device != partials.device
            or any(s <= 0 for s in (*partials.stride(), *out.stride()))):
        raise ValueError('Unexpected native B12X FP32 split-K reduction layout')
    triton, reduce = reduction_kernel()
    if m*n:
        reduce[(triton.cdiv(m*n, 256),)](partials, out, m, n, partials.shape[2],
            *partials.stride(), out.stride(0), out.stride(1), 256, num_warps=4)


def register():
    global _installed
    from vllm.platforms import current_platform
    native = importlib.import_module('b12x._lib.dense_gemm')
    if os.environ.get('B12X_DENSE_SPLITK_TURBO') != '0' or native._B12X_DENSE_SPLITK_TURBO:
        raise ValueError('FP32 reducer requires native B12X atomic BF16 split-K disabled before import')
    if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() != NATIVE_SHA:
        raise ValueError('Unreviewed native B12X dense implementation')
    if tuple(current_platform.get_device_capability() or ()) != (12, 1):
        raise ValueError('Private B12X reducer requires SM121')
    if _installed is not None:
        if native._reduce_split_k2_bf16 is not reduce_partials:
            raise RuntimeError('Private B12X reducer hook changed')
        return
    _installed = native._reduce_split_k2_bf16
    native._reduce_split_k2_bf16 = reduce_partials
