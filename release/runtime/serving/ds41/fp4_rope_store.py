# SPDX-License-Identifier: Apache-2.0
# RoPE arithmetic adapted from vLLM's fused_compress_quant_cache.py.
"""Fused native V4.1 GPT-J RoPE and reference-format FP4 main-cache store.

Consumes the native compressor's BF16 latent without an intermediate rotated
tensor. Only closed, owned groups are read. SWA and indexer writers are separate.
"""
import torch
import triton
import triton.language as tl

from .fp4_main_kv import MAX_WRITE_ROWS, _indices, _layout, _store_row


@triton.jit
def _rope_insert(latent, positions, cos_sin, cache, slots,
                 CAPACITY: tl.constexpr, POSITION_LIMIT: tl.constexpr,
                 COS_STRIDE: tl.constexpr, PAGE_STRIDE: tl.constexpr,
                 STATES: tl.constexpr, RATIO: tl.constexpr):
    token = tl.program_id(0)
    slot = tl.load(slots + token)
    if slot < 0 or slot >= CAPACITY:
        return
    position = tl.load(positions + token)
    if position < 0 or position >= POSITION_LIMIT:
        return
    if (position + 1) % RATIO != 0:
        return
    normed = tl.load(latent + token.to(tl.int64) * 512 + tl.arange(0, 512)).to(tl.float32)
    even, odd = tl.split(tl.reshape(normed, (256, 2)))
    pair = tl.arange(0, 256) - 224
    cs = cos_sin + (position // RATIO * RATIO) * COS_STRIDE
    c = tl.load(cs + tl.maximum(pair, 0), pair >= 0, other=1.0).to(tl.float32)
    s = tl.load(cs + 32 + tl.maximum(pair, 0), pair >= 0, other=0.0).to(tl.float32)
    # Preserve the native writer's BF16 rounding BEFORE FP4 scale selection.
    row = tl.interleave(even * c - odd * s, odd * c + even * s).to(tl.bfloat16)
    _store_row(row.to(tl.float32), slot, cache, CAPACITY, PAGE_STRIDE, STATES)


def rope_quant_insert(latent, positions, cos_sin_cache, kv_cache, slot_mapping,
                      compress_ratio, fp8_scale=None, *, check_bounds=True):
    """Native insert signature, but exclusively for 288-byte FP4 main pages.

The default checks slot and position bounds on the host. A validated native
allocator may use check_bounds=False; invalid accesses remain kernel-masked.
"""
    _layout(kv_cache)
    if (compress_ratio not in (1, 2) or fp8_scale is not None
            or latent.device != kv_cache.device or latent.dtype != torch.bfloat16
            or latent.ndim != 2 or latent.shape[1] != 512 or not latent.is_contiguous()
            or slot_mapping.ndim != 1 or slot_mapping.numel() > MAX_WRITE_ROWS
            or positions.device != kv_cache.device or positions.ndim != 1
            or positions.dtype not in (torch.int32, torch.int64) or not positions.is_contiguous()
            or slot_mapping.numel() > min(latent.shape[0], positions.numel())
            or cos_sin_cache.device != kv_cache.device or cos_sin_cache.ndim != 2
            or cos_sin_cache.dtype not in (torch.bfloat16, torch.float32)
            or cos_sin_cache.shape[1] != 64 or cos_sin_cache.stride(1) != 1
            or cos_sin_cache.stride(0) < 64):
        raise ValueError('Expected bounded native BF16 latent/RoPE inputs, CR1/CR2, and no global scale')
    slots = _indices(kv_cache, slot_mapping, check_bounds)
    count = slots.numel()
    if check_bounds:
        pos = positions[:count]
        if ((slots >= 0) & ((pos < 0) | (pos >= len(cos_sin_cache)))).any().item():
            raise ValueError('Main-cache position exceeds RoPE table bounds')
    if count:
        _rope_insert[(count,)](latent, positions, cos_sin_cache, kv_cache, slots,
            CAPACITY=kv_cache.shape[0]*kv_cache.shape[1], POSITION_LIMIT=len(cos_sin_cache),
            COS_STRIDE=cos_sin_cache.stride(0), PAGE_STRIDE=kv_cache.stride(0),
            STATES=kv_cache.shape[1], RATIO=compress_ratio, num_warps=4,
            enable_fp_fusion=True)
