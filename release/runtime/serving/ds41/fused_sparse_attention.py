"""Experimental image-safe packed sparse attention; NOT registered in serving.

Read FP8 SWA and FP4 main pages directly. Two passes preserve a global
normalizer and the existing two-term BF16 probability expansion. Queries
stay BF16, dots/partials/LSE stay FP32. No image-window truncation, KV-format
change, query/probability FP8 conversion, or persistent GPU workspace.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _selected(cache, indices, position, length, WIDTH: tl.constexpr,
              CAPACITY: tl.constexpr, PAGE_STRIDE: tl.constexpr,
              STATES: tl.constexpr, FP4: tl.constexpr, SCALE_BYTES: tl.constexpr = 8):
    channel = tl.arange(0, 512)
    slot = tl.load(indices + position, (position < length) & (position < WIDTH), other=-1)
    live = (slot >= 0) & (slot < CAPACITY)
    safe = tl.where(live, slot, 0).to(tl.int64)
    page = (safe // STATES)[:, None] * PAGE_STRIDE
    state = safe % STATES
    if FP4:
        offset = page + state[:, None] * 288
        packed = tl.load(cache + offset + channel[None, :] // 2, live[:, None], other=0)
        code = (packed.to(tl.int32) >> ((channel[None, :] % 2) * 4)) & 15
        magnitude = code & 7
        exponent, mantissa = magnitude >> 1, magnitude & 1
        value = tl.where(exponent == 0, mantissa.to(tl.float32) * .5,
                         (1. + mantissa.to(tl.float32) * .5)
                         * tl.exp2(exponent.to(tl.float32) - 1.))
        value = tl.where((code & 8) != 0, -value, value)
        scale = tl.load(cache + offset + 256 + channel[None, :] // 16,
                        live[:, None], other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        restored = value * scale
    else:
        offset = page + state[:, None] * 576
        nope_mask = live[:, None] & (channel[None, :] < 448)
        raw = tl.load(cache + offset + channel[None, :], nope_mask, other=0)
        quantized = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        exponent = tl.load(cache + page + STATES * 576 + state[:, None] * SCALE_BYTES
                           + channel[None, :] // (512 // SCALE_BYTES), nope_mask, other=127).to(tl.float32) - 127.
        nope = quantized * tl.exp2(exponent)
        rope_mask = live[:, None] & (channel[None, :] >= 448)
        rope_offset = 448 + 2 * tl.maximum(channel - 448, 0)
        low = tl.load(cache + offset + rope_offset[None, :], rope_mask, other=0).to(tl.uint16)
        high = tl.load(cache + offset + rope_offset[None, :] + 1, rope_mask, other=0).to(tl.uint16)
        rope = (low | (high << 8)).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        restored = tl.where(channel[None, :] < 448, nope, rope)
    return tl.where(live[:, None], restored, 0.).to(tl.bfloat16), live, slot >= CAPACITY


@triton.jit
def _normalizer(q, cache, indices, length, maximum, total, error,
                WIDTH: tl.constexpr, CAPACITY: tl.constexpr,
                PAGE_STRIDE: tl.constexpr, STATES: tl.constexpr,
                FP4: tl.constexpr, SCALE: tl.constexpr, BN: tl.constexpr,
                SPLIT, SPLITS: tl.constexpr, SCALE_BYTES: tl.constexpr = 8):
    for begin in range(SPLIT, tl.cdiv(length, BN), SPLITS):
        position = begin * BN + tl.arange(0, BN)
        kv, live, invalid = _selected(cache, indices, position, length, WIDTH,
                                      CAPACITY, PAGE_STRIDE, STATES, FP4, SCALE_BYTES)
        tl.atomic_or(error, 1, mask=tl.sum(invalid.to(tl.int32), 0) > 0, sem="relaxed")
        scores = tl.dot(q, tl.trans(kv)).to(tl.float32) * SCALE
        scores = tl.where(live[None, :], scores, float('-inf'))
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        safe_maximum = tl.where(new_maximum == float('-inf'), 0., new_maximum)
        total = total * tl.exp(maximum - safe_maximum) + tl.sum(tl.exp(scores - safe_maximum[:, None]), 1)
        total = tl.where(new_maximum == float('inf'), 1., total)
        maximum = new_maximum
    return maximum, total


@triton.jit
def _weighted_values(q, cache, indices, length, lse, high, low,
                     WIDTH: tl.constexpr, CAPACITY: tl.constexpr,
                     PAGE_STRIDE: tl.constexpr, STATES: tl.constexpr,
                     FP4: tl.constexpr, SCALE: tl.constexpr, BN: tl.constexpr,
                     SPLIT, SPLITS: tl.constexpr, SCALE_BYTES: tl.constexpr = 8):
    for begin in range(SPLIT, tl.cdiv(length, BN), SPLITS):
        position = begin * BN + tl.arange(0, BN)
        kv, live, invalid = _selected(cache, indices, position, length, WIDTH,
                                      CAPACITY, PAGE_STRIDE, STATES, FP4, SCALE_BYTES)
        scores = tl.dot(q, tl.trans(kv)).to(tl.float32) * SCALE
        p = tl.where(live[None, :] & (tl.abs(lse[:, None]) < float('inf')),
                     tl.exp(scores - lse[:, None]), 0.)
        p_hi = p.to(tl.bfloat16)
        p_lo = (p - p_hi.to(tl.float32)).to(tl.bfloat16)
        high = tl.dot(p_hi, kv, high)
        low = tl.dot(p_lo, kv, low)
    return high, low


@triton.jit
def _attention(query, swa, si, sl, main, ci, cl, sinks, output, normalizers, error,
               HEADS: tl.constexpr, Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr,
               SW: tl.constexpr, SC: tl.constexpr, SP: tl.constexpr, SS: tl.constexpr,
               CW: tl.constexpr, CC: tl.constexpr, CP: tl.constexpr, CS: tl.constexpr,
               MAIN: tl.constexpr, MAIN_FP4: tl.constexpr, SINKS: tl.constexpr,
               SINK_STRIDE: tl.constexpr, SCALE: tl.constexpr,
               BH: tl.constexpr, BN: tl.constexpr, SB: tl.constexpr = 8, CB: tl.constexpr = 8):
    token, tile = tl.program_id(0), tl.program_id(1)
    h = tile * BH + tl.arange(0, BH)
    d = tl.arange(0, 512)
    q = tl.load(query + token * Q0 + h[:, None] * Q1 + d[None, :] * Q2)
    length = tl.minimum(tl.maximum(tl.load(sl + token), 0), SW)
    if SINKS:
        maximum = tl.load(sinks + h * SINK_STRIDE).to(tl.float32)
        total = tl.full((BH,), 1., tl.float32)
    else:
        maximum = tl.full((BH,), float('-inf'), tl.float32)
        total = tl.full((BH,), 0., tl.float32)
    maximum, total = _normalizer(q, swa, si + token * SW, length, maximum, total,
                                error, SW, SC, SP, SS, False, SCALE, BN, 0, 1, SB)
    if MAIN:
        main_length = tl.minimum(tl.maximum(tl.load(cl + token), 0), CW)
        maximum, total = _normalizer(q, main, ci + token * CW, main_length, maximum,
                                    total, error, CW, CC, CP, CS, MAIN_FP4, SCALE, BN, 0, 1, CB)
    lse = tl.where(maximum == float('-inf'), float('-inf'), maximum + tl.log(total))
    high = tl.full((BH, 512), 0., tl.float32)
    low = tl.full((BH, 512), 0., tl.float32)
    high, low = _weighted_values(q, swa, si + token * SW, length, lse, high, low,
                                 SW, SC, SP, SS, False, SCALE, BN, 0, 1, SB)
    if MAIN:
        high, low = _weighted_values(q, main, ci + token * CW, main_length, lse, high, low,
                                     CW, CC, CP, CS, MAIN_FP4, SCALE, BN, 0, 1, CB)
    tl.store(output + (token * HEADS + h[:, None]) * 512 + d[None, :], high + low)
    tl.store(normalizers + token * HEADS + h, lse * 1.4426950408889634)


@triton.jit
def _split_attention(query, swa, si, sl, main, ci, cl, sinks, partial, local_lse, global_lse, error,
                     HEADS: tl.constexpr, Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr,
                     SW: tl.constexpr, SC: tl.constexpr, SP: tl.constexpr, SS: tl.constexpr,
                     CW: tl.constexpr, CC: tl.constexpr, CP: tl.constexpr, CS: tl.constexpr,
                     MAIN: tl.constexpr, MAIN_FP4: tl.constexpr, SINKS: tl.constexpr,
                     SINK_STRIDE: tl.constexpr, SCALE: tl.constexpr,
                     BH: tl.constexpr, BN: tl.constexpr, SPLITS: tl.constexpr, STAGE: tl.constexpr, SB: tl.constexpr = 8, CB: tl.constexpr = 8):
    token, tile, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    h = tile * BH + tl.arange(0, BH)
    d = tl.arange(0, 512)
    q = tl.load(query + token * Q0 + h[:, None] * Q1 + d[None, :] * Q2)
    length = tl.minimum(tl.maximum(tl.load(sl + token), 0), SW)
    if MAIN:
        main_length = tl.minimum(tl.maximum(tl.load(cl + token), 0), CW)
    if STAGE == 0:
        # A sink has mass once globally, even if some splits have no keys.
        if SINKS:
            maximum = tl.where(split == 0, tl.load(sinks + h * SINK_STRIDE).to(tl.float32), float('-inf'))
            total = tl.full((BH,), 1., tl.float32)
        else:
            maximum = tl.full((BH,), float('-inf'), tl.float32)
            total = tl.full((BH,), 0., tl.float32)
        maximum, total = _normalizer(q, swa, si + token * SW, length, maximum, total,
                                    error, SW, SC, SP, SS, False, SCALE, BN, split, SPLITS, SB)
        if MAIN:
            maximum, total = _normalizer(q, main, ci + token * CW, main_length, maximum,
                                        total, error, CW, CC, CP, CS, MAIN_FP4, SCALE, BN, split, SPLITS, CB)
        lse = tl.where(maximum == float('-inf'), float('-inf'), maximum + tl.log(total))
        tl.store(local_lse + (token * HEADS + h) * SPLITS + split, lse)
    else:
        lse = tl.load(global_lse + token * HEADS + h)
        high = tl.full((BH, 512), 0., tl.float32)
        low = tl.full((BH, 512), 0., tl.float32)
        high, low = _weighted_values(q, swa, si + token * SW, length, lse, high, low,
                                     SW, SC, SP, SS, False, SCALE, BN, split, SPLITS, SB)
        if MAIN:
            high, low = _weighted_values(q, main, ci + token * CW, main_length, lse, high, low,
                                         CW, CC, CP, CS, MAIN_FP4, SCALE, BN, split, SPLITS, CB)
        tl.store(partial + ((token * HEADS + h[:, None]) * SPLITS + split) * 512 + d[None, :], high + low)


@triton.jit
def _global_normalizer(local, output, ROWS: tl.constexpr, SPLITS: tl.constexpr):
    row = tl.program_id(0) * 128 + tl.arange(0, 128)
    split = tl.arange(0, SPLITS)
    value = tl.load(local + row[:, None] * SPLITS + split[None, :], row[:, None] < ROWS, other=float('-inf'))
    maximum = tl.max(value, 1)
    safe_maximum = tl.where(maximum == float('-inf'), 0., maximum)
    lse = maximum + tl.log(tl.sum(tl.exp(value - safe_maximum[:, None]), 1))
    lse = tl.where(tl.abs(maximum) == float('inf'), maximum, lse)
    tl.store(output + row, lse, row < ROWS)


@triton.jit
def _sum_partials(partial, global_lse, output, normalizers, SPLITS: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    d = tile * 128 + tl.arange(0, 128)
    split = tl.arange(0, SPLITS)
    values = tl.load(partial + (row * SPLITS + split[:, None]) * 512 + d[None, :])
    tl.store(output + row * 512 + d, tl.sum(values, 0))
    if tile == 0:
        tl.store(normalizers + row, tl.load(global_lse + row) * 1.4426950408889634)


def _validate_segment(query, cache, indices, lengths, formats):
    if (not isinstance(cache, torch.Tensor) or cache.device != query.device
            or cache.dtype != torch.uint8 or cache.ndim != 3
            or cache.shape[1] < 1 or cache.shape[2] not in formats
            or cache.stride(2) != 1 or cache.stride(1) != cache.shape[2]
            or cache.stride(0) < cache.shape[1] * cache.shape[2]):
        raise ValueError('Expected packed page-contiguous cache on the query device')
    if (not isinstance(indices, torch.Tensor) or not isinstance(lengths, torch.Tensor)
            or indices.device != query.device or lengths.device != query.device
            or indices.dtype not in (torch.int32, torch.int64)
            or lengths.dtype not in (torch.int32, torch.int64)
            or indices.ndim not in (2, 3) or indices.shape[0] != query.shape[0]
            or (indices.ndim == 3 and indices.shape[1] != 1)
            or lengths.numel() != query.shape[0]
            or not indices.is_contiguous() or not lengths.is_contiguous()):
        raise ValueError('Expected contiguous per-token integer indices and lengths')


def packed_sparse_attention_with_lse(query, swa_cache, swa_indices, swa_lengths,
                                     workspace=None, *, compressed_cache=None,
                                     compressed_indices=None, compressed_lengths=None,
                                     sinks=None, scale=None, split_k=None):
    """Bounded eager candidate; validates positive slot bounds before returning.

    Negative indices and entries past each row's length are padding, as in
    the current path. Zero-length/no-sink rows return zero and -inf LSE.
    The one host bounds observation means this is not CUDA-graph compatible.
    """
    if (not query.is_cuda or query.dtype != torch.bfloat16 or query.ndim != 3
            or not 0 <= query.shape[0] <= 64 or query.shape[1] not in (32, 64)
            or query.shape[2] != 512 or query.requires_grad):
        raise ValueError('Expected inference BF16 CUDA [0..64 tokens, 32/64 heads, 512]')
    if torch.cuda.is_current_stream_capturing():
        raise ValueError('Bounds-checked eager attention cannot run inside graph capture')
    _validate_segment(query, swa_cache, swa_indices, swa_lengths, (584, 592))
    present = compressed_indices is not None
    if present != (compressed_lengths is not None) or present != (compressed_cache is not None):
        raise ValueError('Compressed cache, indices and lengths must be supplied together')
    if present:
        _validate_segment(query, compressed_cache, compressed_indices, compressed_lengths, (288, 584, 592))
    if sinks is not None and (sinks.device != query.device or sinks.ndim != 1
            or sinks.numel() != query.shape[1] or sinks.dtype not in (torch.float32, torch.bfloat16)
            or sinks.requires_grad):
        raise ValueError('Expected one FP32/BF16 inference sink per head on the query device')
    scale = 512 ** -.5 if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError('Attention scale must be finite')
    if split_k is None:
        # More key parallelism only for small query batches. Large batches
        # already expose many CTAs and retain the no-partial-buffer path.
        split_k = 8 if query.shape[0] <= 2 else 2 if query.shape[0] <= 8 else 1
    if type(split_k) is not int or split_k not in (1, 2, 8):
        raise ValueError('Expected 1, 2 or 8 key splits')
    output = torch.empty(query.shape, device=query.device, dtype=torch.float32)
    lse = torch.empty(query.shape[:2], device=query.device, dtype=torch.float32)
    if not query.shape[0]:
        return output, lse
    main = compressed_cache if present else swa_cache
    ci = compressed_indices if present else swa_indices
    cl = compressed_lengths if present else swa_lengths
    error = torch.zeros((), device=query.device, dtype=torch.int32)
    common = dict(
        HEADS=query.shape[1], Q0=query.stride(0), Q1=query.stride(1), Q2=query.stride(2),
        SW=swa_indices.shape[-1], SC=swa_cache.shape[0] * swa_cache.shape[1],
        SP=swa_cache.stride(0), SS=swa_cache.shape[1], SB=swa_cache.shape[2]-576,
        CW=ci.shape[-1], CC=main.shape[0] * main.shape[1], CP=main.stride(0), CS=main.shape[1],
        CB=8 if main.shape[-1] == 288 else main.shape[-1]-576,
        MAIN=present, MAIN_FP4=main.shape[-1] == 288, SINKS=sinks is not None,
        SINK_STRIDE=sinks.stride(0) if sinks is not None else 1, SCALE=scale,
        BH=16, BN=32, num_warps=8, num_stages=1, enable_fp_fusion=False)
    arguments = (query, swa_cache, swa_indices, swa_lengths, main, ci, cl,
                 sinks if sinks is not None else query)
    if split_k == 1:
        _attention[(query.shape[0], query.shape[1] // 16)](*arguments, output, lse, error, **common)
    else:
        # At the automatic split counts this scratch is at most 2 MiB.
        # Even explicit 8-way/64-token diagnostics remain bounded at 64 MiB.
        partial = torch.empty((*query.shape[:2], split_k, 512), device=query.device, dtype=torch.float32)
        local_lse = torch.empty((*query.shape[:2], split_k), device=query.device, dtype=torch.float32)
        global_lse = torch.empty(query.shape[:2], device=query.device, dtype=torch.float32)
        grid = (query.shape[0], query.shape[1] // 16, split_k)
        stage_args = (*arguments, partial, local_lse, global_lse, error)
        _split_attention[grid](*stage_args, **common, SPLITS=split_k, STAGE=0)
        _global_normalizer[(triton.cdiv(query.shape[0]*query.shape[1], 128),)](
            local_lse, global_lse, ROWS=query.shape[0]*query.shape[1], SPLITS=split_k,
            num_warps=4, enable_fp_fusion=False)
        _split_attention[grid](*stage_args, **common, SPLITS=split_k, STAGE=1)
        _sum_partials[(query.shape[0]*query.shape[1], 4)](
            partial, global_lse, output, lse, SPLITS=split_k, num_warps=4,
            enable_fp_fusion=False)
    if error.item():
        raise ValueError('Sparse slot exceeds allocated packed cache')
    return output, lse
