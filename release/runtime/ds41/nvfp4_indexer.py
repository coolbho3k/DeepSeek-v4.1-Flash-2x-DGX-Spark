# SPDX-License-Identifier: AGPL-3.0-only
"""Opt-in NVFP4 index keys for the V4.1 sparse indexer.

DS41_INDEXER_K_FORMAT=nvfp4 stores each 128-value post-RoPE index key as
E2M1 with one E4M3 scale per 16 values (72 bytes). Pages keep the segregated
MXFP4 layout: all packed value rows (64 bytes) first, then all scale rows (8
bytes). vLLM pads indexer pages to 512 bytes, so a 64- or 128-state MXFP4 page
(68-byte rows) already occupies exactly 64 or 128 NVFP4 rows: page size, block
count and KV capacity do not change. Only the writer, readers and query format
do. Cached pages are not interchangeable with MXFP4 pages; the format is fixed
for the life of the process.

Scale selection per 16-value group (keys and NVFP4 queries) is fp4_main_kv's
search: every E4M3 scale from amax/6.5 to amax/2.5, starting from E4M3(amax/6) and
keeping a candidate only if its reconstruction has strictly lower SSE (exact integer
comparison). Every group therefore reconstructs no worse than E4M3(amax/6) or
four-over-six would, and ties keep the amax/6 bytes. The group floor 6 * 2**-9
keeps every scale nonzero. Key RoPE matches spark_indexer_k_math's parity kernel.

Decode queries use vLLM's FP8 indexer query (per-token-per-head power-of-two scale
folded into the head weights), scored by FP8 x NVFP4 in BF16 MMA:
    logits[i, j] = sum_h relu(q[i, h] . k[j]) * weights[i, h],  ks[i] <= j < ke[i]
and -inf elsewhere. E2M1 x E4M3 products and E4M3 queries are exact in BF16.
Prefill queries (quadratic work) are NVFP4 with a power-of-two per-token-head
scale folded into the weights, scored on the block-scaled FP4 tensor cores by
nvfp4-indexer-native (release/runtime/kernels/nvfp4_indexer.cu).
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl

from .fp4_main_kv import _decoded_magnitude, _encode_e2m1, _search_scales

FORMATS = ('mxfp4', 'nvfp4')
ROW_BYTES = 72
VALUE_BYTES = 64
SCALE_BYTES = 8
HEAD_DIM = 128
HEADS = 32


def configured_format():
    value = os.environ.get('DS41_INDEXER_K_FORMAT', 'mxfp4')
    if value not in FORMATS:
        raise ValueError('DS41_INDEXER_K_FORMAT must be mxfp4 or nvfp4')
    if value == 'nvfp4' and any(os.environ.get(k) != '1' for k in (
            'DS41_ENABLE_DCP2', 'DS41_ENABLE_FP4_MAIN_KV', 'DS41_ENABLE_FP4_INDEXER',
            'DS41_ENABLE_INDEXER_K_PARITY')):
        raise ValueError('NVFP4 index keys require the full-FP4 DCP runtime with index-key parity')
    return value


DECODE_QUERIES = ('fp8', 'nvfp4')


def configured_decode_query():
    value = os.environ.get('DS41_INDEXER_DECODE_QUERY', 'fp8')
    if value not in DECODE_QUERIES:
        raise ValueError('DS41_INDEXER_DECODE_QUERY must be fp8 or nvfp4')
    if value == 'nvfp4' and os.environ.get('DS41_INDEXER_K_FORMAT', 'mxfp4') != 'nvfp4':
        raise ValueError('NVFP4 decode queries require DS41_INDEXER_K_FORMAT=nvfp4')
    return value


# Selected once per process before any hook, cache or graph exists.
FORMAT = configured_format()
ENABLED = FORMAT == 'nvfp4'
DECODE_QUERY = configured_decode_query()


def require_stable_format():
    if configured_format() != FORMAT or configured_decode_query() != DECODE_QUERY:
        raise RuntimeError('Indexer key format cannot change after startup')


# ---------------------------------------------------------------- scale selection

@triton.jit
def _nvfp4_encode(groups):
    """[8, 16] FP32 (BF16-exact) values -> (E2M1 codes [8, 16] int32, E4M3 scale bytes [8] uint8)."""
    amax = tl.maximum(tl.max(tl.abs(groups), axis=1), 6.0 * (2.0 ** -9))
    scales6 = tl.div_rn(amax, 6.0).to(tl.float8e4nv)
    return _search_scales(groups, amax, _encode_e2m1(groups, scales6), scales6)


# ---------------------------------------------------------------- writer

@triton.jit
def _store_kernel(k, k_stride, positions, cos_sin, cos_sin_stride, cache, slots,
                  STATES: tl.constexpr, PAGE_STRIDE: tl.constexpr,
                  COMPRESS_RATIO: tl.constexpr):
    token = tl.program_id(0)
    slot = tl.load(slots + token)
    if slot < 0:
        return
    position = tl.load(positions + token)
    # Only the last token of a group publishes that group's key.
    if (position + 1) % COMPRESS_RATIO != 0:
        return
    x = tl.load(k + token * k_stride + tl.arange(0, 128)).to(tl.float32)
    even, odd = tl.split(tl.reshape(x, (64, 2)))
    rope = tl.arange(0, 64) - 32
    is_rope = rope >= 0
    column = tl.maximum(rope, 0)
    # A latent stands for the first token of its group.
    base = cos_sin + ((position // COMPRESS_RATIO) * COMPRESS_RATIO) * cos_sin_stride
    cos = tl.load(base + column, mask=is_rope, other=1.0).to(tl.float32)
    sin = tl.load(base + 32 + column, mask=is_rope, other=0.0).to(tl.float32)
    # Same contraction and BF16 round trip as the parity writer.
    new_even = tl.fma(-odd, sin, even * cos).to(tl.bfloat16).to(tl.float32)
    new_odd = tl.fma(even, sin, odd * cos).to(tl.bfloat16).to(tl.float32)
    groups = tl.reshape(tl.interleave(new_even, new_odd), (8, 16))
    codes, scales = _nvfp4_encode(groups)
    low, high = tl.split(tl.reshape(codes, (64, 2)))
    packed = (low | (high << 4)).to(tl.uint8)
    page = cache + (slot // STATES).to(tl.int64) * PAGE_STRIDE
    row = slot % STATES
    tl.store(page + row * 64 + tl.arange(0, 64), packed)
    tl.store(page + STATES * 64 + row * 8 + tl.arange(0, 8), scales)


def _pages(cache):
    if (not cache.is_cuda or cache.dtype != torch.uint8 or cache.ndim != 3
            or cache.shape[1] not in (64, 128) or cache.shape[2] != ROW_BYTES
            or cache.stride(2) != 1 or cache.stride(1) != ROW_BYTES
            or cache.stride(0) < cache.shape[1] * ROW_BYTES or cache.stride(0) % 16):
        raise ValueError('Expected NVFP4 index pages [pages, 64|128, 72]')


def store(normalized, positions, cos_sin, cache, slots, *, compress_ratio):
    """k_norm output (BF16 [rows, 128]) -> RoPE -> NVFP4 -> paged store; -1 slots skip.

    Row count is bounded by the caller (spark_indexer_k_math.MAX_ROWS).
    """
    _pages(cache)
    rows = slots.numel()
    if (normalized.dtype != torch.bfloat16 or normalized.ndim != 2 or normalized.shape[1] != 128
            or normalized.stride(1) != 1 or normalized.shape[0] < rows
            or positions.numel() < rows or compress_ratio not in (1, 2)
            or cos_sin.ndim != 2 or cos_sin.shape[1] != 64 or cos_sin.stride(1) != 1
            or any(t.device != cache.device for t in (normalized, positions, cos_sin, slots))):
        raise ValueError('Invalid NVFP4 index-key write')
    if rows:
        _store_kernel[(rows,)](normalized, normalized.stride(0), positions, cos_sin,
            cos_sin.stride(0), cache, slots, STATES=cache.shape[1], PAGE_STRIDE=cache.stride(0),
            COMPRESS_RATIO=compress_ratio, num_warps=1, enable_fp_fusion=False)


# ---------------------------------------------------------------- gathers

@triton.jit
def _gather_requests_kernel(cache, table, table_stride, cu_lens, requests, values, scales, total,
                            STATES: tl.constexpr, PAGE_STRIDE: tl.constexpr,
                            PAGES: tl.constexpr, COLUMNS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = i < total
    request = tl.zeros((BLOCK,), tl.int32)
    for r in range(1, requests):
        request += (tl.load(cu_lens + r) <= i).to(tl.int32)
    local = i - tl.load(cu_lens + request, live, other=0)
    column = local // STATES
    ok = live & (column < COLUMNS)
    page = tl.load(table + request.to(tl.int64) * table_stride + column, ok, other=-1).to(tl.int64)
    ok = ok & (page >= 0) & (page < PAGES)
    base = tl.where(ok, page, 0) * PAGE_STRIDE
    row = local % STATES
    byte = tl.arange(0, 64)
    tl.store(values + i[:, None] * 64 + byte[None, :],
             tl.load(cache + base[:, None] + row[:, None] * 64 + byte[None, :], ok[:, None], other=0),
             live[:, None])
    scale = tl.arange(0, 8)
    tl.store(scales + i[:, None] * 8 + scale[None, :],
             tl.load(cache + base[:, None] + STATES * 64 + row[:, None] * 8 + scale[None, :],
                     ok[:, None], other=0), live[:, None])


def gather_requests(cache, values, scales, block_table, cu_seq_lens):
    """Drop-in for ops.cp_gather_indexer_k_quant_cache on NVFP4 pages.

    Request r's keys cu[r]..cu[r+1] come from its block-table row in page order.
    Rows past the table or naming invalid pages gather zeros (score 0, never NaN).
    """
    if cache.ndim == 4:
        cache = cache.squeeze(-2)
    _pages(cache)
    total = values.shape[0]
    if (values.dtype != torch.uint8 or values.shape[1:] != (64,) or not values.is_contiguous()
            or scales.dtype != torch.uint8 or scales.shape != (total, 8) or not scales.is_contiguous()
            or block_table.ndim != 2 or block_table.dtype not in (torch.int32, torch.int64)
            or block_table.stride(1) != 1 or cu_seq_lens.dtype not in (torch.int32, torch.int64)
            or cu_seq_lens.numel() != block_table.shape[0] + 1):
        raise ValueError('Invalid NVFP4 index-key gather workspace or metadata')
    if total:
        _gather_requests_kernel[(triton.cdiv(total, 32),)](cache, block_table, block_table.stride(0),
            cu_seq_lens, block_table.shape[0], values, scales, total, STATES=cache.shape[1],
            PAGE_STRIDE=cache.stride(0), PAGES=cache.shape[0], COLUMNS=block_table.shape[1],
            BLOCK=32, num_warps=4)


def workspace_shapes(total_seq_lens, head_dim, fp8_dtype, use_fp4_cache):
    """Replacement for vLLM's _gather_workspace_shapes: 64 value + 8 scale bytes per key."""
    if not use_fp4_cache or head_dim != HEAD_DIM:
        raise ValueError('NVFP4 index keys require the FP4 indexer route and 128-dim keys')
    return (((total_seq_lens, 64), torch.uint8), ((total_seq_lens, 8), torch.uint8))


def quant_view(kv_cache, head_dim, use_fp4_cache):
    """Replacement for vLLM's kv_cache_as_quant_view: [pages, states, 1, 72] over padded pages."""
    if not use_fp4_cache or head_dim != HEAD_DIM:
        raise ValueError('NVFP4 index keys require the FP4 indexer route and 128-dim keys')
    _pages(kv_cache)
    pages, states, _ = kv_cache.shape
    return torch.as_strided(kv_cache, size=(pages, states, 1, ROW_BYTES),
                            stride=(kv_cache.stride(0), ROW_BYTES, ROW_BYTES, 1))


# ---------------------------------------------------------------- scoring

@triton.jit
def _logits_kernel(q, q_row, q_head, weights, w_row, values, scales, starts, ends,
                   out, out_row, M, N, BM: tl.constexpr, BN: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    in_rows = rows < M
    in_cols = cols < N
    lo = tl.load(starts + rows, in_rows, other=0)
    hi = tl.load(ends + rows, in_rows, other=0)
    live = (in_rows[:, None] & in_cols[None, :]
            & (cols[None, :] >= lo[:, None]) & (cols[None, :] < hi[:, None]))
    result = tl.full((BM, BN), float('-inf'), tl.float32)
    if tl.max(tl.max(live.to(tl.int32), axis=1), axis=0) > 0:
        byte = tl.arange(0, 64)
        packed = tl.load(values + cols[:, None].to(tl.int64) * 64 + byte[None, :],
                         in_cols[:, None], other=0).to(tl.int32)
        codes = tl.interleave(packed & 15, packed >> 4)
        magnitude = _decoded_magnitude(codes)
        element = tl.where((codes & 8) != 0, -magnitude, magnitude)
        group = tl.load(scales + cols[:, None].to(tl.int64) * 8 + tl.arange(0, 8)[None, :],
                        in_cols[:, None], other=0)
        group = group.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        key = tl.reshape(tl.reshape(element, (BN, 8, 16)) * group[:, :, None], (BN, 128))
        head = tl.arange(0, 32)
        dim = tl.arange(0, 128)
        query = tl.load(q + rows[:, None, None].to(tl.int64) * q_row + head[None, :, None] * q_head
                        + dim[None, None, :], in_rows[:, None, None], other=0.0)
        query = tl.reshape(query.to(tl.bfloat16), (BM * 32, 128))
        score = tl.dot(query, tl.trans(key.to(tl.bfloat16)))
        score = tl.reshape(tl.maximum(score, 0.0), (BM, 32, BN))
        weight = tl.load(weights + rows[:, None] * w_row + head[None, :], in_rows[:, None], other=0.0)
        result = tl.where(live, tl.sum(score * weight[:, :, None], axis=1), result)
    tl.store(out + rows[:, None].to(tl.int64) * out_row + cols[None, :], result,
             in_rows[:, None] & in_cols[None, :])


def _fp8_query(q):
    if q.dtype in (torch.int8, torch.uint8):
        q = q.view(torch.float8_e4m3fn)
    if q.dtype != torch.float8_e4m3fn or q.shape[-2:] != (HEADS, HEAD_DIM) or q.stride(-1) != 1:
        raise ValueError('Expected FP8 E4M3 indexer queries [..., 32, 128]')
    return q


def mqa_logits(q, weights, key_values, key_scales, starts, ends, out=None):
    """FP8 query x NVFP4 key logits [M, N]; -inf outside [starts, ends) per row."""
    q = _fp8_query(q)
    if key_scales.dtype == torch.int32:
        key_scales = key_scales.view(torch.uint8)
    key_values = key_values.view(torch.uint8)
    m, n = q.shape[0], key_values.shape[0]
    if (q.ndim != 3 or q.stride(1) != HEAD_DIM or weights.shape != (m, HEADS)
            or weights.dtype != torch.float32 or weights.stride(1) != 1
            or key_values.shape != (n, 64) or not key_values.is_contiguous()
            or key_scales.shape != (n, 8) or not key_scales.is_contiguous()
            or starts.shape != (m,) or ends.shape != (m,)
            or starts.dtype != torch.int32 or ends.dtype != torch.int32
            or any(t.device != q.device for t in (weights, key_values, key_scales, starts, ends))):
        raise ValueError('Invalid NVFP4 MQA logits inputs')
    if out is None:
        out = torch.empty((m, n), device=q.device, dtype=torch.float32)
    elif out.shape != (m, n) or out.dtype != torch.float32 or out.stride(1) != 1:
        raise ValueError('Invalid NVFP4 logits output buffer')
    if m and n:
        # Two query rows (64 MMA rows) x 128 keys: best of a GB10 sweep for decode,
        # 24-row DSpark verification and prefill (larger tiles spill registers).
        _logits_kernel[(triton.cdiv(m, 2), triton.cdiv(n, 128))](q, q.stride(0), q.stride(1),
            weights, weights.stride(0), key_values, key_scales, starts, ends, out, out.stride(0),
            m, n, BM=2, BN=128, num_warps=4)
    return out


def mqa_logits_native_signature(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=False):
    """Adapter with vLLM's fp8_fp4_mqa_logits call shape (q=(values, None), kv=(k, s))."""
    values, q_scale = q
    if q_scale is not None:
        raise ValueError('NVFP4 index keys use FP8 queries without companion scales')
    keys, key_scales = kv
    return mqa_logits(values, weights, keys, key_scales, cu_seqlen_ks.to(torch.int32),
                      cu_seqlen_ke.to(torch.int32))


# ---------------------------------------------------------------- decode scorers

@triton.jit
def _gather_pages_kernel(cache, table, start, count, values, scales,
                         STATES: tl.constexpr, PAGE_STRIDE: tl.constexpr,
                         TABLE_STRIDE: tl.constexpr, PAGES: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    position = start + i
    page = tl.load(table + (position // STATES) * TABLE_STRIDE, i < count, other=-1).to(tl.int64)
    ok = (i < count) & (page >= 0) & (page < PAGES)
    base = tl.where(ok, page, 0) * PAGE_STRIDE
    row = position % STATES
    byte = tl.arange(0, 64)
    tl.store(values + i[:, None] * 64 + byte[None, :],
             tl.load(cache + base[:, None] + row[:, None] * 64 + byte[None, :], ok[:, None], other=0),
             (i < count)[:, None])
    scale = tl.arange(0, 8)
    tl.store(scales + i[:, None] * 8 + scale[None, :],
             tl.load(cache + base[:, None] + STATES * 64 + row[:, None] * 8 + scale[None, :],
                     ok[:, None], other=0), (i < count)[:, None])


def _check_view(kv):
    if (not kv.is_cuda or kv.dtype != torch.uint8 or kv.ndim != 4
            or kv.shape[2:] != (1, ROW_BYTES) or kv.shape[1] not in (64, 128)
            or kv.stride(1) != ROW_BYTES or kv.stride(-1) != 1
            or kv.stride(0) < kv.shape[1] * ROW_BYTES or kv.stride(0) % 16):
        raise ValueError('Expected NVFP4 index pages viewed as [pages, 64|128, 1, 72]')


def paged_logits(q, kv, weights, lengths, table, schedule_metadata, *,
                 max_model_len, clean_logits=False, indices=None, state_chunk=8192):
    """Eager DCP decode scorer (NVFP4 counterpart of dcp_indexer_mxfp4.paged_logits)."""
    values, q_scale = q
    values = _fp8_query(values)
    if (values.ndim != 4 or values.shape[1] != 1 or q_scale is not None or indices is not None
            or lengths.shape != (values.shape[0], 1)
            or type(state_chunk) is not int or not 1 <= state_chunk <= 8192
            or type(max_model_len) is not int or not 1 <= max_model_len <= 1048576):
        raise ValueError('Expected bounded FP8 next_n1 indexer queries and NVFP4 context')
    _check_view(kv)
    if (weights.shape != (values.shape[0], HEADS) or weights.dtype != torch.float32
            or table.ndim != 2 or table.shape[0] != values.shape[0]
            or table.dtype not in (torch.int32, torch.int64)
            or lengths.dtype not in (torch.int32, torch.int64)
            or any(x.device != kv.device for x in (values, weights, lengths, table))):
        raise ValueError('Invalid NVFP4 weights, lengths, table or devices')
    counts = lengths[:, 0].cpu().tolist()
    if any(n < 0 or n > max_model_len for n in counts):
        raise ValueError('Local context exceeds bounded logits allocation')
    required = [(n + kv.shape[1] - 1) // kv.shape[1] for n in counts]
    columns = max(required, default=0)
    if columns > table.shape[1]:
        raise ValueError('Indexer block table is too short')
    if columns:
        used = (torch.arange(columns, device=kv.device)[None, :]
                < torch.tensor(required, device=kv.device)[:, None])
        page_ids = table[:, :columns]
        if (used & ((page_ids < 0) | (page_ids >= kv.shape[0]))).any().item():
            raise ValueError('Invalid physical indexer page ID')
    output = torch.full((len(counts), max_model_len), -torch.inf, device=kv.device, dtype=torch.float32)
    capacity = min(state_chunk, max(counts, default=0))
    if not capacity:
        return output
    keys = torch.empty((capacity, 64), device=kv.device, dtype=torch.uint8)
    key_scales = torch.empty((capacity, 8), device=kv.device, dtype=torch.uint8)
    zero = torch.zeros(1, device=kv.device, dtype=torch.int32)
    for request, length in enumerate(counts):
        query = values[request, 0:1].contiguous()
        weight = weights[request:request + 1].contiguous()
        for start in range(0, length, capacity):
            n = min(capacity, length - start)
            _gather_pages_kernel[(triton.cdiv(n, 32),)](kv, table[request], start, n, keys, key_scales,
                STATES=kv.shape[1], PAGE_STRIDE=kv.stride(0), TABLE_STRIDE=table.stride(1),
                PAGES=kv.shape[0], BLOCK=32, num_warps=4)
            mqa_logits(query, weight, keys[:n], key_scales[:n], zero,
                       torch.full((1,), n, device=kv.device, dtype=torch.int32),
                       out=output[request:request + 1, start:start + n])
    return output


@triton.jit
def _graph_gather_kernel(cache, table, lengths, values, scales, errors,
                         N: tl.constexpr, NL: tl.constexpr, CAP: tl.constexpr,
                         STATES: tl.constexpr, PAGE_STRIDE: tl.constexpr,
                         TABLE_STRIDE: tl.constexpr, COLUMNS: tl.constexpr,
                         LENGTH_STRIDE: tl.constexpr, PAGES: tl.constexpr, BLOCK: tl.constexpr):
    # Same masking and error codes as dcp_indexer_graph._gather; 8 scale bytes per key.
    lane = tl.arange(0, NL)
    sizes = tl.load(lengths + lane * LENGTH_STRIDE, lane < N, other=0)
    bad_length = tl.sum(((lane < N) & ((sizes < 0) | (sizes > CAP))).to(tl.int32), 0) > 0
    count = tl.minimum(tl.maximum(tl.max(sizes, 0), 0), CAP)
    if tl.program_id(0) == 0 and bad_length:
        tl.atomic_or(errors, 1)
    start = tl.program_id(0) * BLOCK
    if start >= count:
        return
    i = start + tl.arange(0, BLOCK)
    column = i // STATES
    live = (i < count) & (i < CAP)
    in_table = live & (column < COLUMNS)
    page = tl.load(table + tl.where(in_table, column, 0) * TABLE_STRIDE, in_table, other=-1).to(tl.int64)
    valid = in_table & (page >= 0) & (page < PAGES)
    flags = tl.where(tl.sum((live & ~in_table).to(tl.int32), 0) > 0, 2, 0)
    flags |= tl.where(tl.sum((in_table & ~valid).to(tl.int32), 0) > 0, 4, 0)
    if flags != 0:
        tl.atomic_or(errors, flags)
    base = tl.where(valid, page, 0) * PAGE_STRIDE
    row = i % STATES
    byte = tl.arange(0, 64)
    tl.store(values + i[:, None] * 64 + byte[None, :],
             tl.load(cache + base[:, None] + row[:, None] * 64 + byte[None, :], valid[:, None], other=0),
             live[:, None])
    scale = tl.arange(0, 8)
    tl.store(scales + i[:, None] * 8 + scale[None, :],
             tl.load(cache + base[:, None] + STATES * 64 + row[:, None] * 8 + scale[None, :],
                     valid[:, None], other=0), live[:, None])


@triton.jit
def _paged_keys(cache, table, request, t_req, t_col, errors, cols, needed,
                STATES: tl.constexpr, PAGE_STRIDE: tl.constexpr, COLUMNS: tl.constexpr,
                PAGES: tl.constexpr, BN: tl.constexpr):
    """BF16 NVFP4 keys [BN, 128] read in place from pages; the gather's masking and error codes.

    Keys the table does not cover, or on invalid pages, read as zero (score 0, never NaN).
    """
    column = cols // STATES
    in_table = needed & (column < COLUMNS)
    page = tl.load(table + request * t_req + tl.where(in_table, column, 0) * t_col,
                   in_table, other=-1).to(tl.int64)
    valid = in_table & (page >= 0) & (page < PAGES)
    flags = tl.where(tl.sum((needed & ~in_table).to(tl.int32), 0) > 0, 2, 0)
    flags |= tl.where(tl.sum((in_table & ~valid).to(tl.int32), 0) > 0, 4, 0)
    if flags != 0:
        tl.atomic_or(errors + request, flags)
    base = tl.where(valid, page, 0) * PAGE_STRIDE
    slot = (cols % STATES).to(tl.int64)
    packed = tl.load(cache + base[:, None] + slot[:, None] * 64 + tl.arange(0, 64)[None, :],
                     valid[:, None], other=0).to(tl.int32)
    codes = tl.interleave(packed & 15, packed >> 4)
    magnitude = _decoded_magnitude(codes)
    element = tl.where((codes & 8) != 0, -magnitude, magnitude)
    group = tl.load(cache + base[:, None] + STATES * 64 + slot[:, None] * 8 + tl.arange(0, 8)[None, :],
                    valid[:, None], other=0)
    group = group.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    key = tl.reshape(tl.reshape(element, (BN, 8, 16)) * group[:, :, None], (BN, 128))
    return key.to(tl.bfloat16)


@triton.jit
def _paged_logits_kernel(q, q_req, q_row, q_head, weights, w_req, w_row, lengths, l_req, l_row,
                         cache, table, t_req, t_col, errors, out, out_row,
                         N: tl.constexpr, NL: tl.constexpr, PAIRS: tl.constexpr, CAP: tl.constexpr,
                         STATES: tl.constexpr, PAGE_STRIDE: tl.constexpr, COLUMNS: tl.constexpr,
                         PAGES: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    # _graph_gather_kernel + _logits_kernel in one pass: the same row pairs, key tiles,
    # BF16 MMA and head reduction, without staging every key through a workspace.
    request = tl.program_id(0) // PAIRS
    rows = (tl.program_id(0) % PAIRS) * BM + tl.arange(0, BM)
    in_rows = rows < N
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    in_cols = cols < CAP
    lane = tl.arange(0, NL)
    sizes = tl.load(lengths + request * l_req + lane * l_row, lane < N, other=0)
    if tl.program_id(1) == 0:
        if tl.sum(((lane < N) & ((sizes < 0) | (sizes > CAP))).to(tl.int32), 0) > 0:
            tl.atomic_or(errors + request, 1)
    count = tl.minimum(tl.maximum(tl.max(sizes, 0), 0), CAP)
    hi = tl.load(lengths + request * l_req + rows * l_row, in_rows, other=0)
    hi = tl.minimum(tl.maximum(hi, 0), CAP)
    live = in_rows[:, None] & in_cols[None, :] & (cols[None, :] < hi[:, None])
    result = tl.full((BM, BN), float('-inf'), tl.float32)
    if tl.max(tl.max(live.to(tl.int32), axis=1), axis=0) > 0:
        key = _paged_keys(cache, table, request, t_req, t_col, errors, cols, in_cols & (cols < count),
                          STATES, PAGE_STRIDE, COLUMNS, PAGES, BN)
        head = tl.arange(0, 32)
        dim = tl.arange(0, 128)
        query = tl.load(q + request * q_req + rows[:, None, None].to(tl.int64) * q_row
                        + head[None, :, None] * q_head + dim[None, None, :], in_rows[:, None, None], other=0.0)
        query = tl.reshape(query.to(tl.bfloat16), (BM * 32, 128))
        score = tl.dot(query, tl.trans(key))
        score = tl.reshape(tl.maximum(score, 0.0), (BM, 32, BN))
        weight = tl.load(weights + request * w_req + rows[:, None] * w_row + head[None, :],
                         in_rows[:, None], other=0.0)
        result = tl.where(live, tl.sum(score * weight[:, :, None], axis=1), result)
    tl.store(out + (request * N + rows)[:, None].to(tl.int64) * out_row + cols[None, :], result,
             in_rows[:, None] & in_cols[None, :])


@triton.jit
def _candidate_logits_kernel(q, q_req, q_row, q_head, weights, w_req, w_row, lengths, l_req, l_row,
                             cache, table, t_req, t_col, candidates, c_row, errors, out, out_row,
                             N: tl.constexpr, CAP: tl.constexpr, K: tl.constexpr, NB: tl.constexpr,
                             LOCAL: tl.constexpr, STATES: tl.constexpr, PAGE_STRIDE: tl.constexpr,
                             COLUMNS: tl.constexpr, PAGES: tl.constexpr, BN: tl.constexpr):
    # One query row x BN candidate keys. Global candidate block b holds this rank's
    # local columns [b * LOCAL, (b + 1) * LOCAL) (DCP interleave 1, even block size).
    # The row is scored twice as a BM=2 pair so the MMA tile and head reduction match
    # _paged_logits_kernel exactly: candidate scores are bitwise the full scorer's.
    row = tl.program_id(0)
    request = row // N
    local_row = row % N
    slot = tl.program_id(1) * BN + tl.arange(0, BN)
    chosen = slot // LOCAL
    block = tl.load(candidates + row * c_row + chosen, chosen < K, other=-1).to(tl.int64)
    raw = tl.load(lengths + request * l_req + local_row * l_row)
    if tl.program_id(1) == 0:
        if (raw < 0) | (raw > CAP):
            tl.atomic_or(errors + request, 1)
    hi = tl.minimum(tl.maximum(raw, 0), CAP)
    live = (chosen < K) & (block >= 0) & (block < NB) & (block * LOCAL + slot % LOCAL < hi)
    if tl.max(live.to(tl.int32), axis=0) > 0:
        cols = tl.where(live, block * LOCAL + slot % LOCAL, 0)
        key = _paged_keys(cache, table, request, t_req, t_col, errors, cols, live,
                          STATES, PAGE_STRIDE, COLUMNS, PAGES, BN)
        pair = tl.arange(0, 2)
        head = tl.arange(0, 32)
        dim = tl.arange(0, 128)
        query = tl.load(q + request * q_req + local_row * q_row + pair[:, None, None] * 0
                        + head[None, :, None] * q_head + dim[None, None, :])
        query = tl.reshape(query.to(tl.bfloat16), (64, 128))
        score = tl.dot(query, tl.trans(key))
        score = tl.reshape(tl.maximum(score, 0.0), (2, 32, BN))
        weight = tl.load(weights + request * w_req + local_row * w_row + pair[:, None] * 0 + head[None, :])
        both = tl.sum(score * weight[:, :, None], axis=1)
        result = tl.max(tl.where(pair[:, None] == 0, both, float('-inf')), axis=0)
        tl.store(out + row.to(tl.int64) * out_row + cols, result, live)


def _graph_decode_layout(q, kv, weights, lengths, table, max_model_len, indices, state_chunk):
    values, q_scale = q
    values = _fp8_query(values)
    if (values.ndim != 4 or not 1 <= values.shape[0] <= 24
            or values.shape[0] * values.shape[1] > 24 or not 1 <= values.shape[1] <= 4
            or q_scale is not None or indices is not None or lengths.shape != values.shape[:2]
            or type(max_model_len) is not int or not 1 <= max_model_len <= 1048576
            or type(state_chunk) is not int or not 1 <= state_chunk <= 8192):
        raise ValueError(f'Expected at most24 FP8 query rows: values={values.shape}, lengths={lengths.shape}')
    batch, next_n = values.shape[:2]
    _check_view(kv)
    if (weights.shape not in ((batch * next_n, HEADS), (batch, next_n, HEADS))
            or weights.dtype != torch.float32 or table.ndim != 2 or table.shape[0] != batch
            or table.dtype not in (torch.int32, torch.int64)
            or lengths.dtype not in (torch.int32, torch.int64)
            or any(x.device != kv.device for x in (values, weights, lengths, table))
            or any(s < 0 for x in (values, weights, lengths, table) for s in x.stride())):
        raise ValueError('Invalid NVFP4 graph page/query layout or metadata')
    weights = weights.reshape(batch, next_n, HEADS)
    return values, (weights if weights.stride(2) == 1 else weights.contiguous())


_ERRORS = ((1, 'Local context exceeds bounded logits allocation'),
           (2, 'Indexer block table is too short'), (4, 'Invalid physical indexer page ID'))


def graph_paged_logits(q, kv, weights, lengths, table, schedule_metadata, *,
                       max_model_len, clean_logits=False, indices=None, state_chunk=8192):
    """Capture-safe decode/DSpark scorer (NVFP4 counterpart of dcp_indexer_graph.paged_logits).

    Scores keys in place from their pages (no key workspace). Device lengths bound
    each row's live columns and -inf tail; no host readback inside capture. Invalid
    metadata is masked and reported through the owner, exactly as the gathered scorer.
    """
    from .graph_validation import check_flags, require_capture_owner
    require_capture_owner()
    values, weights = _graph_decode_layout(q, kv, weights, lengths, table, max_model_len, indices, state_chunk)
    batch, next_n = values.shape[:2]
    output = torch.empty((batch * next_n, max_model_len), device=kv.device, dtype=torch.float32)
    errors = torch.zeros(batch, device=kv.device, dtype=torch.int32)
    pairs = triton.cdiv(next_n, 2)
    _paged_logits_kernel[(batch * pairs, triton.cdiv(max_model_len, 128))](
        values, values.stride(0), values.stride(1), values.stride(2), weights, weights.stride(0),
        weights.stride(1), lengths, lengths.stride(0), lengths.stride(1), kv, table, table.stride(0),
        table.stride(1), errors, output, output.stride(0), N=next_n, NL=triton.next_power_of_2(next_n),
        PAIRS=pairs, CAP=max_model_len, STATES=kv.shape[1], PAGE_STRIDE=kv.stride(0),
        COLUMNS=table.shape[1], PAGES=kv.shape[0], BM=2, BN=128, num_warps=4)
    check_flags(errors, _ERRORS)
    return output


def graph_candidate_logits(q, kv, weights, lengths, table, candidates, *, block_size, rank, world,
                           max_model_len, indices=None, state_chunk=8192):
    """Capture-safe decode scorer for indexers that consume two-level candidate blocks.

    Computes graph_paged_logits' values only at this rank's columns of each row's
    candidate blocks ([rows, K] global block ids, -1 padding). Every other column is
    left unwritten: the caller's apply_candidate_mask must follow, and it sets every
    non-candidate or past-length column to -inf (it never keeps an unwritten one).
    """
    from .graph_validation import check_flags, require_capture_owner
    require_capture_owner()
    values, weights = _graph_decode_layout(q, kv, weights, lengths, table, max_model_len, indices, state_chunk)
    batch, next_n = values.shape[:2]
    rows = batch * next_n
    if (world != 2 or rank not in (0, 1) or type(block_size) is not int or block_size < 2
            or block_size % world or candidates.ndim != 2 or candidates.shape[0] != rows
            or not 1 <= candidates.shape[1] <= 8192 or candidates.stride(1) != 1
            or candidates.dtype not in (torch.int32, torch.int64) or candidates.device != kv.device):
        raise ValueError('Expected DCP2 decode candidate blocks for every query row')
    output = torch.empty((rows, max_model_len), device=kv.device, dtype=torch.float32)
    errors = torch.zeros(batch, device=kv.device, dtype=torch.int32)
    local, k = block_size // world, candidates.shape[1]
    _candidate_logits_kernel[(rows, triton.cdiv(k * local, 128))](
        values, values.stride(0), values.stride(1), values.stride(2), weights, weights.stride(0),
        weights.stride(1), lengths, lengths.stride(0), lengths.stride(1), kv, table, table.stride(0),
        table.stride(1), candidates, candidates.stride(0), errors, output, output.stride(0),
        N=next_n, CAP=max_model_len, K=k, NB=triton.cdiv(max_model_len * world, block_size),
        LOCAL=local, STATES=kv.shape[1], PAGE_STRIDE=kv.stride(0), COLUMNS=table.shape[1],
        PAGES=kv.shape[0], BN=128, num_warps=4)
    check_flags(errors, _ERRORS)
    return output


def _graph_paged_logits_gathered(q, kv, weights, lengths, table, schedule_metadata, *,
                                 max_model_len, clean_logits=False, indices=None, state_chunk=8192):
    """The previous workspace-gather scorer; kept only as the probes' bitwise reference."""
    from .graph_validation import check_flags, require_capture_owner
    require_capture_owner()
    values, q_scale = q
    values = _fp8_query(values)
    if (values.ndim != 4 or not 1 <= values.shape[0] <= 24
            or values.shape[0] * values.shape[1] > 24 or not 1 <= values.shape[1] <= 4
            or q_scale is not None or indices is not None or lengths.shape != values.shape[:2]
            or type(max_model_len) is not int or not 1 <= max_model_len <= 1048576
            or type(state_chunk) is not int or not 1 <= state_chunk <= 8192):
        raise ValueError(f'Expected at most24 FP8 query rows: values={values.shape}, lengths={lengths.shape}')
    batch, next_n = values.shape[:2]
    _check_view(kv)
    if (weights.shape not in ((batch * next_n, HEADS), (batch, next_n, HEADS))
            or weights.dtype != torch.float32 or table.ndim != 2 or table.shape[0] != batch
            or table.dtype not in (torch.int32, torch.int64)
            or lengths.dtype not in (torch.int32, torch.int64)
            or any(x.device != kv.device for x in (values, weights, lengths, table))
            or any(s < 0 for x in (values, weights, lengths, table) for s in x.stride())):
        raise ValueError('Invalid NVFP4 graph page/query layout or metadata')
    # At 1M states the key workspace is 72 MiB; the DCP caller normally asks for half.
    keys = torch.empty((max_model_len, 64), device=kv.device, dtype=torch.uint8)
    key_scales = torch.empty((max_model_len, 8), device=kv.device, dtype=torch.uint8)
    output = torch.empty((batch * next_n, max_model_len), device=kv.device, dtype=torch.float32)
    errors = torch.zeros(batch, device=kv.device, dtype=torch.int32)
    starts = torch.zeros(next_n, device=kv.device, dtype=torch.int32)
    weights = weights.reshape(batch, next_n, HEADS)
    for request in range(batch):
        sizes = lengths[request]
        _graph_gather_kernel[(triton.cdiv(max_model_len, 32),)](
            kv, table[request], sizes, keys, key_scales, errors[request:request + 1],
            N=next_n, NL=triton.next_power_of_2(next_n), CAP=max_model_len,
            STATES=kv.shape[1], PAGE_STRIDE=kv.stride(0), TABLE_STRIDE=table.stride(1),
            COLUMNS=table.shape[1], LENGTH_STRIDE=sizes.stride(0), PAGES=kv.shape[0],
            BLOCK=32, num_warps=4)
        ends = sizes.clamp(0, max_model_len).to(torch.int32).contiguous()
        mqa_logits(values[request].contiguous(), weights[request].contiguous(), keys, key_scales,
                   starts, ends, out=output[request * next_n:(request + 1) * next_n])
    check_flags(errors, _ERRORS)
    return output


# ---------------------------------------------------------------- NVFP4 prefill queries

@triton.jit
def _query_kernel(q, q_row, q_head, positions, cos_sin, cos_sin_stride, weights, w_row,
                  values, scales, head_weights, softmax_scale, head_scale, HEAD_BLOCK: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1) * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    x = tl.load(q + token * q_row + head[:, None] * q_head + tl.arange(0, 128)[None, :]).to(tl.float32)
    even, odd = tl.split(tl.reshape(x, (HEAD_BLOCK, 64, 2)))
    # Same GPT-J RoPE on the last 64 dims (pairs 32..63) and BF16 round trip as vLLM's
    # indexer query path; the first 64 dims pass through unchanged (cos 1, sin 0).
    rope = tl.arange(0, 64) - 32
    is_rope = rope >= 0
    column = tl.maximum(rope, 0)
    position = tl.load(positions + token)
    cos = tl.load(cos_sin + position * cos_sin_stride + column, mask=is_rope, other=1.0).to(tl.float32)[None, :]
    sin = tl.load(cos_sin + position * cos_sin_stride + 32 + column, mask=is_rope, other=0.0).to(tl.float32)[None, :]
    new_even = (even * cos - odd * sin).to(tl.bfloat16).to(tl.float32)
    new_odd = (odd * cos + even * sin).to(tl.bfloat16).to(tl.float32)
    vector = tl.interleave(new_even, new_odd)
    # NVFP4 has no tensor scale: a power of two per token and head brings the head's
    # largest value to <= 1024 (so every searched scale, <= amax/2.5, is below E4M3's
    # 448) and is folded into its weight.
    amax = tl.maximum(tl.max(tl.abs(vector), axis=1), 1e-30)
    bits = (amax * (1.0 / 1024.0)).to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    unscale = ((127 - exponent) << 23).to(tl.float32, bitcast=True)
    scale = ((127 + exponent) << 23).to(tl.float32, bitcast=True)
    codes, scale_bits = _nvfp4_encode(tl.reshape(vector * unscale[:, None], (HEAD_BLOCK * 8, 16)))
    low, high = tl.split(tl.reshape(codes, (HEAD_BLOCK * 64, 2)))
    row = token * 32 + tl.program_id(1) * HEAD_BLOCK
    tl.store(values + row * 64 + tl.arange(0, HEAD_BLOCK * 64), (low | (high << 4)).to(tl.uint8))
    tl.store(scales + row * 8 + tl.arange(0, HEAD_BLOCK * 8), scale_bits)
    weight = tl.load(weights + token * w_row + head).to(tl.float32)
    tl.store(head_weights + token * 32 + head, weight * softmax_scale * head_scale * scale)


def _query_geometry(rows):
    """(heads per program, warps): measured on SM121; decode is latency-bound."""
    return (1, 1) if rows <= 64 else (4, 2)


def quantize_queries(q, positions, cos_sin, weights, softmax_scale, head_scale):
    """Pre-RoPE BF16 queries [T, 32, 128] -> NVFP4 (values [T,32,64], scales [T,32,8], weights [T,32])."""
    rows = q.shape[0]
    if (q.dtype != torch.bfloat16 or q.ndim != 3 or q.shape[1:] != (HEADS, HEAD_DIM) or q.stride(2) != 1
            or positions.shape != (rows,) or weights.shape != (rows, HEADS) or weights.stride(1) != 1
            or cos_sin.ndim != 2 or cos_sin.shape[1] != 64 or cos_sin.stride(1) != 1
            or any(t.device != q.device for t in (positions, weights, cos_sin))):
        raise ValueError('Invalid NVFP4 prefill query inputs')
    values = torch.empty((rows, HEADS, 64), device=q.device, dtype=torch.uint8)
    scales = torch.empty((rows, HEADS, 8), device=q.device, dtype=torch.uint8)
    head_weights = torch.empty((rows, HEADS), device=q.device, dtype=torch.float32)
    if rows:
        block, warps = _query_geometry(rows)
        _query_kernel[(rows, HEADS // block)](q, q.stride(0), q.stride(1), positions, cos_sin,
            cos_sin.stride(0), weights, weights.stride(0), values, scales, head_weights, float(softmax_scale),
            float(head_scale), HEAD_BLOCK=block, num_warps=warps, enable_fp_fusion=False)
    return values, scales, head_weights


class QueryPackage:
    """Pre-RoPE BF16 indexer queries carried to the prefill scorer; quantized per chunk.

    Returned by the patched indexer forward in place of the (unused) FP8 q_scale.
    Building one launches nothing, so decode graphs are unaffected.
    """
    __slots__ = ('q', 'positions', 'cos_sin', 'weights', 'softmax_scale', 'head_scale')

    def __init__(self, q, positions, cos_sin, weights, softmax_scale, head_scale):
        self.q, self.positions, self.cos_sin, self.weights = q, positions, cos_sin, weights
        self.softmax_scale, self.head_scale = softmax_scale, head_scale

    def quantize(self, start, end):
        return quantize_queries(self.q[start:end], self.positions[start:end], self.cos_sin,
                                self.weights[start:end], self.softmax_scale, self.head_scale)


_native_library = None


def _native():
    global _native_library
    if _native_library is None:
        root = Path(__file__).resolve().parents[1] / 'nvfp4-indexer-native'
        receipt = json.loads((root / 'complete.json').read_bytes())
        binary = root / 'nvfp4_indexer.so'
        if hashlib.sha256(binary.read_bytes()).hexdigest() != receipt['binary_sha256']:
            raise ValueError('NVFP4 indexer native library differs from its build receipt')
        library = ctypes.CDLL(str(binary))
        if library.ds41_nvfp4_indexer_abi() != 2:
            raise ValueError('Unexpected NVFP4 indexer native ABI')
        library.ds41_nvfp4_logits.argtypes = [ctypes.c_void_p] * 8 + [
            ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        library.ds41_nvfp4_logits.restype = ctypes.c_int
        _native_library = library
    return _native_library


SPLIT_TARGET_CTAS = 192


def key_split(rows, keys):
    """Keys per CTA slice: enough (row, slice) CTAs to fill GB10 when rows are few."""
    ceil = lambda a, b: -(-a // b)
    slices = max(1, ceil(SPLIT_TARGET_CTAS, ceil(rows, 2)))
    split = max(2048, ceil(ceil(keys, slices), 128) * 128)
    return min(split, max(128, ceil(keys, 128) * 128))


def nvfp4_logits(values, scales, head_weights, key_values, key_scales, starts, ends, *, out=None, clean=False):
    """NVFP4 query x NVFP4 key logits [M, N] on FP4 tensor cores.

    clean=False: DeepGEMM clean_logits=False contract (only [starts, ends) defined).
    clean=True: -inf everywhere outside [starts, ends). Launch only; capture-safe.
    """
    key_values = key_values.view(torch.uint8)
    if key_scales.dtype != torch.uint8:
        key_scales = key_scales.view(torch.uint8)
    m, n = values.shape[0], key_values.shape[0]
    if starts.dtype != torch.int32 or not starts.is_contiguous():
        starts = starts.to(torch.int32).contiguous()
    if ends.dtype != torch.int32 or not ends.is_contiguous():
        ends = ends.to(torch.int32).contiguous()
    if (values.shape != (m, HEADS, 64) or not values.is_contiguous()
            or scales.shape != (m, HEADS, 8) or not scales.is_contiguous()
            or head_weights.shape != (m, HEADS) or head_weights.dtype != torch.float32
            or not head_weights.is_contiguous()
            or key_values.shape != (n, 64) or not key_values.is_contiguous()
            or key_scales.shape != (n, 8) or not key_scales.is_contiguous()
            or starts.shape != (m,) or ends.shape != (m,)
            or any(t.device != values.device or not t.is_cuda
                   for t in (scales, head_weights, key_values, key_scales, starts, ends))):
        raise ValueError('Invalid NVFP4 logits inputs')
    if out is None:
        out = torch.empty((m, n), device=values.device, dtype=torch.float32)
    elif out.shape != (m, n) or out.dtype != torch.float32 or out.stride(1) != 1 or out.device != values.device:
        raise ValueError('Invalid NVFP4 logits output buffer')
    code = _native().ds41_nvfp4_logits(values.data_ptr(), scales.data_ptr(), head_weights.data_ptr(),
        key_values.data_ptr(), key_scales.data_ptr(), starts.data_ptr(), ends.data_ptr(), out.data_ptr(),
        m, n, out.stride(0), torch.cuda.current_stream(values.device).cuda_stream, key_split(m, n), int(clean))
    if code:
        raise RuntimeError(f'NVFP4 logits failed ({code})')
    return out


def prefill_logits(package, start, end, q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=False):
    """Prefill replacement for fp8_fp4_mqa_logits; FP8 x NVFP4 fallback when no package."""
    if package is None:
        return mqa_logits_native_signature(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits)
    if not isinstance(package, QueryPackage) or end - start != q[0].shape[0]:
        raise ValueError('NVFP4 prefill query package does not match the chunk')
    values, scales, head_weights = package.quantize(start, end)
    return nvfp4_logits(values, scales, head_weights, kv[0], kv[1], cu_seqlen_ks, cu_seqlen_ke)


# ---------------------------------------------------------------- NVFP4 decode queries

def _decode_query_layout(values, scales, head_weights, batch, next_n):
    rows = batch * next_n
    if values.shape[0] != rows:
        raise ValueError('NVFP4 decode queries do not match the padded decode batch')
    return (values.reshape(batch, next_n, HEADS, 64), scales.reshape(batch, next_n, HEADS, 8),
            head_weights.reshape(batch, next_n, HEADS))


def paged_logits_nvfp4(values, scales, head_weights, kv, lengths, table, *, max_model_len, state_chunk=8192):
    """Eager DCP decode scorer with NVFP4 queries [B, next_n, 32, 64|8]; lengths [B, next_n]."""
    _check_view(kv)
    batch, next_n = values.shape[:2]
    if (values.shape != (batch, next_n, HEADS, 64) or scales.shape != (batch, next_n, HEADS, 8)
            or head_weights.shape != (batch, next_n, HEADS) or lengths.shape != (batch, next_n)
            or table.ndim != 2 or table.shape[0] != batch
            or type(max_model_len) is not int or not 1 <= max_model_len <= 1048576):
        raise ValueError('Invalid NVFP4 decode query batch')
    counts = lengths.max(1).values.cpu().tolist()
    if any(c < 0 or c > max_model_len for c in counts):
        raise ValueError('Local context exceeds bounded logits allocation')
    output = torch.full((batch * next_n, max_model_len), -torch.inf, device=kv.device, dtype=torch.float32)
    capacity = min(state_chunk, max(counts, default=0))
    if not capacity:
        return output
    keys = torch.empty((capacity, 64), device=kv.device, dtype=torch.uint8)
    key_scales = torch.empty((capacity, 8), device=kv.device, dtype=torch.uint8)
    for request, length in enumerate(counts):
        rows = slice(request * next_n, (request + 1) * next_n)
        ends_all = lengths[request].to(torch.int32)
        for start in range(0, length, capacity):
            n = min(capacity, length - start)
            _gather_pages_kernel[(triton.cdiv(n, 32),)](kv, table[request], start, n, keys, key_scales,
                STATES=kv.shape[1], PAGE_STRIDE=kv.stride(0), TABLE_STRIDE=table.stride(1),
                PAGES=kv.shape[0], BLOCK=32, num_warps=4)
            chunk_ends = (ends_all - start).clamp(0, n).contiguous()
            nvfp4_logits(values[request].contiguous(), scales[request].contiguous(),
                         head_weights[request].contiguous(), keys[:n], key_scales[:n],
                         torch.zeros(next_n, device=kv.device, dtype=torch.int32), chunk_ends,
                         out=output[rows, start:start + n], clean=True)
    return output


def graph_paged_logits_nvfp4(values, scales, head_weights, kv, lengths, table, *, max_model_len):
    """Capture-safe decode/DSpark scorer with NVFP4 queries (graph_paged_logits counterpart)."""
    from .graph_validation import check_flags, require_capture_owner
    require_capture_owner()
    _check_view(kv)
    batch, next_n = values.shape[:2]
    if (not 1 <= batch <= 24 or batch * next_n > 24 or not 1 <= next_n <= 4
            or values.shape != (batch, next_n, HEADS, 64) or scales.shape != (batch, next_n, HEADS, 8)
            or head_weights.shape != (batch, next_n, HEADS) or lengths.shape != (batch, next_n)
            or table.ndim != 2 or table.shape[0] != batch
            or table.dtype not in (torch.int32, torch.int64) or lengths.dtype not in (torch.int32, torch.int64)
            or type(max_model_len) is not int or not 1 <= max_model_len <= 1048576):
        raise ValueError('Invalid NVFP4 graph decode query batch')
    keys = torch.empty((max_model_len, 64), device=kv.device, dtype=torch.uint8)
    key_scales = torch.empty((max_model_len, 8), device=kv.device, dtype=torch.uint8)
    output = torch.empty((batch * next_n, max_model_len), device=kv.device, dtype=torch.float32)
    errors = torch.zeros(batch, device=kv.device, dtype=torch.int32)
    starts = torch.zeros(next_n, device=kv.device, dtype=torch.int32)
    for request in range(batch):
        sizes = lengths[request]
        _graph_gather_kernel[(triton.cdiv(max_model_len, 32),)](
            kv, table[request], sizes, keys, key_scales, errors[request:request + 1],
            N=next_n, NL=triton.next_power_of_2(next_n), CAP=max_model_len,
            STATES=kv.shape[1], PAGE_STRIDE=kv.stride(0), TABLE_STRIDE=table.stride(1),
            COLUMNS=table.shape[1], LENGTH_STRIDE=sizes.stride(0), PAGES=kv.shape[0],
            BLOCK=32, num_warps=4)
        ends = sizes.clamp(0, max_model_len).to(torch.int32).contiguous()
        nvfp4_logits(values[request].contiguous(), scales[request].contiguous(),
                     head_weights[request].contiguous(), keys, key_scales, starts, ends,
                     out=output[request * next_n:(request + 1) * next_n], clean=True)
    check_flags(errors, ((1, 'Local context exceeds bounded logits allocation'),
        (2, 'Indexer block table is too short'), (4, 'Invalid physical indexer page ID')))
    return output


def make_decode_logits(fp8_scorer, dcp_group=None):
    """Decode dispatcher for the recompiled sparse_attn_indexer.

    FP8 queries (default) keep the installed FP8 x NVFP4 scorer. With
    DS41_INDEXER_DECODE_QUERY=nvfp4 the QueryPackage's decode rows are quantized like
    prefill and scored on FP4 tensor cores; batches needing ragged padding (short
    chunked prefill mixed into decode) keep the FP8 scorer.

    Indexers that consume two-level candidate blocks pass ``candidates`` (rows x K
    global block ids); with the capture-safe FP8 scorer only those blocks' columns
    are scored (graph_candidate_logits), bitwise equal to scoring every column,
    because the caller's apply_candidate_mask discards all others. Other routes
    score every column and rely on the same mask.
    """
    nvfp4_scorer = graph_paged_logits_nvfp4 if fp8_scorer is graph_paged_logits else paged_logits_nvfp4

    def decode_logits(package, count, requires_padding, q, kv, weights, lengths, table, schedule_metadata,
                      *, max_model_len, clean_logits=False, indices=None, candidates=None,
                      candidate_block_size=0):
        values = q[0]
        if (DECODE_QUERY != 'nvfp4' or package is None or requires_padding or indices is not None
                or values.ndim != 4 or count != values.shape[0] * values.shape[1]):
            if candidates is not None and fp8_scorer is graph_paged_logits and dcp_group is not None:
                group = dcp_group()
                return graph_candidate_logits(q, kv, weights, lengths, table, candidates,
                                              block_size=candidate_block_size, rank=group.rank_in_group,
                                              world=group.world_size, max_model_len=max_model_len,
                                              indices=indices)
            return fp8_scorer(q, kv, weights, lengths, table, schedule_metadata, max_model_len=max_model_len,
                              clean_logits=clean_logits, indices=indices)
        batch, next_n = values.shape[:2]
        nv_values, nv_scales, nv_weights = _decode_query_layout(*package.quantize(0, count), batch, next_n)
        if lengths.ndim == 1:
            lengths = lengths[:, None]
        if lengths.shape == (batch, 1) and next_n > 1:
            lengths = lengths.expand(batch, next_n)
        return nvfp4_scorer(nv_values, nv_scales, nv_weights, kv, lengths, table, max_model_len=max_model_len)

    return decode_logits


# ---------------------------------------------------------------- torch references (tests)

_LEVELS = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
_ORDER = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])  # even codes first: argmin = nearest-even


_E4M3 = torch.arange(1, 0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn).float()  # positive, ascending


def _reference_codes(x, scale):
    """Nearest-even E2M1 codes of x / scale and |reconstruction| (float64) for x [..., 16], scale [...]."""
    normalized = x / scale[..., None]
    distance = (normalized.abs()[..., None] - _LEVELS[_ORDER]).abs()
    codes = _ORDER[distance.argmin(-1)] | (torch.signbit(x).long() << 3)
    return codes, _LEVELS[codes & 7].double() * scale.double()[..., None]


def _reference_mx_scale(x):
    """DeepSeek's MXFP4 scale 2**ceil(log2(amax32 / 6)) broadcast to 16-groups of x [n, 8, 16]."""
    amax32 = x.abs().amax(-1).reshape(-1, 4, 2).amax(-1).repeat_interleave(2, -1)
    ratio = amax32.clamp_min(6 * 2.0 ** -126) * torch.tensor(1 / 6, dtype=torch.float32)
    mantissa, exponent = torch.frexp(ratio)
    exponent = torch.where(mantissa == 0.5, exponent - 1, exponent)
    return torch.ldexp(torch.ones_like(ratio), exponent), (exponent >= -9) & (exponent <= 8)


def reference_groups(x, mode='search'):
    """BF16-exact float32 groups [..., 16] -> (E2M1 codes [..., 16], E4M3 scale bytes [...]).

    'div6': E4M3(amax/6). 'four_over_six': E4M3(amax/4) if strictly better. 'search' (the
    writers): starting from E4M3(amax/6), every E4M3 scale in [amax/6.5, amax/2.5] in
    ascending order replaces the best only if strictly better. SSEs are compared as
    sum((b - a) * (b + a - 2|x|)) in float64, exact for these operands.
    """
    x = x.float()
    ax = x.abs().double()
    amax = x.abs().amax(-1).clamp_min(6 * 2 ** -9)
    best = (amax / 6).to(torch.float8_e4m3fn)
    codes, restored = _reference_codes(x, best.float())
    scales = best.view(torch.uint8)
    candidates = []
    if mode == 'four_over_six':
        candidates = [(amax / 4).clamp_max(448).to(torch.float8_e4m3fn).float()]
    elif mode == 'search':
        low, high = amax / 6.5, amax / 2.5
        candidates = [torch.where((value >= low) & (value <= high), value, torch.nan)
                      for value in _E4M3 if bool(((value >= low) & (value <= high)).any())]
    elif mode != 'div6':
        raise ValueError(mode)
    for scale in candidates:
        live = ~scale.isnan()
        trial_codes, trial = _reference_codes(x, torch.where(live, scale, 1.0))
        better = live & (((trial - restored) * (trial + restored - 2 * ax)).sum(-1) < 0)
        codes = torch.where(better[..., None], trial_codes, codes)
        restored = torch.where(better[..., None], trial, restored)
        scales = torch.where(better, torch.where(live, scale, 1.0).to(torch.float8_e4m3fn).view(torch.uint8), scales)
    return codes, scales


def reference_quantize(values, mode='search'):
    """Post-RoPE BF16 vectors [n, 128] -> (packed [n, 64] uint8, scales [n, 8] uint8)."""
    codes, scales = reference_groups(values.cpu().float().reshape(-1, 8, 16), mode)
    codes = codes.reshape(-1, 128)
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8), scales.reshape(-1, 8)


def reference_mxfp4(values):
    """DeepSeek's MXFP4 quantize-dequantize of [n, 128] (32-value blocks, E8M0 scales), float32."""
    x = values.cpu().float().reshape(-1, 8, 16)
    scale, _ = _reference_mx_scale(x)
    normalized = (x / scale[..., None]).clamp(-6, 6)
    distance = (normalized.abs()[..., None] - _LEVELS[_ORDER]).abs()
    restored = _LEVELS[_ORDER[distance.argmin(-1)]] * scale[..., None]
    return torch.where(torch.signbit(x), -restored, restored).reshape(-1, 128)


def reference_query_rope(q, positions, table):
    """vLLM's indexer query RoPE (last 64 dims, GPT-J pairs), unfused float32, BF16 result."""
    x = q.cpu().float()
    cos = table.cpu().float()[positions.cpu()][:, None, :32]
    sin = table.cpu().float()[positions.cpu()][:, None, 32:]
    even, odd = x[..., 64::2], x[..., 65::2]
    rope = torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1).reshape(*x.shape[:2], 64)
    return torch.cat((x[..., :64], rope.bfloat16().float()), -1).bfloat16()


def reference_queries(q_roped, weights, softmax_scale, head_scale):
    """RoPE'd BF16 queries [T, 32, 128] -> NVFP4 (packed [T,32,64], scales [T,32,8], weights [T,32])."""
    x = q_roped.cpu().float()
    amax = x.abs().amax(-1).clamp_min(1e-30) * torch.tensor(1 / 1024, dtype=torch.float32)
    mantissa, exponent = torch.frexp(amax)
    exponent = torch.where(mantissa == 0.5, exponent - 1, exponent)
    packed, scales = reference_quantize(torch.ldexp(x, -exponent[..., None]).reshape(-1, 128))
    head_weights = weights.cpu().float() * softmax_scale * head_scale * torch.ldexp(torch.ones_like(amax), exponent)
    return packed.reshape(*x.shape[:2], 64), scales.reshape(*x.shape[:2], 8), head_weights


def reference_dequantize(packed, scales):
    packed = packed.cpu().long()
    codes = torch.stack((packed & 15, packed >> 4), -1).reshape(-1, 128)
    value = _LEVELS[codes & 7]
    value = torch.where((codes & 8) != 0, -value, value)
    group = scales.cpu().view(torch.float8_e4m3fn).float()
    return (value.reshape(-1, 8, 16) * group[..., None]).reshape(-1, 128)


def reference_nvfp4_logits(q_packed, q_scales, head_weights, packed, scales, starts, ends):
    """Float64 logits of NVFP4 queries x NVFP4 keys; -inf outside each row's range."""
    query = reference_dequantize(q_packed.reshape(-1, 64), q_scales.reshape(-1, 8)).double()
    query = query.reshape(-1, HEADS, HEAD_DIM)
    key = reference_dequantize(packed, scales).double()
    logits = (torch.einsum('mhd,nd->mhn', query, key).clamp_min(0) * head_weights.cpu().double()[..., None]).sum(1)
    column = torch.arange(key.shape[0])
    live = (column[None] >= starts.cpu()[:, None]) & (column[None] < ends.cpu()[:, None])
    return torch.where(live, logits, torch.full_like(logits, float('-inf'))).float()


def reference_logits(q, weights, packed, scales, starts, ends):
    key = reference_dequantize(packed, scales).double()
    query = _fp8_query(q).cpu().float().double()
    score = torch.einsum('mhd,nd->mhn', query, key).clamp_min(0)
    logits = (score * weights.cpu().double()[..., None]).sum(1)
    column = torch.arange(key.shape[0])
    live = (column[None] >= starts.cpu()[:, None]) & (column[None] < ends.cpu()[:, None])
    return torch.where(live, logits, torch.full_like(logits, float('-inf'))).float()
