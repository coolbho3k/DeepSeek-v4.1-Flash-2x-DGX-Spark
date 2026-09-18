"""Bounded MXFP4 indexer decode gather for the pinned native MQA kernel.

Pages contain all packed E2M1 rows followed by all UE8M0 scale rows. Preserve
padded physical page strides and pass native int8/int32 scalar-type tags.
No runtime hooks, cache allocation policy, or architecture guards change here.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _gather(cache, table, keys, scales, start, count,
            STATES: tl.constexpr, PAGE_STRIDE: tl.constexpr,
            TABLE_STRIDE: tl.constexpr, PAGES: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pos = start + i
    page = tl.load(table + (pos // STATES) * TABLE_STRIDE,
                   i < count, other=-1).to(tl.int64)
    valid = (i < count) & (page >= 0) & (page < PAGES)
    offset = tl.where(valid, page, 0) * PAGE_STRIDE
    row = pos % STATES
    channel = tl.arange(0, 64)
    packed = tl.load(cache + offset[:, None] + row[:, None] * 64 + channel[None, :],
                     valid[:, None], other=0)
    # Four scale bytes are opaque packed UE8M0, NOT one FP32 multiplier.
    scale_ptr = (cache + offset + STATES * 64 + row * 4).to(tl.pointer_type(tl.int32))
    scale = tl.load(scale_ptr, valid, other=0)
    tl.store(keys + i[:, None] * 64 + channel[None, :], packed, i[:, None] < count)
    tl.store(scales + i, scale, i < count)


def paged_logits(q, kv, weights, lengths, table, schedule_metadata, *,
                 max_model_len, clean_logits=False, indices=None, state_chunk=8192):
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits
    values, q_scale = q
    if (values.ndim != 4 or values.shape[1:] != (1, 32, 64)
            or values.dtype not in (torch.int8, torch.uint8)
            or q_scale is None or q_scale.shape != values.shape[:-1]
            or q_scale.dtype != torch.int32 or indices is not None
            or lengths.shape != (values.shape[0], 1)
            or type(state_chunk) is not int or not 1 <= state_chunk <= 8192
            or type(max_model_len) is not int or not 1 <= max_model_len <= 1048576):
        raise ValueError('Expected bounded MXFP4 next_n1 indexer queries and context')
    if (not kv.is_cuda or kv.dtype != torch.uint8 or kv.ndim != 4
            or kv.shape[2:] != (1, 68) or kv.shape[1] not in (64, 128)
            or kv.stride(1) != 68 or kv.stride(-1) != 1
            or kv.stride(0) < kv.shape[1] * 68 or kv.stride(0) % 4):
        raise ValueError('Expected aligned segregated MXFP4 indexer pages')
    if (weights.shape != (values.shape[0], 32) or weights.dtype != torch.float32
            or table.ndim != 2 or table.shape[0] != values.shape[0]
            or table.dtype not in (torch.int32, torch.int64)
            or lengths.dtype not in (torch.int32, torch.int64)
            or any(x.device != kv.device for x in (values, q_scale, weights, lengths, table))):
        raise ValueError('Invalid MXFP4 weights, lengths, table or devices')
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
    output = torch.full((len(counts), max_model_len), -torch.inf,
                        device=kv.device, dtype=torch.float32)
    capacity = min(state_chunk, max(counts, default=0))
    if not capacity:
        return output
    keys = torch.empty((capacity, 64), device=kv.device, dtype=torch.uint8)
    scales = torch.empty(capacity, device=kv.device, dtype=torch.int32)
    row_start = torch.zeros(1, device=kv.device, dtype=torch.int32)
    full_end = torch.full((1,), capacity, device=kv.device, dtype=torch.int32)
    for request, length in enumerate(counts):
        query = values[request].contiguous().view(torch.int8)
        query_scale = q_scale[request].contiguous()
        weight = weights[request:request + 1].contiguous()
        for start in range(0, length, capacity):
            n = min(capacity, length - start)
            _gather[(triton.cdiv(n, 32),)](kv, table[request], keys, scales, start, n,
                STATES=kv.shape[1], PAGE_STRIDE=kv.stride(0), TABLE_STRIDE=table.stride(1),
                PAGES=kv.shape[0], BLOCK=32, num_warps=4)
            row_end = full_end if n == capacity else torch.full((1,), n, device=kv.device, dtype=torch.int32)
            logits = fp8_fp4_mqa_logits((query, query_scale),
                (keys[:n].view(torch.int8), scales[:n]), weight, row_start, row_end,
                clean_logits=False)
            output[request, start:start + n].copy_(logits[0, :n])
    return output
