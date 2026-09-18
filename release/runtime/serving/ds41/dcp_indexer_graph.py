# SPDX-License-Identifier: AGPL-3.0-only
# DS41 whole-graph/DSpark adaptation. Original MXFP4 layout and the native
# DeepGEMM scoring operation are retained; no replacement quantization math.
"""Static-workspace MXFP4 paged scoring for decode and DSpark verification.

Device lengths control masked gathers and native MQA row ends. No length or
page readback occurs inside capture. Invalid metadata is safely masked and
reported through the model graph owner before its output is returned.
"""
import torch
import triton
import triton.language as tl

from .graph_validation import check_flags, require_capture_owner


@triton.jit
def _gather(cache, table, lengths, keys, scales, errors,
            N: tl.constexpr, NL: tl.constexpr, CAP: tl.constexpr,
            STATES: tl.constexpr, PAGE_STRIDE: tl.constexpr,
            TABLE_STRIDE: tl.constexpr, COLUMNS: tl.constexpr,
            LENGTH_STRIDE: tl.constexpr, PAGES: tl.constexpr, BLOCK: tl.constexpr):
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
    page = tl.load(table + tl.where(in_table, column, 0) * TABLE_STRIDE,
                   in_table, other=-1).to(tl.int64)
    valid = in_table & (page >= 0) & (page < PAGES)
    flags = tl.where(tl.sum((live & ~in_table).to(tl.int32), 0) > 0, 2, 0)
    flags |= tl.where(tl.sum((in_table & ~valid).to(tl.int32), 0) > 0, 4, 0)
    if flags != 0:
        tl.atomic_or(errors, flags)
    offset = tl.where(valid, page, 0) * PAGE_STRIDE
    row = i % STATES
    channel = tl.arange(0, 64)
    packed = tl.load(cache + offset[:, None] + row[:, None] * 64 + channel[None, :],
                     valid[:, None], other=0)
    scale_ptr = (cache + offset + STATES * 64 + row * 4).to(tl.pointer_type(tl.int32))
    scale = tl.load(scale_ptr, valid, other=0)
    tl.store(keys + i[:, None] * 64 + channel[None, :], packed, live[:, None])
    tl.store(scales + i, scale, live)


@triton.jit
def _copy_logits(source, lengths, output, ROWS: tl.constexpr, CAP: tl.constexpr,
                 SOURCE_STRIDE: tl.constexpr, LENGTH_STRIDE: tl.constexpr,
                 OUTPUT_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(lengths + row * LENGTH_STRIDE)
    valid = (col < CAP) & (col < length)
    value = tl.load(source + row * SOURCE_STRIDE + col, valid, other=-float('inf'))
    tl.store(output + row * OUTPUT_STRIDE + col, value, col < CAP)


def paged_logits(q, kv, weights, lengths, table, schedule_metadata, *,
                 max_model_len, clean_logits=False, indices=None, state_chunk=8192):
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits
    require_capture_owner()
    values, q_scale = q
    if (values.ndim != 4 or not 1 <= values.shape[0] <= 24
            or values.shape[0] * values.shape[1] > 24
            or not 1 <= values.shape[1] <= 4 or values.shape[2:] != (32, 64)
            or values.dtype not in (torch.int8, torch.uint8)
            or q_scale is None or q_scale.shape != values.shape[:-1]
            or q_scale.dtype != torch.int32 or indices is not None
            or lengths.shape != values.shape[:2]
            or type(max_model_len) is not int or not 1 <= max_model_len <= 1048576
            or type(state_chunk) is not int or not 1 <= state_chunk <= 8192):
        raise ValueError(f'Expected at most24 MXFP4 query rows: values={values.shape}, scale={None if q_scale is None else q_scale.shape}, lengths={lengths.shape}, cap={max_model_len}')
    batch, next_n = values.shape[:2]
    if (not kv.is_cuda or kv.dtype != torch.uint8 or kv.ndim != 4
            or kv.shape[2:] != (1, 68) or kv.shape[1] not in (64, 128)
            or kv.stride(1) != 68 or kv.stride(-1) != 1
            or kv.stride(0) < kv.shape[1] * 68 or kv.stride(0) % 4
            or weights.shape not in ((batch * next_n, 32), (batch, next_n, 32))
            or weights.dtype != torch.float32 or table.ndim != 2 or table.shape[0] != batch
            or table.dtype not in (torch.int32, torch.int64)
            or lengths.dtype not in (torch.int32, torch.int64)
            or any(x.device != kv.device for x in (values, q_scale, weights, lengths, table))
            or any(s < 0 for x in (values, q_scale, weights, lengths, table) for s in x.stride())):
        raise ValueError('Invalid MXFP4 graph page/query layout or metadata')
    # Shared across requests and released inside the capture pool. At1M
    # states this is68MiB; the native DCP caller normally requests half that.
    keys = torch.empty((max_model_len, 64), device=kv.device, dtype=torch.int8)
    scales = torch.empty(max_model_len, device=kv.device, dtype=torch.int32)
    output = torch.empty((batch * next_n, max_model_len), device=kv.device, dtype=torch.float32)
    errors = torch.zeros(batch, device=kv.device, dtype=torch.int32)
    starts = torch.zeros(next_n, device=kv.device, dtype=torch.int32)
    weights = weights.reshape(batch, next_n, 32)
    for request in range(batch):
        sizes = lengths[request]
        _gather[(triton.cdiv(max_model_len, 32),)](
            kv, table[request], sizes, keys, scales, errors[request:request + 1],
            N=next_n, NL=triton.next_power_of_2(next_n), CAP=max_model_len,
            STATES=kv.shape[1], PAGE_STRIDE=kv.stride(0), TABLE_STRIDE=table.stride(1),
            COLUMNS=table.shape[1], LENGTH_STRIDE=sizes.stride(0), PAGES=kv.shape[0],
            BLOCK=32, num_warps=4)
        ends = sizes.clamp(0, max_model_len).to(torch.int32).contiguous()
        logits = fp8_fp4_mqa_logits(
            (values[request].contiguous().view(torch.int8), q_scale[request].contiguous()),
            (keys, scales), weights[request].contiguous(), starts, ends, clean_logits=False)
        if (logits.device != kv.device or logits.dtype != torch.float32
                or logits.ndim != 2 or logits.shape[0] != next_n
                or logits.shape[1] < max_model_len or logits.stride(1) != 1):
            raise ValueError('Native MXFP4 scorer returned an incompatible logits buffer')
        destination = output[request * next_n:(request + 1) * next_n]
        _copy_logits[(next_n, triton.cdiv(max_model_len, 256))](
            logits, ends, destination, ROWS=next_n, CAP=max_model_len,
            SOURCE_STRIDE=logits.stride(0), LENGTH_STRIDE=ends.stride(0),
            OUTPUT_STRIDE=destination.stride(0), BLOCK=256, num_warps=4)
    check_flags(errors, ((1, 'Local context exceeds bounded logits allocation'),
        (2, 'Indexer block table is too short'), (4, 'Invalid physical indexer page ID')))
    return output
