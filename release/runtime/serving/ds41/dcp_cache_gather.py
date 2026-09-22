"""Fused packed FP8/UE8M0 + BF16-RoPE cache gather, without offset matrices."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gather(cache, slots, output, count, cache_rows,
            PAGE_STRIDE: tl.constexpr, STATES: tl.constexpr, BLOCK_ROWS: tl.constexpr, SCALE_BYTES: tl.constexpr = 8):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    channel = tl.arange(0, 512)
    slot = tl.load(slots + row, row < count, other=-1)
    valid = (row < count) & (slot >= 0) & (slot < cache_rows)
    safe = tl.where(valid, slot, 0)
    page, state = safe // STATES, safe % STATES
    page_offset = page[:, None] * PAGE_STRIDE
    value_offset = state[:, None] * 576
    nope_mask = valid[:, None] & (channel[None, :] < 448)
    raw = tl.load(cache + page_offset + value_offset + channel[None, :], nope_mask, other=0)
    quantized = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    exponent = tl.load(cache + page_offset + STATES * 576 + state[:, None] * SCALE_BYTES
                       + channel[None, :] // (512 // SCALE_BYTES), nope_mask, other=127).to(tl.float32) - 127.
    nope = quantized * tl.exp2(exponent)
    rope_mask = valid[:, None] & (channel[None, :] >= 448)
    rope_offset = 448 + 2 * tl.maximum(channel - 448, 0)
    low = tl.load(cache + page_offset + value_offset + rope_offset[None, :], rope_mask, other=0).to(tl.uint16)
    high = tl.load(cache + page_offset + value_offset + rope_offset[None, :] + 1, rope_mask, other=0).to(tl.uint16)
    rope = (low | (high << 8)).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    value = tl.where(channel[None, :] < 448, nope, rope)
    value = tl.where(valid[:, None], value, 0.).to(tl.bfloat16)
    tl.store(output + row[:, None] * 512 + channel[None, :], value, row[:, None] < count)


def gather(cache, slots):
    """Caller validates cache layout/slot bounds; allocate only BF16 output."""
    if cache.device != slots.device or not cache.is_cuda or slots.dtype not in (torch.int32, torch.int64):
        raise ValueError('Expected integer slots on the same CUDA device as packed cache')
    if (cache.dtype != torch.uint8 or cache.ndim != 3 or cache.shape[-1] not in (584, 592)
            or cache.stride(2) != 1 or cache.stride(1) != cache.shape[2]
            or cache.stride(0) < cache.shape[1] * cache.shape[2]):
        raise ValueError('Expected packed group-32/64 FP8 plus BF16-RoPE pages')
    flat = slots.to(torch.int64).contiguous().reshape(-1)
    output = torch.empty((*slots.shape, 512), dtype=torch.bfloat16, device=cache.device)
    if flat.numel():
        _gather[(triton.cdiv(flat.numel(), 4),)](cache, flat, output, flat.numel(),
            cache.shape[0] * cache.shape[1], PAGE_STRIDE=cache.stride(0),
            STATES=cache.shape[1], BLOCK_ROWS=4, SCALE_BYTES=cache.shape[2]-576, num_warps=4)
    return output
