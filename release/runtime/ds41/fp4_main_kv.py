"""V4.1 post-RoPE main KV: E2M1 values, E4M3/group16, no global scale.

Each main-cache state is 256 value bytes followed by 32 scale bytes. This
format is NOT used for SWA or the separately specified MXFP4 indexer cache.
Writes require unique live slots and finite post-RoPE BF16 values within the
model's trained range. Negative slots are padding. Native allocator callers
may skip host bounds checks; kernels still mask out-of-range memory accesses.
"""
import torch
import triton
import triton.language as tl

STATE_BYTES = 288
MAX_WRITE_ROWS = 1056
MAX_GATHER_ROWS = 32 * 512


@triton.jit
def _store_row(x, slot, cache, CAPACITY: tl.constexpr,
               PAGE_STRIDE: tl.constexpr, STATES: tl.constexpr):
    groups = tl.reshape(x, (32, 16))
    amax = tl.maximum(tl.max(tl.abs(groups), axis=1), 6.0 * (2.0 ** -9))
    scales = tl.div_rn(amax, 6.0).to(tl.float8e4nv)
    normalized = tl.div_rn(groups, scales.to(tl.float32)[:, None])
    magnitude = tl.abs(normalized)
    # E2M1 round-to-nearest-even; adjacent code parity selects exact ties.
    codes = ((magnitude > 0.25).to(tl.int32)
             + (magnitude >= 0.75).to(tl.int32)
             + (magnitude > 1.25).to(tl.int32)
             + (magnitude >= 1.75).to(tl.int32)
             + (magnitude > 2.5).to(tl.int32)
             + (magnitude >= 3.5).to(tl.int32)
             + (magnitude > 5.0).to(tl.int32))
    signs = (groups.to(tl.uint32, bitcast=True) >> 31).to(tl.int32)
    codes = tl.reshape(codes | (signs << 3), (256, 2))
    low, high = tl.split(codes)
    packed = (low | (high << 4)).to(tl.uint8)
    live = (slot >= 0) & (slot < CAPACITY)
    safe = tl.where(live, slot, 0)
    offset = (safe // STATES) * PAGE_STRIDE + (safe % STATES) * 288
    tl.store(cache + offset + tl.arange(0, 256), packed, live)
    tl.store(cache + offset + 256 + tl.arange(0, 32),
             scales.to(tl.uint8, bitcast=True), live)


@triton.jit
def _store(values, slots, cache, CAPACITY: tl.constexpr,
           VALUE_STRIDE: tl.constexpr, PAGE_STRIDE: tl.constexpr,
           STATES: tl.constexpr):
    row = tl.program_id(0)
    x = tl.load(values + row * VALUE_STRIDE + tl.arange(0, 512)).to(tl.float32)
    slot = tl.load(slots + row)
    _store_row(x, slot, cache, CAPACITY, PAGE_STRIDE, STATES)


@triton.jit
def _gather(cache, slots, output, count, CAPACITY: tl.constexpr,
            PAGE_STRIDE: tl.constexpr, STATES: tl.constexpr,
            BLOCK_ROWS: tl.constexpr):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    channel = tl.arange(0, 512)
    slot = tl.load(slots + row, row < count, other=-1)
    live = (row < count) & (slot >= 0) & (slot < CAPACITY)
    safe = tl.where(live, slot, 0)
    offset = ((safe // STATES) * PAGE_STRIDE + (safe % STATES) * 288)[:, None]
    packed = tl.load(cache + offset + channel[None, :] // 2, live[:, None], other=0)
    code = (packed.to(tl.int32) >> ((channel[None, :] % 2) * 4)) & 15
    magnitude = code & 7
    exponent = magnitude >> 1
    mantissa = magnitude & 1
    value = tl.where(exponent == 0, mantissa.to(tl.float32) * 0.5,
                     (1.0 + mantissa.to(tl.float32) * 0.5)
                     * tl.exp2(exponent.to(tl.float32) - 1.0))
    value = tl.where((code & 8) != 0, -value, value)
    scale = tl.load(cache + offset + 256 + channel[None, :] // 16,
                    live[:, None], other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    restored = tl.where(live[:, None], value * scale, 0.0).to(tl.bfloat16)
    tl.store(output + row[:, None] * 512 + channel[None, :], restored, row[:, None] < count)


def _layout(cache):
    if (not cache.is_cuda or cache.dtype != torch.uint8 or cache.ndim != 3
            or cache.shape[1] < 1 or cache.shape[2] != STATE_BYTES
            or cache.stride(2) != 1 or cache.stride(1) != STATE_BYTES
            or cache.stride(0) < cache.shape[1] * STATE_BYTES):
        raise ValueError('Expected CUDA FP4 main pages [pages, states, 288], contiguous within each page')


def _indices(cache, slots, check_bounds):
    if slots.device != cache.device or slots.dtype not in (torch.int32, torch.int64):
        raise ValueError('Expected integer slots on the cache device')
    if check_bounds and (slots >= cache.shape[0] * cache.shape[1]).any().item():
        raise ValueError('Main-cache slot exceeds allocated capacity')
    return slots.to(torch.int64).contiguous().reshape(-1)


def store(cache, values, slots, *, check_bounds=True):
    """Quantize post-RoPE values into caller-owned main-cache pages in place."""
    _layout(cache)
    if (values.device != cache.device or values.dtype != torch.bfloat16
            or values.ndim != 2 or values.shape[1] != 512
            or not 0 <= values.shape[0] <= MAX_WRITE_ROWS
            or values.stride(1) != 1 or values.stride(0) < 512
            or slots.ndim != 1 or len(slots) != len(values)):
        raise ValueError('Expected at most 1056 BF16 main states [rows, 512] and one slot per row')
    flat = _indices(cache, slots, check_bounds)
    if len(values):
        _store[(len(values),)](values, flat, cache,
            CAPACITY=cache.shape[0]*cache.shape[1], VALUE_STRIDE=values.stride(0),
            PAGE_STRIDE=cache.stride(0), STATES=cache.shape[1], num_warps=4,
            enable_fp_fusion=False)


def gather(cache, slots, *, check_bounds=True):
    """Expand only selected sparse main states, bounded to a 16 MiB output."""
    _layout(cache)
    if slots.numel() > MAX_GATHER_ROWS:
        raise ValueError('Gather exceeds the bounded 32-token by 512-key workspace')
    flat = _indices(cache, slots, check_bounds)
    output = torch.empty((*slots.shape, 512), device=cache.device, dtype=torch.bfloat16)
    if flat.numel():
        _gather[(triton.cdiv(flat.numel(), 4),)](cache, flat, output, flat.numel(),
            CAPACITY=cache.shape[0]*cache.shape[1], PAGE_STRIDE=cache.stride(0),
            STATES=cache.shape[1], BLOCK_ROWS=4, num_warps=4,
            enable_fp_fusion=False)
    return output
