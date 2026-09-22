"""V4.1 post-RoPE NVFP4 main KV with four-over-six scale selection.

Each main-cache state is 256 value bytes followed by 32 scale bytes. This
format is NOT used for SWA or the separately specified MXFP4 indexer cache.
For each group of 16, choose E4M3(amax/4) only if its reconstructed E2M1
values have lower SSE than the historical E4M3(amax/6) candidate. Ties keep
the historical bytes. The format remains exactly 4.5 bits/value, with an
implicit outer scale of one. DS41_FP4_KV_MODE=legacy selects the old writer.
Caller-owned (including display-backed) pages and all readers are unchanged. Triton hashes the updated writer at JIT time.
Writes require unique live slots and finite post-RoPE BF16 values within the
model's trained range. Negative slots are padding. Native allocator callers
may skip host bounds checks; kernels still mask out-of-range memory accesses.
"""
import os

import torch
import triton
import triton.language as tl

STATE_BYTES = 288
MAX_WRITE_ROWS = 1056
MAX_GATHER_ROWS = 32 * 512


def quantization_mode():
    mode = os.environ.get('DS41_FP4_KV_MODE', 'nvfp4_4over6')
    if mode not in ('nvfp4_4over6', 'legacy'):
        raise ValueError('DS41_FP4_KV_MODE must be nvfp4_4over6 or legacy')
    return mode


# Select once before graph capture. Both workers receive the same profile.
QUANTIZATION_MODE = quantization_mode()
FOUR_OVER_SIX = QUANTIZATION_MODE == 'nvfp4_4over6'


def _writer_geometry(rows, compress_ratio=1):
    # Measured SM121 launch choices; no runtime autotuning or GPU state.
    # Decode exposes independent groups across SMs. Prefill amortizes the
    # slot/address work over a full row; CR2 has fewer live DCP writers.
    if rows <= 32:
        return 4, 1
    if rows <= 512:
        return 16, 1
    return 32, 2 if compress_ratio == 2 else 1


@triton.jit
def _encode_e2m1(groups, scales):
    # BF16/E4M3 ratios cannot approach an E2M1 midpoint closer than the
    # input's BF16 step, unless they hit it exactly. FP16 rounding removes
    # reciprocal noise at exact ties without moving any other value across
    # an E2M1 midpoint. Keep the native nearest-even FP4 converter.
    inverse = tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;",
        constraints="=f,f", args=[scales.to(tl.float32)], dtype=tl.float32,
        is_pure=True, pack=1)
    normalized = (groups * inverse[:, None]).to(tl.float16).to(tl.float32)
    low, high = tl.split(tl.reshape(normalized, (groups.shape[0], 8, 2)))
    packed = tl.inline_asm_elementwise(
        "{ .reg .b8 q; cvt.rn.satfinite.e2m1x2.f32 q, $2, $1; cvt.u32.u8 $0, q; }",
        constraints="=r,f,f", args=[low, high], dtype=tl.uint32,
        is_pure=True, pack=1)
    return tl.interleave(packed & 15, packed >> 4).to(tl.int32)


@triton.jit
def _decoded_magnitude(codes):
    magnitude = codes & 7
    bits = (((magnitude >> 1) + 126) << 23) | ((magnitude & 1) << 22)
    return tl.where(magnitude < 2, magnitude.to(tl.float32) * 0.5,
                    bits.to(tl.uint32).to(tl.float32, bitcast=True))


@triton.jit
def _prefer_four(groups, amax, codes6, scales6, codes4, scales4):
    # Compare SSE4-SSE6 in exact integer units; the common x*x cancels.
    # Let e=floor(log2(amax)), u=2**(e-12). Wherever q4!=q6, |x| is
    # >=amax/32, so BF16 x and both reconstructions are exact multiples
    # of u. Else the contribution is zero regardless of rounding x/u.
    # x/u<8192 and q/u<=8704, so the 16-term signed sum fits int32.
    exponent = ((amax.to(tl.uint32, bitcast=True) >> 23) & 255).to(tl.int32)
    inverse_unit = ((266 - exponent) << 23).to(tl.uint32).to(tl.float32, bitcast=True)
    q6 = _decoded_magnitude(codes6) * scales6.to(tl.float32)[:, None]
    q4 = _decoded_magnitude(codes4) * scales4.to(tl.float32)[:, None]
    i6 = (q6 * inverse_unit[:, None]).to(tl.int32)
    i4 = (q4 * inverse_unit[:, None]).to(tl.int32)
    ix = (tl.abs(groups) * inverse_unit[:, None]).to(tl.int32)
    difference = (i4 - i6) * (i4 + i6 - 2 * ix)
    return tl.sum(difference, axis=1) < 0


@triton.jit
def _store_row(x, slot, cache, CAPACITY: tl.constexpr,
               PAGE_STRIDE: tl.constexpr, STATES: tl.constexpr,
               FOUR_OVER_SIX: tl.constexpr, GROUPS: tl.constexpr, group_start):
    groups = tl.reshape(x, (GROUPS, 16))
    amax = tl.maximum(tl.max(tl.abs(groups), axis=1), 6.0 * (2.0 ** -9))
    scales6 = tl.div_rn(amax, 6.0).to(tl.float8e4nv)
    codes6 = _encode_e2m1(groups, scales6)
    scales = scales6.to(tl.uint8, bitcast=True)
    codes = codes6
    if FOUR_OVER_SIX:
        # Saturate the additional candidate so /4 cannot overflow E4M3 for
        # groups still representable by /6. Preserve the historical /6 path.
        scales4 = tl.minimum(tl.div_rn(amax, 4.0), 448.0).to(tl.float8e4nv)
        codes4 = _encode_e2m1(groups, scales4)
        use4 = _prefer_four(groups, amax, codes6, scales6, codes4, scales4)
        scales = tl.where(use4, scales4.to(tl.uint8, bitcast=True),
                          scales6.to(tl.uint8, bitcast=True))
        codes = tl.where(use4[:, None], codes4, codes6)
    codes = tl.reshape(codes, (GROUPS * 8, 2))
    low, high = tl.split(codes)
    packed = (low | (high << 4)).to(tl.uint8)
    live = (slot >= 0) & (slot < CAPACITY)
    safe = tl.where(live, slot, 0)
    offset = (safe // STATES) * PAGE_STRIDE + (safe % STATES) * 288
    tl.store(cache + offset + group_start * 8 + tl.arange(0, GROUPS * 8), packed, live)
    tl.store(cache + offset + 256 + group_start + tl.arange(0, GROUPS),
             scales, live)


@triton.jit
def _store(values, slots, cache, CAPACITY: tl.constexpr,
           VALUE_STRIDE: tl.constexpr, PAGE_STRIDE: tl.constexpr,
           STATES: tl.constexpr, FOUR_OVER_SIX: tl.constexpr, GROUPS: tl.constexpr = 32):
    row = tl.program_id(0)
    group_start = tl.program_id(1) * GROUPS
    x = tl.load(values + row * VALUE_STRIDE + group_start * 16 + tl.arange(0, GROUPS * 16)).to(tl.float32)
    slot = tl.load(slots + row)
    _store_row(x, slot, cache, CAPACITY, PAGE_STRIDE, STATES, FOUR_OVER_SIX, GROUPS, group_start)


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
        groups, warps = _writer_geometry(len(values))
        _store[(len(values), 32 // groups)](values, flat, cache,
            CAPACITY=cache.shape[0]*cache.shape[1], VALUE_STRIDE=values.stride(0),
            PAGE_STRIDE=cache.stride(0), STATES=cache.shape[1],
            FOUR_OVER_SIX=FOUR_OVER_SIX, GROUPS=groups, num_warps=warps,
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
