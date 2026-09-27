"""V4.1 post-RoPE NVFP4 main KV with searched per-group scales.

Each main-cache state is 256 value bytes followed by 32 scale bytes. This
format is NOT used for SWA. DS41_FP4_KV_MODE selects the scale per group of 16:
  nvfp4_search (default)  every E4M3 scale from amax/6.5 to amax/2.5 (at most 12),
                          keeping one only if its reconstruction has strictly lower
                          SSE; the search starts from E4M3(amax/6), so ties keep the
                          historical bytes and it is never worse than /6 or /4.
  nvfp4_4over6            E4M3(amax/4) only if strictly better than E4M3(amax/6).
  legacy                  E4M3(amax/6).
The format remains exactly 4.5 bits/value, with an implicit outer scale of one.
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


MODES = {'legacy': 0, 'nvfp4_4over6': 1, 'nvfp4_search': 2}


def quantization_mode():
    mode = os.environ.get('DS41_FP4_KV_MODE', 'nvfp4_search')
    if mode not in MODES:
        raise ValueError('DS41_FP4_KV_MODE must be nvfp4_search, nvfp4_4over6 or legacy')
    return mode


# Select once before graph capture. Both workers receive the same profile.
QUANTIZATION_MODE = quantization_mode()
SCALE_MODE = MODES[QUANTIZATION_MODE]


def _writer_geometry(rows, compress_ratio=1):
    # Measured SM121 launch choices; no runtime autotuning or GPU state.
    # Decode exposes independent groups across SMs. Prefill amortizes the
    # slot/address work over a full row; CR2 has fewer live DCP writers.
    if SCALE_MODE == 2:
        # The search encodes each group 13 times: smaller tiles until the grid
        # fills the GPU, then two warps per whole row (no spills at any size).
        if rows <= 32:
            return 4, 1
        if rows <= 512:
            return 8, 1
        return (16, 1) if rows <= 1056 else (32, 2)
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
def _search_scales(groups, amax, codes6, scales6):
    """Best of every E4M3 scale in [amax/6.5, amax/2.5], starting from E4M3(amax/6).

    At most 12 E4M3 codes lie in that 2.6x range (8 per binade); they are tried in
    ascending order and one replaces the best only if its SSE is strictly lower, so
    ties keep E4M3(amax/6). E4M3(amax/4) is in the range, so the result is never
    worse than four-over-six either. Each candidate's SSE is compared against the /6
    reconstruction, SSE_c - SSE_6 = sum((qc - q6) * (qc + q6 - 2|x|)), in _prefer_four's
    integer units u = 2**(floor(log2(amax)) - 12). Wherever either reconstruction is
    nonzero, |x| > scale/4 >= 2**(e-5) (E4M3(amax/6) is >= amax/8, or the 2**-9
    subnormal where that still bounds |x|), so BF16 x and both reconstructions are
    exact multiples of u, each below 2.4 * amax < 19661 u: every product fits int32
    and the 16-term sum is exact in int64. Elsewhere both are zero and the term
    vanishes however x/u rounds. Only per-group scalars carry between candidates; the
    chosen scale is encoded once at the end.
    Returns (codes [G, 16] int32, scale bytes [G] uint8).
    """
    exponent = ((amax.to(tl.uint32, bitcast=True) >> 23) & 255).to(tl.int32)
    inverse_unit = ((266 - exponent) << 23).to(tl.uint32).to(tl.float32, bitcast=True)
    qx2 = 2 * (tl.abs(groups) * inverse_unit[:, None]).to(tl.int32)
    q6 = (_decoded_magnitude(codes6) * scales6.to(tl.float32)[:, None] * inverse_unit[:, None]).to(tl.int32)
    low = tl.div_rn(amax, 6.5)
    first = low.to(tl.float8e4nv)
    first_bits = first.to(tl.uint8, bitcast=True).to(tl.int32) + (first.to(tl.float32) < low).to(tl.int32)
    high = tl.div_rn(amax, 2.5)
    best = tl.zeros(amax.shape, tl.int64)
    best_bits = scales6.to(tl.uint8, bitcast=True).to(tl.int32)
    for step in tl.static_range(12):
        bits = first_bits + step
        candidate = bits.to(tl.uint8).to(tl.float8e4nv, bitcast=True)
        value = candidate.to(tl.float32)
        qc = (_decoded_magnitude(_encode_e2m1(groups, candidate)) * value[:, None]
              * inverse_unit[:, None]).to(tl.int32)
        delta = tl.sum(((qc - q6) * (qc + q6 - qx2)).to(tl.int64), axis=1)
        better = (bits < 0x7F) & (value > 0) & (value <= high) & (delta < best)
        best = tl.where(better, delta, best)
        best_bits = tl.where(better, bits, best_bits)
    scales = best_bits.to(tl.uint8)
    codes = tl.where((best < 0)[:, None], _encode_e2m1(groups, scales.to(tl.float8e4nv, bitcast=True)), codes6)
    return codes, scales


@triton.jit
def _store_row(x, slot, cache, CAPACITY: tl.constexpr,
               PAGE_STRIDE: tl.constexpr, STATES: tl.constexpr,
               SCALE_MODE: tl.constexpr, GROUPS: tl.constexpr, group_start):
    groups = tl.reshape(x, (GROUPS, 16))
    amax = tl.maximum(tl.max(tl.abs(groups), axis=1), 6.0 * (2.0 ** -9))
    scales6 = tl.div_rn(amax, 6.0).to(tl.float8e4nv)
    codes6 = _encode_e2m1(groups, scales6)
    scales = scales6.to(tl.uint8, bitcast=True)
    codes = codes6
    if SCALE_MODE == 2:
        codes, scales = _search_scales(groups, amax, codes6, scales6)
    elif SCALE_MODE == 1:
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
           STATES: tl.constexpr, SCALE_MODE: tl.constexpr, GROUPS: tl.constexpr = 32):
    row = tl.program_id(0)
    group_start = tl.program_id(1) * GROUPS
    x = tl.load(values + row * VALUE_STRIDE + group_start * 16 + tl.arange(0, GROUPS * 16)).to(tl.float32)
    slot = tl.load(slots + row)
    _store_row(x, slot, cache, CAPACITY, PAGE_STRIDE, STATES, SCALE_MODE, GROUPS, group_start)


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
            SCALE_MODE=SCALE_MODE, GROUPS=groups, num_warps=warps,
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
