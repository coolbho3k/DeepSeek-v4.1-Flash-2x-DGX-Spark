# SPDX-License-Identifier: AGPL-3.0-only
"""One-pass packed sparse split-K attention candidate; no runtime registration.

Uses the existing exact FP4/FP8 cache decoder and two-term BF16 probabilities.
Only the small-row split path changes; prefill and cache allocation are retained.
"""
import ast
import triton
import triton.language as tl
from .fused_sparse_attention import _selected


@triton.jit
def _segment(q, cache, indices, length, maximum, total, high, low, error,
             WIDTH: tl.constexpr, CAPACITY: tl.constexpr, PAGE_STRIDE: tl.constexpr,
             STATES: tl.constexpr, FP4: tl.constexpr, SCALE: tl.constexpr,
             BN: tl.constexpr, SPLIT, SPLITS: tl.constexpr):
    for begin in range(SPLIT, tl.cdiv(length, BN), SPLITS):
        position = begin * BN + tl.arange(0, BN)
        kv, live, invalid = _selected(cache, indices, position, length, WIDTH,
                                      CAPACITY, PAGE_STRIDE, STATES, FP4)
        tl.atomic_or(error, 1, mask=tl.sum(invalid.to(tl.int32), 0) > 0, sem='relaxed')
        scores = tl.dot(q, tl.trans(kv)).to(tl.float32) * SCALE
        scores = tl.where(live[None, :], scores, float('-inf'))
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        safe = tl.where(new_maximum == float('-inf'), 0., new_maximum)
        finite = tl.abs(new_maximum) < float('inf')
        alpha = tl.where(finite, tl.exp(maximum - safe), 0.)
        p = tl.where(live[None, :] & finite[:, None], tl.exp(scores - safe[:, None]), 0.)
        total = total * alpha + tl.sum(p, 1)
        total = tl.where(new_maximum == float('inf'), 1., total)
        p_hi = p.to(tl.bfloat16)
        p_lo = (p - p_hi.to(tl.float32)).to(tl.bfloat16)
        high = tl.dot(p_hi, kv, high * alpha[:, None])
        low = tl.dot(p_lo, kv, low * alpha[:, None])
        maximum = new_maximum
    return maximum, total, high, low


@triton.jit
def _online(query, swa, si, sl, main, ci, cl, sinks, partial, local_lse, error,
            HEADS: tl.constexpr, Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr,
            SW: tl.constexpr, SC: tl.constexpr, SP: tl.constexpr, SS: tl.constexpr,
            CW: tl.constexpr, CC: tl.constexpr, CP: tl.constexpr, CS: tl.constexpr,
            MAIN: tl.constexpr, MAIN_FP4: tl.constexpr, SINKS: tl.constexpr,
            SINK_STRIDE: tl.constexpr, SCALE: tl.constexpr,
            BH: tl.constexpr, BN: tl.constexpr, SPLITS: tl.constexpr):
    token, tile, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    h = tile * BH + tl.arange(0, BH)
    d = tl.arange(0, 512)
    q = tl.load(query + token * Q0 + h[:, None] * Q1 + d[None, :] * Q2)
    if SINKS:
        maximum = tl.where(split == 0, tl.load(sinks + h * SINK_STRIDE).to(tl.float32), float('-inf'))
        total = tl.full((BH,), 1., tl.float32)
    else:
        maximum = tl.full((BH,), float('-inf'), tl.float32)
        total = tl.full((BH,), 0., tl.float32)
    high = tl.full((BH, 512), 0., tl.float32)
    low = tl.full((BH, 512), 0., tl.float32)
    length = tl.minimum(tl.maximum(tl.load(sl + token), 0), SW)
    maximum, total, high, low = _segment(q, swa, si + token * SW, length,
        maximum, total, high, low, error, SW, SC, SP, SS, False, SCALE, BN, split, SPLITS)
    if MAIN:
        length = tl.minimum(tl.maximum(tl.load(cl + token), 0), CW)
        maximum, total, high, low = _segment(q, main, ci + token * CW, length,
            maximum, total, high, low, error, CW, CC, CP, CS, MAIN_FP4, SCALE, BN, split, SPLITS)
    lse = tl.where(maximum == float('-inf'), float('-inf'), maximum + tl.log(total))
    value = tl.where(((total > 0) & (tl.abs(lse) < float('inf')))[:, None],
                      (high + low) / total[:, None], 0.)
    tl.store(partial + ((token * HEADS + h[:, None]) * SPLITS + split) * 512 + d[None, :], value)
    tl.store(local_lse + (token * HEADS + h) * SPLITS + split, lse)


@triton.jit
def _merge(partial, local_lse, output, normalizers, SPLITS: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    d = tile * 128 + tl.arange(0, 128)
    split = tl.arange(0, SPLITS)
    lse = tl.load(local_lse + row * SPLITS + split)
    maximum = tl.max(lse, 0)
    safe = tl.where(tl.abs(maximum) < float('inf'), maximum, 0.)
    weights = tl.where(tl.abs(lse) < float('inf'), tl.exp(lse - safe), 0.)
    total = tl.sum(weights, 0)
    weights = tl.where((total > 0) & (tl.abs(maximum) < float('inf')), weights / total, 0.)
    values = tl.load(partial + (row * SPLITS + split[:, None]) * 512 + d[None, :])
    tl.store(output + row * 512 + d, tl.sum(values * weights[:, None], 0))
    if tile == 0:
        normalizer = tl.where(tl.abs(maximum) == float('inf'), maximum, maximum + tl.log(total))
        tl.store(normalizers + row, normalizer * 1.4426950408889634)


def wrap(original):
    """Clone the qualified wrapper, replacing only its four split launches."""
    source = original.__ds41_patch_source__
    begin = source.index('        _split_attention[grid](')
    end = source.index('    _ds41_check_flags(error', begin)
    replacement = (
        '        _online(grid, arguments, partial, local_lse, error, common, split_k)\n'
        '        _online_merge[(query.shape[0]*query.shape[1], 4)](\n'
        '            partial, local_lse, output, lse, SPLITS=split_k,\n'
        '            num_warps=4, enable_fp_fusion=False)\n')
    source = source[:begin] + replacement + source[end:]
    source = source.replace('        global_lse = torch.empty(query.shape[:2], device=query.device, dtype=torch.float32)\n', '')
    source = source.replace('        stage_args = (*arguments, partial, local_lse, global_lse, error)\n', '')
    def launch(grid, args, partial, local_lse, error, common, splits):
        _online[grid](*args, partial, local_lse, error, **common, SPLITS=splits)
    tree = ast.parse(source)
    definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    definition.decorator_list = []
    namespace = dict(original.__globals__, _online=launch, _online_merge=_merge)
    exec(compile(tree, __file__ + ':wrapper', 'exec'), namespace)
    result = namespace[original.__name__]
    result.__ds41_patch_source__ = source
    return result
