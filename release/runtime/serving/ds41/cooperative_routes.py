# SPDX-License-Identifier: AGPL-3.0-only
"""Single launch: sparse-bank mapping, FP16 conversion and counter reset.

MiaAI cooperative native ABI uses FP16 route weights (unlike our staged FP32
fallback). This is a numerical candidate, not a bit-identical substitution.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _prepare(X, Ids, Weights, Mapping, HalfX, Local, HalfWeights, Counters,
             ROWS: tl.constexpr, XR: tl.constexpr, XC: tl.constexpr,
             IR: tl.constexpr, IC: tl.constexpr, WR: tl.constexpr, WC: tl.constexpr,
             B: tl.constexpr):
    # One bounded vector per row instead of a single 131072-element program.
    # Routing and counters have one writer; later kernels on the same stream
    # observe completion of the complete preparation grid.
    row = tl.program_id(0)
    pos = tl.arange(0, B)
    x = tl.load(X+row*XR+pos*XC, pos < 5120, other=0.)
    tl.store(HalfX+row*5120+pos, x.to(tl.float16), pos < 5120)
    if row == 0:
        slot = tl.arange(0, 256)
        raw = tl.load(Ids+(slot//6)*IR+(slot%6)*IC, slot < ROWS*6, other=-1).to(tl.int64)
        mapped = tl.load(Mapping+tl.where((raw >= 0)&(raw < 384), raw, 384))
        tl.store(Local+slot, mapped, slot < ROWS*6)
        rw = tl.load(Weights+(slot//6)*WR+(slot%6)*WC, slot < ROWS*6, other=0.)
        tl.store(HalfWeights+slot, rw.to(tl.float16), slot < ROWS*6)
        ctr = tl.arange(0, 4096)
        tl.store(Counters+ctr, 0, ctr < 2547)


def prepare(x, ids, weights, mapping, counters):
    half_x = torch.empty(x.shape, device=x.device, dtype=torch.float16)
    local = torch.empty(ids.shape, device=x.device, dtype=torch.int64)
    half_weights = torch.empty(weights.shape, device=x.device, dtype=torch.float16)
    _prepare[(len(x),)](x, ids, weights, mapping, half_x, local, half_weights, counters,
        len(x), *x.stride(), *ids.stride(), *weights.stride(),
        8192, num_warps=8, enable_fp_fusion=False)
    return half_x, local, half_weights
