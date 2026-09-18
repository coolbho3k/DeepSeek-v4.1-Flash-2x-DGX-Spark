# SPDX-License-Identifier: AGPL-3.0-only
"""Exact hierarchical top-k with device-side live-length pruning.

The fixed-capacity graph never reads a length on the CPU. Short rows bypass
all sorting on the GPU, preserving the native sequential-ID shortcut. Other
rows use the existing IEEE/tie integer ordering and exact sorted merges.
"""
import ast
import inspect
import triton
import triton.language as tl


@triton.jit
def _stage(Source, Counts, Dest, Output, WIDTH: tl.constexpr, SOURCE_STRIDE: tl.constexpr,
           DEST_STRIDE: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr,
           LEVEL: tl.constexpr, FIRST: tl.constexpr, FINAL: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    count = tl.load(Counts + row)
    if count <= K:
        if FINAL:
            out = tl.arange(0, K)
            tl.store(Output + row * K + out, tl.where(out < count, out, -1))
    else:
        live_count = count
        for _ in tl.static_range(LEVEL):
            live_count = tl.cdiv(live_count, BLOCK) * K
        col = block * BLOCK + tl.arange(0, BLOCK)
        active = (col < WIDTH) & (col < live_count)
        if block * BLOCK < live_count:
            if FIRST:
                score = tl.load(Source + row * SOURCE_STRIDE + col, active, other=-float('inf'))
                score = tl.where(score == 0., 0., score)
                bits = score.to(tl.uint32, bitcast=True)
                ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
                ordered = tl.where(score != score, 0xffffffff, ordered).to(tl.uint32).to(tl.int64)
                key = (ordered << 20) + (1048575 - col).to(tl.int64)
                key = tl.where(active, key, 0)
            else:
                key = tl.load(Source + row * SOURCE_STRIDE + col, active, other=0)
            selected = tl.sort(key, descending=True)
            lane = tl.arange(0, BLOCK)
            if FINAL:
                identifier = (1048575 - (selected & 1048575)).to(tl.int32)
                tl.store(Output + row * K + lane, identifier, lane < K)
            else:
                tl.store(Dest + row * DEST_STRIDE + block * K + lane, selected, lane < K)


def select(logits, counts, output, k):
    import torch
    rows, width = logits.shape
    block = 4096 if k <= 1024 else 8192
    source = logits
    level = 0
    while True:
        blocks = triton.cdiv(width, block)
        final = blocks == 1
        next_width = blocks * k
        destination = output if final else torch.empty((rows, next_width), device=logits.device, dtype=torch.int64)
        _stage[(rows, blocks)](source, counts, destination, output, WIDTH=width,
            SOURCE_STRIDE=source.stride(0), DEST_STRIDE=next_width, K=k, BLOCK=block,
            LEVEL=level, FIRST=level == 0, FINAL=final,
            num_warps=8 if block == 4096 else 16, enable_fp_fusion=False)
        if final:
            return
        source, width = destination, next_width
        level += 1


def wrap(original):
    source = inspect.getsource(original)
    begin = source.index('    take = min(k, width)')
    source = source[:begin] + '    _length_aware_select(logits, counts, output, k)\n'
    tree = ast.parse(source)
    definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    definition.decorator_list = []
    namespace = dict(original.__globals__, _length_aware_select=select)
    exec(compile(tree, __file__ + ':wrapper', 'exec'), namespace)
    return namespace[original.__name__]
