# SPDX-License-Identifier: AGPL-3.0-only
"""Prepare a minimal underfilled-row experiment; does not compile or install.

The parent is MiaAI's grouped kernel with the recipe's existing K3/MUL1
adaptation. Preserve its notices and every arithmetic expression. Skip only
16-row accumulator blocks with no live output rows, using a CTA-uniform
condition. This first candidate leaves the 64-row ABI, scratch, input-load
pipeline and shared-memory allocation unchanged. A true smaller-tile
specialization is a separate occupancy/bandwidth experiment.
"""
import hashlib

PARENT_SHA='83727ef8f2efd6a84ebf4053cf87b166d43080d346bc1e67fc064445304c9d86'
ANCHOR='''            for (int mb = 0; mb < MB; ++mb)
            {
                FragA fa[NA];'''
REPLACEMENT='''            for (int mb = 0; mb < MB; ++mb)
            {
                // DS41 experiment: rows is uniform across this CTA. Padded
                // accumulator blocks have no output stores in either epilogue.
                if (mb * 16 >= rows) break;
                FragA fa[NA];'''


def transform(source):
    if hashlib.sha256(source).hexdigest()!=PARENT_SHA:
        raise ValueError('Unexpected grouped-prefill parent source')
    text=source.decode('utf-8')
    if text.count(ANCHOR)!=1:raise ValueError('Ambiguous grouped main loop')
    result=text.replace(ANCHOR,REPLACEMENT).encode('utf-8')
    if result==source:raise ValueError('Missing row guard')
    return result
