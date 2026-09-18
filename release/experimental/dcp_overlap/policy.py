# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only schedule policy; no imports of torch, CUDA, or the serving stack."""

MODE = 'off'  # Changed only in a newly prepared, private candidate bundle.
MODES = ('off', 'query', 'balanced', 'concurrent')
MAX_ROWS = 512
HEADS = 32
CHANNELS = 512
MAX_TRANSFERS = 2


def validate_mode(mode):
    if mode not in MODES:
        raise ValueError('Unknown DCP overlap mode')
    return mode


def head_schedule(rows, mode):
    """Keep the original 16-head decode / 32-head wide-prefill MMA tiles.

    Splitting the heads never splits a reduction over keys or channels. Wide
    prefill cannot use 16-head calls without changing the selected MMA tile,
    so its result exchange overlaps the next query slab instead.
    """
    validate_mode(mode)
    if type(rows) is not int or not 1 <= rows <= MAX_ROWS or mode == 'off':
        raise ValueError('An enabled, bounded nonempty DCP slab is required')
    if mode == 'balanced' and rows < 32:
        return ((0, 16), (16, 32))
    return ((0, 32),)


def working_bytes(rows):
    """Explicit transport tensor bytes, excluding attention and graph scratch.

    Q receives two ranks of BF16. Result send/receive are full FP32, including
    the LSE. A deferred preceding slab may retain its results until the next
    local attention completes. This is accounting, NOT a measured RAM peak.
    """
    if type(rows) is not int or not 1 <= rows <= MAX_ROWS:
        raise ValueError('Invalid DCP slab size')
    return dict(query_receive=2 * rows * HEADS * CHANNELS * 2,
                result_send=rows * HEADS * (CHANNELS + 1) * 4,
                result_receive=2 * rows * HEADS * (CHANNELS + 1) * 4)
