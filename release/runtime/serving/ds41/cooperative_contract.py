# SPDX-License-Identifier: AGPL-3.0-only
"""Byte-exact scratch alias contract for MiaAI's fixed-shape two-stage MoE.

All ranges are inside the existing serialized dispatcher temps, NEVER its
completion locks. Counters must be reset on EVERY call because fallback uses
these same temps. No additional persistent device storage is allocated.
"""
UPSTREAM_COMMIT = 'b9c49e90bdcc6f1e0192feb57214df11b67d36aa'
CAPACITIES = (6*128*5120*2, 6*128*5120*2, 6*128*1152*2, 6*128*1152*2)
# name, parent temp, byte offset, shape, element bytes, torch dtype
LAYOUT = (
    ('had_g', 0, 0, (144, 5120), 2, 'float16'),
    ('had_u', 1, 0, (144, 5120), 2, 'float16'),
    ('gu_g', 2, 0, (144, 1152), 2, 'float16'),
    ('gu_u', 3, 0, (144, 1152), 2, 'float16'),
    ('activation', 2, 144*1152*2, (144, 1152), 2, 'float16'),
    ('down', 0, 144*5120*2, (144, 5120), 4, 'float32'),
    ('counters', 3, 144*1152*2, (2547,), 4, 'int32'),
)


def intervals():
    import math
    result = []
    for name, parent, offset, shape, size, dtype in LAYOUT:
        end = offset+math.prod(shape)*size
        if offset % size or not 0 <= offset < end <= CAPACITIES[parent]:
            raise ValueError('Invalid cooperative scratch range: '+name)
        if any(p == parent and max(offset, a) < min(end, b) for _, p, a, b in result):
            raise ValueError('Overlapping live cooperative scratch: '+name)
        result.append((name, parent, offset, end))
    return result


def eligible_shape(x_shape, ids_shape):
    return (len(x_shape) == 2 and 1 <= x_shape[0] <= 24 and x_shape[1] == 5120
        and tuple(ids_shape) == (x_shape[0], 6))


def selected_shape(x_shape, ids_shape):
    # Full MiaAI cooperative dispatch, including C1 speculative verification.
    return eligible_shape(x_shape, ids_shape)


def forward_replacements():
    # Insert inside the existing try/lock, AFTER Bank/Workspace initialization
    # and validation; retain the exact registered GroupedDispatcher owner.
    anchor = "if bank.owner is not experts or len(experts) != len(bank.keys):\n                raise ValueError('Expert bank changed after native pointer capture')"
    addition = '''
            if _ds41_coop_eligible(x.shape, ids.shape):
                result = _ds41_coop_call(work, bank, x, ids, weights)
                self.last_schedule = dict(mode='miaai_two_stage_cooperative',
                    assignments=ids.numel(), fallback_experts=0,
                    host_count_readback=False, counts=None)
                work.ready.record(stream)
                work.pending, work.stream_id = True, stream.cuda_stream
                return result'''
    return [(anchor, anchor+addition)]
