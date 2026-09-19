# SPDX-License-Identifier: AGPL-3.0-only
"""Extend MiaAI's fixed target kernel to 36 rows without changing its math.

Derived from MiaAI Lab / Wesley Young and contributors, upstream revision
b9c49e90bdcc6f1e0192feb57214df11b67d36aa; ExLlamaV3 by Turboderp.
Keep corresponding original license/notices with generated sources.
"""
import hashlib

WRAPPER_SHA = '8b94fe324029dceea5cbd1685ec02ad32d3839a5a975e1a1601acc8ba877f10f'
KERNEL_SHA = '7b6eaf25dd48d77a22a6a7d7e7c12ddde0d3b8ae81a5fb85c36c714e58ba97f4'


def transform(wrapper, kernel):
    for raw, expected in ((wrapper, WRAPPER_SHA), (kernel, KERNEL_SHA)):
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Changed pinned C6 native parent')
    def edit(raw, before, after):
        if raw.count(before) != 1:
            raise ValueError('Changed capacity anchor: '+before.decode())
        return raw.replace(before, after)
    wrapper = edit(wrapper, b'ROWS_MAX = 24;', b'ROWS_MAX = 36;')
    wrapper = edit(wrapper, b'144 routed slots and 24 output rows', b'216 routed slots and 36 output rows')
    wrapper += b'\nextern "C" int goal50_coop_experiment() { return 401; }\n'
    kernel = edit(kernel, b'p.slots_max = 144; p.rows_max = 24;', b'p.slots_max = 216; p.rows_max = 36;')
    kernel = edit(kernel, b'p.ctr_a_len = 1296; p.ctr_b_len = 960;', b'p.ctr_a_len = 1944; p.ctr_b_len = 1440;')
    old = b'// Local DS41 C6 adaptation: 24 physical rows / 144 routed slots, ABI 2.'
    new = b'// Local DS41 K5/C6 adaptation: 36 physical rows / 216 routed slots, ABI 2.'
    return edit(wrapper, old, new), edit(kernel, old, new)
