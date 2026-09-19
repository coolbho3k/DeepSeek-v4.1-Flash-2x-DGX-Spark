# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only, explicit resource envelope for six-request DSpark K<=5."""
from dataclasses import dataclass
import math

CAPACITIES = (6*128*5120*2, 6*128*5120*2, 6*128*1152*2, 6*128*1152*2)
MAX_ROWS = 36
MAX_SLOTS = 216
COUNTERS = MAX_SLOTS*9 + MAX_ROWS*40 + 2 + MAX_SLOTS + 1 + MAX_SLOTS
LAYOUT = (
    ('had_g', 0, 0, (MAX_SLOTS, 5120), 2, 'float16'),
    ('had_u', 1, 0, (MAX_SLOTS, 5120), 2, 'float16'),
    ('gu_g', 2, 0, (MAX_SLOTS, 1152), 2, 'float16'),
    ('gu_u', 3, 0, (MAX_SLOTS, 1152), 2, 'float16'),
    ('activation', 2, MAX_SLOTS*1152*2, (MAX_SLOTS, 1152), 2, 'float16'),
    ('down', 0, MAX_SLOTS*5120*2, (MAX_SLOTS, 5120), 4, 'float32'),
    ('counters', 3, MAX_SLOTS*1152*2, (COUNTERS,), 4, 'int32'),
)


def intervals():
    result = []
    for name, parent, offset, shape, size, _ in LAYOUT:
        end = offset + math.prod(shape)*size
        if offset % size or not 0 <= offset < end <= CAPACITIES[parent]:
            raise ValueError('Invalid cooperative scratch range: '+name)
        if any(p == parent and max(offset, a) < min(end, b) for _, p, a, b in result):
            raise ValueError('Overlapping live cooperative scratch: '+name)
        result.append((name, parent, offset, end))
    return result


@dataclass(frozen=True)
class Policy:
    draft_tokens: int = 3
    verification: str = 'fixed'
    prefix_lengths: tuple[int, ...] = ()

    def __post_init__(self):
        if type(self.draft_tokens) is not int or self.draft_tokens not in (3, 4, 5):
            raise ValueError('Checkpoint-qualified draft envelope is K3/K4/K5')
        if self.verification not in ('fixed', 'ema', 'confidence'):
            raise ValueError('Unknown verification policy')
        lengths = self.prefix_lengths or (self.draft_tokens,)
        if (not isinstance(lengths, tuple) or lengths != tuple(sorted(set(lengths)))
                or any(type(k) is not int or not 1 <= k <= self.draft_tokens for k in lengths)
                or lengths[-1] != self.draft_tokens
                or (self.verification == 'fixed' and lengths != (self.draft_tokens,))):
            raise ValueError('Prefixes must be explicit, ordered, bounded and include full K')
        object.__setattr__(self, 'prefix_lengths', lengths)

    def graph_sizes(self):
        # Target and drafter use the same CLI capture-size list, but their
        # graph managers have distinct request/query-length descriptors.
        sizes = {1, 2, 3, 4, 6, 8}
        sizes.update(n*self.draft_tokens for n in range(1, 7))
        sizes.update(n*(k+1) for n in range(1, 7) for k in self.prefix_lengths)
        assert max(sizes) <= MAX_ROWS
        return tuple(sorted(sizes))

    def target_key(self, requests, prefix):
        if type(requests) is not int or not 1 <= requests <= 6 or prefix not in self.prefix_lengths:
            raise ValueError('Uncaptured request/prefix geometry')
        # Equal total rows do not identify an equivalent attention graph.
        return (requests*(prefix+1), requests, prefix+1)
