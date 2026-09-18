# SPDX-License-Identifier: AGPL-3.0-only
"""Retain the reachable SWA checkpoint below a speculative partial tail.

The native lookup peeks one physical block past a candidate before dropping
it. Sparse retention must also retain the preceding aligned boundary when
that peek is beyond the prompt's complete hashed blocks. Keep the original
boundary too: later decode may complete it. No attention, slots or KV bytes
are changed; retained free blocks remain evictable through the native pool.
"""
import hashlib
import os
from pathlib import Path

SOURCE_SHA = '128b98a0511f67d32f44767aa1658776a8246374b9eb8461d397e683ff3d984d'
_installed = None


def replay_boundaries(boundaries, block_size):
    return tuple(dict.fromkeys((*boundaries, *(max(0, b-block_size) for b in boundaries))))


def register():
    global _installed
    if any(os.environ.get(key) != '1' for key in ('DS41_ENABLE_COMBINED_MIAAI', 'DS41_ENABLE_DSPARK')):
        raise ValueError('Speculative prefix retention requires the explicit DS41 DSpark runtime')
    from vllm.v1.core import single_type_kv_cache_manager as native
    if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() != SOURCE_SHA:
        raise RuntimeError('Unreviewed native SWA prefix-retention source')
    cls = native.SlidingWindowManager
    if _installed is not None:
        if cls.reachable_block_mask.__func__ is not _installed:
            raise RuntimeError('Speculative prefix-retention binding changed')
        return
    original = cls.reachable_block_mask.__func__

    def mask(cls, start_block, end_block, alignment_tokens, kv_cache_spec, use_eagle,
             retention_interval=None, reachable_boundaries=(), dcp_world_size=1):
        if use_eagle and retention_interval is not None and alignment_tokens is not None:
            if (kv_cache_spec.block_size != 32 or kv_cache_spec.sliding_window != 128
                    or kv_cache_spec.state_content_size_bytes != 584 or dcp_world_size != 1):
                raise ValueError('Only the existing replicated FP8 DS41 SWA layout is supported')
            reachable_boundaries = replay_boundaries(reachable_boundaries, 32)
        return original(cls, start_block, end_block, alignment_tokens, kv_cache_spec,
                        use_eagle, retention_interval, reachable_boundaries, dcp_world_size)

    cls.reachable_block_mask = classmethod(mask)
    _installed = mask
