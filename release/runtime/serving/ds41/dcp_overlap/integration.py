# SPDX-License-Identifier: AGPL-3.0-only
"""Startup-only integration, applied to a freshly prepared private bundle."""
import ast
import functools

from . import policy

MARKER = 'ds41_dcp_overlap_v1'


def wrap_init_device(original):
    @functools.wraps(original)
    def initialize(worker):
        result = original(worker)
        if policy.validate_mode(policy.MODE) != 'off':
            from vllm.distributed import get_dcp_group
            from .transport import prepare
            transport = prepare(get_dcp_group())
            # Low-latency two-rank collectives for small decode messages.
            from ds41 import fastcomm
            fastcomm.prepare()
            fastcomm.set_side_stream(transport.stream)
        return result
    return initialize


def wrap_forward(original):
    """Rewrite only the compressed-attention exchange/compute schedule.

    Metadata, native SWA visibility, sparse ordering/localization, cache bytes,
    sink splitting, original slab size, and SWA-only calls remain unchanged.
    """
    mode = policy.validate_mode(policy.MODE)
    if mode == 'off':
        return original
    from .attention import make_head_attention, step
    from .transport import prepared
    source = getattr(original, '__ds41_patch_source__', '')
    attention = original.__globals__['bf16_sparse_attention_with_lse']
    head_attention = make_head_attention(attention)
    tail = (
        '            partial, lse = bf16_sparse_attention_with_lse(\n'
        '                local_q, swa, sw_ids, sw_lens, sinks=sinks, scale=self.scale, **extra)\n'
        '            if not swa_only:\n'
        '                all_packed = group.all_gather(_ds41_pack_result(partial, lse, rank), dim=0)\n'
        '                _ds41_merge_packed(partial, lse, all_packed, rank, output[rows])\n'
        '            else:\n'
        '                output[rows].copy_(partial)')
    replacement = (
        '            if swa_only:\n'
        '                partial, lse = bf16_sparse_attention_with_lse(\n'
        '                    local_q, swa, sw_ids, sw_lens, sinks=sinks, scale=self.scale, **extra)\n'
        '                output[rows].copy_(partial)\n'
        '            else:\n'
        '                pending = _ds41_overlap_step(\n'
        '                    query_transfer, local_q, swa, sw_ids, sw_lens, sinks, self.scale,\n'
        '                    extra, output[rows], pending, attention=_ds41_head_attention,\n'
        '                    mode=_ds41_overlap_mode)')
    rules = [
        ('    for base, count, indices, lengths in (',
         '    pending = None\n    for base, count, indices, lengths in ('),
        ('                local_q = group.all_gather(local_q, dim=1)',
         "                query_transfer = _ds41_overlap_transport(group).gather(local_q, kind='query')"),
        (tail, replacement),
    ]
    for old, new in rules:
        if source.count(old) != 1:
            raise RuntimeError('Unreviewed DCP overlap integration anchor: ' + old[:90])
        source = source.replace(old, new)
    source += '\n    if pending is not None:\n        pending.finish()\n'
    tree = ast.parse(source)
    definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    definition.decorator_list = []
    namespace = dict(original.__globals__, _ds41_overlap_step=step,
                     _ds41_overlap_transport=prepared, _ds41_head_attention=head_attention,
                     _ds41_overlap_mode=mode)
    exec(compile(tree, __file__ + ':forward', 'exec'), namespace)
    scheduled = namespace[definition.name]
    scheduled.__ds41_patch_source__ = source

    @functools.wraps(original)
    def forward(self, q, output, flashmla_metadata, swa_metadata,
                self_kv_cache, swa_kv_cache, swa_only, *, group=None):
        if policy.MODE != mode:
            raise RuntimeError('DCP overlap selection changed after startup')
        if swa_only:
            return original(self, q, output, flashmla_metadata, swa_metadata,
                            self_kv_cache, swa_kv_cache, swa_only, group=group)
        if group is None:
            from vllm.distributed import get_dcp_group
            group = get_dcp_group()
        with prepared(group).session():
            return scheduled(self, q, output, flashmla_metadata, swa_metadata,
                             self_kv_cache, swa_kv_cache, swa_only, group=group)
    # Runtime guards inspect both this source and globals. A callable object
    # is not substituted for the native function; keep the exact math binding.
    forward.__ds41_patch_source__ = source
    forward.__ds41_overlap_marker__ = MARKER
    forward.__ds41_overlap_mode__ = mode
    forward.__ds41_overlap_original__ = original
    forward.__ds41_overlap_scheduled__ = scheduled
    forward.__ds41_overlap_attention__ = attention
    # fp4.register inspects __globals__ for the selected attention binding.
    # Use a distinct function globals dict, never mutate this module's globals.
    from types import FunctionType
    bindings = dict(original.__globals__)
    bindings.update(forward.__globals__)
    bindings['bf16_sparse_attention_with_lse'] = attention
    result = FunctionType(forward.__code__, bindings, forward.__name__,
        forward.__defaults__, forward.__closure__)
    result.__kwdefaults__ = forward.__kwdefaults__
    result.__dict__.update(forward.__dict__)
    result.__qualname__ = forward.__qualname__
    return result


def forward_admitted(forward, expected_original_text):
    if policy.validate_mode(policy.MODE) == 'off':
        return expected_original_text in getattr(forward, '__ds41_patch_source__', '')
    scheduled = getattr(forward, '__ds41_overlap_scheduled__', None)
    original = getattr(forward, '__ds41_overlap_original__', None)
    dependencies = ('bf16_sparse_attention_with_lse', 'sparse_global_to_local_slots',
                    'partition_indices', '_ds41_pack_result', '_ds41_merge_packed')
    return (scheduled is not None and original is not None
            and all(forward.__globals__.get(k) is scheduled.__globals__.get(k)
                    is original.__globals__.get(k) for k in dependencies)
            and getattr(forward, '__ds41_overlap_marker__', None) == MARKER
            and getattr(forward, '__ds41_overlap_mode__', None) == policy.MODE
            and '_ds41_overlap_step(' in getattr(forward, '__ds41_patch_source__', '')
            and getattr(forward, '__ds41_overlap_attention__', None)
            is forward.__globals__.get('bf16_sparse_attention_with_lse'))


def bind_sparse_mapper(forward, candidate):
    """Preserve the existing post-FP4 sparse-mapper installation on all clones."""
    key = 'sparse_global_to_local_slots'
    functions = [forward]
    if getattr(forward, '__ds41_overlap_marker__', None) == MARKER:
        functions += [forward.__ds41_overlap_scheduled__, forward.__ds41_overlap_original__]
    old = forward.__globals__[key]
    if any(function.__globals__.get(key) is not old for function in functions):
        raise RuntimeError('DCP overlap sparse dependencies diverged before installation')
    for function in functions:
        function.__globals__[key] = candidate
