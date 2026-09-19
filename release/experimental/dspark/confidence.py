# SPDX-License-Identifier: AGPL-3.0-only
"""SM121 admission for the existing device-driven flattened indexer path.

Upstream advertises this path on Hopper only. Our SM121 FP4 adapter consumes
the same flattened query/table/causal-bound tensors. No metadata arithmetic,
cache writer, quantizer, confidence head or rejection sampler is replaced.
Enable only in the frozen confidence candidate; GPU mismatch/replay tests and
full-model qualification are required before publication.
"""
import hashlib
import functools
import json
from pathlib import Path

INDEXER_SHA = '392d93da110ec942db3ce17895dc358bf92501edb01a8b9c19b6bdfab917bb7f'
CACHE_SHA = '16ecd91362de27abd91232c0a48865962939017336d57a2c2585dcfee6689575'
ADAPTIVE_SHA = 'f31f039ad279cdc25933b6b769ed06d3a333e9892740d3429d519b2a9ed0c2d7'
GRAPH_MODE = (
    'self.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE',
    'self.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY',
)
SLOT_VIEW = (
    '        slots = self.slot_mappings if out is None else out\n',
    '        from .dspark_experiment.confidence import active_query_starts\n'
    '        query_start_loc = active_query_starts(self, idx_mapping, query_start_loc)\n'
    '        slots = self.slot_mappings if out is None else out\n',
)


def valid_start_shape(requests, maximum, shape, stride):
    return (type(requests) is int and type(maximum) is int and 0 <= requests <= maximum <= 6
        and len(shape) == 1 and requests+1 <= shape[0] <= maximum+1 and stride == 1)


def active_query_starts(tables, mapping, starts):
    """A zero-copy active prefix of native confidence's persistent GPU buffer.

    The native slot kernel launches mapping.shape[0]+1 programs and reads no
    offset beyond that prefix. Retain exact-shape checking in our cache hook;
    never resize/overwrite the padded buffer used by attention or graphs.
    """
    import torch
    if (mapping.ndim != 1 or starts.ndim != 1
            or not valid_start_shape(mapping.shape[0],tables.max_num_reqs,starts.shape,starts.stride(0))
            or starts.dtype != torch.int32 or starts.device != tables.device):
        raise ValueError('Invalid native padded query-start buffer')
    return starts[:mapping.shape[0]+1]


def observe_budget(counts, scheduled, verified):
    """CPU counters only; no request IDs, tokens, GPU reads or policy changes."""
    if any(type(x) is not int or x < 0 for x in scheduled) or type(verified) is not int:
        raise ValueError('Invalid native confidence budget receipt')
    proposed=sum(scheduled)
    if not 0 <= verified <= proposed:raise ValueError('Native confidence budget exceeds proposals')
    counts['steps'] += 1
    counts['request_rounds'] += len(scheduled)
    counts['scheduled_drafts'] += proposed
    counts['verified_drafts'] += verified
    counts['trimmed_steps'] += int(verified < proposed)
    return counts['steps']==1 or counts['steps']%128==0 or (verified<proposed and counts['trimmed_steps']==1)


def cache_initializer_source(raw):
    """Keep eager prefill when native confidence enables varlen decode graphs.

    Extend the existing source-pinned cache initializer transaction, rather
    than replacing its DCP ownership hook or changing a live native file.
    Native confidence unconditionally requests piecewise prefill; this recipe
    has always used eager prefill and owned full decode graphs instead.
    The adaptive manager and its varlen decode descriptors remain enabled.
    """
    if hashlib.sha256(raw).hexdigest() != CACHE_SHA:
        raise ValueError('Changed V2 cache initializer adapter')
    before = 'compiled_initialize = _compile(original_initialize, [\n'
    text = raw.decode()
    if text.count(before) != 1:
        raise ValueError('Changed initializer patch transaction')
    text = text.replace(before, before+'        '+repr(GRAPH_MODE)+',\n')
    if text.count(SLOT_VIEW[0]) != 1:raise ValueError('Changed slot-buffer view anchor')
    text = text.replace(*SLOT_VIEW)
    compile(text, 'vllm_v2_cache.py', 'exec')
    return text.encode()


def supports_platform(native, cuda, capability):
    return bool(native or (cuda and capability == 121))


def make_patches():
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.mla import indexer
    from vllm.v1.worker.gpu.spec_decode import adaptive_verification as adaptive
    if hashlib.sha256(Path(indexer.__file__).read_bytes()).hexdigest() != INDEXER_SHA:
        raise ValueError('Changed native flattened-query metadata implementation')
    original = indexer._supports_flattened_device_query_lens
    def supported():
        native = original()
        if native:
            return True
        capability = current_platform.get_device_capability()
        return supports_platform(native, current_platform.is_cuda(),
            None if capability is None else capability.to_int())
    if hashlib.sha256(Path(adaptive.__file__).read_bytes()).hexdigest() != ADAPTIVE_SHA:
        raise ValueError('Changed native confidence budget implementation')
    manager=adaptive.AdaptiveVerificationManager
    get_tokens=manager.get_num_tokens
    @functools.wraps(get_tokens)
    def budget(instance,*args,**kwargs):
        result=get_tokens(instance,*args,**kwargs)
        scheduled,_,verified=instance._batch_budget
        counts=getattr(instance,'_ds41_budget_counts',None)
        if counts is None:
            counts=instance._ds41_budget_counts=dict(steps=0,request_rounds=0,
                scheduled_drafts=0,verified_drafts=0,trimmed_steps=0)
        if observe_budget(counts,list(scheduled.values()),verified):
            from vllm.distributed.parallel_state import get_tp_group
            if get_tp_group().rank_in_group==0:
                print(json.dumps(dict(stage='ds41_confidence_budget',**counts)),flush=True)
        return result
    return [(indexer, '_supports_flattened_device_query_lens', supported),
            (manager,'get_num_tokens',budget)]
