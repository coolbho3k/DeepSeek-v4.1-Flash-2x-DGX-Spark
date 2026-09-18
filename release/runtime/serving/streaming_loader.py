"""Metadata-first ordering for the pinned V4.1 multimodal weight loader.

The upstream wrapper sorts (name, Tensor) pairs for the entire checkpoint.
Keep exactly its mapped-name ordering and one finalization, but sort names
before get_tensor and retain only the current shard/consumer tensor views.
Installed explicitly by InitialWorker, never by importing this module.
"""
from contextvars import ContextVar
import hashlib
from itertools import groupby
import json
from pathlib import Path

_mapper = ContextVar('ds41_streaming_weight_mapper', default=None)
_registered = False
_auto_loader = None
_safe_open = None
_engram_pattern = None

PINS = {
    'vl': 'ab698e56c83a345ea73e41359cab79b5e484ccfc116ed131347d50cdfe896251',
    'utils': 'de363a63d9ef2772a58a05988dd8634add77aabc5d9fac0cde4c9c3fef092e11',
    'embedding': 'c4b79cfd9063a8b79c524e1176a3185d7d58173d2600fb43733e0d09ee8d8e98',
    'plugin': '410f34999eae5fea656654cb86a3e089b24a38cd108ca56e142c14e317e4d8e5',
}


def ordered_nonengram_weights(files, skip_weight=None):
    mapper = _mapper.get()
    if mapper is None:
        raise RuntimeError('Metadata-first iterator requires its matching wrapper')
    plan = []
    seen = set()
    for path in sorted(files):
        with _safe_open(path, framework='pt', device='cpu') as shard:
            for name in shard.keys():
                if _engram_pattern.search(name) or (skip_weight and skip_weight(name)):
                    continue
                mapped = mapper._map_name(name)
                if mapped is None:
                    continue
                if mapped in seen:
                    raise ValueError(f'Duplicate mapped checkpoint key: {mapped}')
                seen.add(mapped)
                plan.append((mapped, name, path))
    plan.sort(key=lambda row: row[0])
    del seen
    print(json.dumps(dict(stage='ds41_weight_metadata_ordered', tensors=len(plan),
                          tensor_payloads_materialized=0)), flush=True)
    completed = 0
    for path, rows in groupby(plan, key=lambda row: row[2]):
        with _safe_open(path, framework='pt', device='cpu') as shard:
            print(json.dumps(dict(stage='ds41_weight_shard_stream', shard=Path(path).name,
                                  completed_tensors=completed)), flush=True)
            for _, name, _ in rows:
                yield name, shard.get_tensor(name)
                completed += 1


def load_weights(self, weights):
    mapper = self.hf_to_vllm_mapper
    def checked_mapped():
        previous = None
        for item in mapper.apply(weights):
            name = item[0]
            if previous is not None and name < previous:
                raise ValueError('Weight stream is not ordered by mapped name')
            previous = name
            yield item
    token = _mapper.set(mapper)
    try:
        loaded = _auto_loader(self).load_weights(checked_mapped())
    finally:
        _mapper.reset(token)
    # Exactly the upstream wrapper's successful-completion semantics.
    self._weights_finalized = True
    return loaded


def register():
    global _registered, _auto_loader, _safe_open, _engram_pattern
    from safetensors import safe_open
    from ds41 import ssd_embedding, vllm_plugin
    from vllm.model_executor.models import utils
    from vllm.models.deepseek_v4_1.nvidia import vl_model
    for name, module in (('vl',vl_model),('utils',utils),
                         ('embedding',ssd_embedding),('plugin',vllm_plugin)):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != PINS[name]:
            raise ValueError(f'Unreviewed streaming loader consumer: {name}')
    if _registered:
        if (vl_model.DeepseekV41ForCausalLM.load_weights is not load_weights
                or vllm_plugin.nonengram_weights is not ordered_nonengram_weights):
            raise RuntimeError('Registered streaming loader was changed')
        return
    if vllm_plugin.nonengram_weights is not ssd_embedding.nonengram_weights:
        raise RuntimeError('Unexpected pre-existing iterator replacement')
    _auto_loader = utils.AutoWeightsLoader
    _safe_open = safe_open
    _engram_pattern = ssd_embedding.ENGRAM_TABLE
    vl_model.DeepseekV41ForCausalLM.load_weights = load_weights
    vllm_plugin.nonengram_weights = ordered_nonengram_weights
    _registered = True
