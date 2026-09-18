# SPDX-License-Identifier: AGPL-3.0-only
# Integration with the attributed MiaAI/ExLlama serving stack; original notices retained.
"""Scoped 3-bit draft experts; native draft dense/shared/attention weights stay native.

Allocate packed tensors directly. Filter old routed-expert metadata BEFORE tensor
materialization, then load the separately pinned EXL3 artifact. No runtime routing
or speculative-sampling changes, and no transient native FP4 expert allocation.
"""
import hashlib
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

from .draft_exl3_contract import expected_keys, packed_key

DESCRIPTOR_SHA = '023891557c57ab1aece696a07c953746492c967d95666ef31c824483881aefc1'
PARAMETER_BYTES_PER_RANK = 2562494976
NATIVE_EXPERT = re.compile(r'mtp\.[0-2]\.ffn\.experts\.\d+\.')


def artifact():
    root = Path(os.environ['DS41_DRAFT_EXL3_PATH'])
    if not root.is_absolute() or root.resolve() != root:
        raise ValueError('An explicit unredirected draft artifact is required')
    raw = (root/'draft-exl3.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != DESCRIPTOR_SHA:
        raise ValueError('Draft artifact descriptor changed')
    descriptor = json.loads(raw)
    index_raw = (root/'model.safetensors.index.json').read_bytes()
    if hashlib.sha256(index_raw).hexdigest() != descriptor['index_sha256']:
        raise ValueError('Draft artifact index changed')
    index = json.loads(index_raw)['weight_map']
    expected = set().union(*(expected_keys(l, e) for l in range(3) for e in range(128)))
    if set(index) != expected or set(index.values()) != set(descriptor['shards']):
        raise ValueError('Incomplete or extra draft expert tensors')
    for name, info in descriptor['shards'].items():
        path = root/name
        if path.resolve() != path or path.stat().st_size != info['bytes']:
            raise ValueError('Draft shard changed: '+name)
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != info['sha256']:
                raise ValueError('Draft shard digest mismatch: '+name)
    return root, index


def validate_allocation(current, layer, num_experts, hidden_size, intermediate):
    from .combined_dspark import _draft_scope
    parallel = current.parallel_config
    if (_draft_scope.get() is None or num_experts != 128 or hidden_size != 5120
            or intermediate != 1152 or parallel.tensor_parallel_size != 2
            or parallel.enable_expert_parallel or parallel.enable_eplb
            or hasattr(layer, 'w13_weight') or hasattr(layer, 'w2_weight')):
        raise ValueError('Draft EXL3 allocation requires scoped128-expert TP2 H5120/I1152')


def make_quantizer_patches(scope):
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.models.deepseek_v4_1.quant_config import DeepseekV4FP8Config
    from .vllm_exl3 import DS41EXL3MoEMethod
    from .vllm_dcp import _compile
    # Compile from the immutable on-disk method BEFORE target graph rewrites.
    create = _compile(DS41EXL3MoEMethod.create_weights, [
        ('if not current.model_config.enforce_eager or current.parallel_config.enable_expert_parallel:\n'
         '        raise ValueError("Initial DS41 EXL3 backend requires eager TP without expert parallel")',
         '_ds41_validate_draft_allocation(current, layer, num_experts, hidden_size, intermediate_size_per_partition)'),
    ], {'_ds41_validate_draft_allocation': validate_allocation})

    class DraftEXL3MoEMethod(DS41EXL3MoEMethod):
        create_weights = create

    original = DeepseekV4FP8Config.get_quant_method

    def get_quant_method(quantizer, layer, prefix):
        if scope.get() is not None and isinstance(layer, RoutedExperts):
            if (type(quantizer) is not DeepseekV4FP8Config
                    or not re.search(r'(?:^|\.)layers\.(40|41|42)\.ffn\.experts$', prefix)):
                raise ValueError('Unexpected quantizer or prefix in draft-only EXL3 scope: '+prefix)
            return DraftEXL3MoEMethod(layer.moe_config, 3)
        return original(quantizer, layer, prefix)

    return [(DeepseekV4FP8Config, 'get_quant_method', get_quant_method)]


def load_packed(model, root, index):
    from safetensors import safe_open
    params = dict(model.named_parameters())
    loaded = set()
    for shard_name in sorted(set(index.values())):
        with safe_open(root/shard_name, framework='pt', device='cpu') as shard:
            keys = {name for name in index if index[name] == shard_name}
            if set(shard.keys()) != keys:
                raise ValueError('Draft shard contents disagree with the pinned index')
            for name in sorted(keys):
                key = packed_key(name)
                parameter_name = key['param'].replace('.ffn.experts.', '.ffn.experts.routed_experts.')
                param = params[parameter_name]
                param.weight_loader(param, shard.get_tensor(name), parameter_name,
                    shard_id=key['projection'], expert_id=key['expert'])
                loaded.add(parameter_name)
    return loaded


def make_weight_loader(original, stream, scope):
    def load_weights(model, weights):
        if not stream._registered or stream._mapper.get() is not None or scope.get() is None:
            raise RuntimeError('Draft EXL3 loading requires the native scoped streaming transaction')
        root, index = artifact()

        def map_name(name):
            return None if NATIVE_EXPERT.match(name) else model._remap_dspark_name(name)

        token = stream._mapper.set(SimpleNamespace(_map_name=map_name))
        try:
            loaded = original(model, weights)
            loaded.update(load_packed(model, root, index))
        finally:
            stream._mapper.reset(token)
        # vLLM subsequently calls each quant_method's own post-load validation.
        model._ds41_draft_exl3 = dict(descriptor_sha256=DESCRIPTOR_SHA,
            parameter_bytes_per_rank=PARAMETER_BYTES_PER_RANK,
            old_fp4_experts_materialized=False, routed_experts=384)
        print(json.dumps(dict(stage='ds41_draft_exl3_loaded', **model._ds41_draft_exl3)), flush=True)
        return loaded
    return load_weights


def inventory(draft):
    receipt = getattr(draft, '_ds41_draft_exl3', None)
    if receipt is None or receipt['descriptor_sha256'] != DESCRIPTOR_SHA:
        raise ValueError('Missing packed draft loading receipt')
    total = 0
    borrowed_router_bytes = 0
    packed_names = {group+'_'+suffix for group in ('w13','w2')
        for suffix in ('trellis','suh','svh','mul1')}
    for layer in draft.model.layers:
        experts = layer.ffn.experts.routed_experts
        if (hasattr(experts, 'w13_weight') or hasattr(experts, 'w2_weight')
                or len(experts._ds41_experts) != 128 or experts.quant_method.bits != 3
                or len(experts.quant_method.loaded) != 128*12):
            raise ValueError('Incomplete EXL3 draft bank or retained FP4 expert tensors')
        params = dict(experts.named_parameters())
        if set(params) != packed_names | {'e_score_correction_bias'}:
            raise ValueError('Unexpected draft routed-expert parameter inventory')
        bias = params['e_score_correction_bias']
        if (bias is not layer.ffn.gate.e_score_correction_bias
                or tuple(bias.shape) != (128,) or str(bias.dtype) != 'torch.float32'):
            raise ValueError('Native routing bias must remain shared, not copied or quantized')
        borrowed_router_bytes += bias.numel()*bias.element_size()
        total += sum(params[name].numel()*params[name].element_size() for name in packed_names)
    if total != PARAMETER_BYTES_PER_RANK:
        raise ValueError('Packed draft memory inventory mismatch: '+str(total))
    return dict(receipt, actual_parameter_bytes=total, bits=3, codebook='mul1',
        unchanged_router_bias_alias_bytes=borrowed_router_bytes,
        native_dense_shared_attention_unchanged=True)
