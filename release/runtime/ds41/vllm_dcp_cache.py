"""Per-group DCP ownership for the pinned V4.1 cache stack.

Global main/index KV is sharded; SWA and compressor rings are replicated.
No hooks are installed on import. The serving registration must coordinate
these hooks with metadata/attention; the reversible probe context remains.
"""
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
import hashlib
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from .vllm_dcp import _compile
from .vllm_prefill_workspace import row_bound

UPSTREAM = {
    'vllm.v1.kv_cache_interface': 'f1a63db8d1404d8e8998e70f3a373e8e93d159e62be6dc108d0840a9ce0accf3',
    'vllm.v1.core.kv_cache_utils': '0c4b312712598930d2dbdcf917c511de73fe531dd009f5b200e3c945b009bafe',
    'vllm.v1.core.kv_cache_coordinator': 'd7bbe8cd7d9fc1aa639fba5c5a4445fa23601394ff4fa9f2aeaac9606e78352c',
    'vllm.v1.core.single_type_kv_cache_manager': '128b98a0511f67d32f44767aa1658776a8246374b9eb8461d397e683ff3d984d',
    'vllm.v1.worker.block_table': '15ba042b9e55a7ff79cde1fd1be598ec4d038a569556324ed83e2dbd8c66dac4',
    'vllm.v1.worker.gpu_model_runner': '9bc4c51b4c101358b231caa308293986503a0f770b957de29325e10d7890cc32',
}
_table_owners = ContextVar('ds41_probe_table_owners', default=None)
_table_owner = ContextVar('ds41_probe_table_owner', default=None)


def group_kind(spec):
    from vllm.v1.kv_cache_interface import (
        CircularBufferSpec, MLAAttentionSpec, SlidingWindowMLASpec, iter_layer_specs)
    kinds = set()
    for inner in iter_layer_specs(spec):
        if type(inner) is SlidingWindowMLASpec and inner.sliding_window == 128 and inner.block_size == 32:
            kinds.add('swa')
        elif type(inner) is MLAAttentionSpec and inner.block_size == 128 and inner.tokens_per_state in (1, 2):
            kinds.add('global')
        elif type(inner) is CircularBufferSpec and inner.block_size == 8:
            kinds.add('ring')
        else:
            raise ValueError(f'Unreviewed DCP cache spec: {inner}')
    if len(kinds) != 1:
        raise ValueError('Mixed ownership inside a cache group')
    return kinds.pop()


def validate_groups(cache):
    expected = {
        'swa': {f'language_model.model.layers.{i}.attn.swa_cache' for i in range(40)},
        'global': {f'language_model.model.layers.{i}.attn{tail}'
                   for i in (2, 8, 14, 20) for tail in ('', '.indexer.k_cache')},
        'ring': {f'language_model.model.layers.{i}.attn.compressor.state_cache' for i in (2, 8, 14)},
    }
    actual = {kind: set() for kind in expected}
    for group in cache.kv_cache_groups:
        kind = group_kind(group.kv_cache_spec)
        for name in group.layer_names:
            if name in actual[kind]:
                raise ValueError(f'Duplicate cache layer: {name}')
            actual[kind].add(name)
    if actual != expected:
        raise ValueError('DCP cache ownership requires all51 pinned V4.1 caches')


@contextmanager
def table_ownership(cache):
    """Explicit ownership while the real worker constructs all group tables."""
    validate_groups(cache)
    owners = [2 if group_kind(group.kv_cache_spec) == 'global' else 1
              for group in cache.kv_cache_groups]
    token = _table_owners.set(iter(owners))
    try:
        yield owners
    finally:
        _table_owners.reset(token)


def make_cache_patches(config=None):
    """Compile all hooks without mutation; startup may precede config creation."""
    if config is not None:
        row_bound(config)
        if config.parallel_config.decode_context_parallel_size != 2:
            raise ValueError('Cache ownership requires DCP2')
    modules = {name: importlib.import_module(name) for name in UPSTREAM}
    for name, module in modules.items():
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != UPSTREAM[name]:
            raise RuntimeError(f'Unreviewed cache runtime: {name}')
    interface = modules['vllm.v1.kv_cache_interface']
    utils = modules['vllm.v1.core.kv_cache_utils']
    coordinator = modules['vllm.v1.core.kv_cache_coordinator']
    tables = modules['vllm.v1.worker.block_table']
    runner = modules['vllm.v1.worker.gpu_model_runner']
    original_memory = interface.SlidingWindowSpec.max_memory_usage_bytes
    original_width = interface.AttentionSpec.max_num_blocks_per_req
    original_resolve = utils.resolve_dcp_kv_block_size
    original_hybrid = coordinator.HybridKVCacheCoordinator.__init__
    original_table = tables.BlockTable.__init__
    original_runner = runner.GPUModelRunner.may_reinitialize_input_batch

    def is_target(current):
        target = (getattr(current.model_config.hf_config, 'model_type', None) == 'deepseek_v41'
                  and current.parallel_config.decode_context_parallel_size == 2)
        if target:
            row_bound(current)
        return target

    def memory(spec, current):
        if not is_target(current):
            return original_memory(spec, current)
        if group_kind(spec) != 'swa':
            raise ValueError('Only native V4.1 SWA is replicated by this adapter')
        local = NS(model_config=current.model_config,
                   parallel_config=NS(decode_context_parallel_size=1),
                   max_in_flight_tokens=current.max_in_flight_tokens)
        return original_memory(spec, local)

    def width(spec, current, max_len):
        if is_target(current) and isinstance(spec, interface.SlidingWindowSpec):
            assert group_kind(spec) == 'swa'
            return (max_len + spec.block_size - 1) // spec.block_size
        return original_width(spec, current, max_len)

    def block_span(spec, world):
        if world == 1:
            return original_resolve(spec, world)
        if world != 2:
            raise ValueError('Only two DCP ranks are reviewed')
        return spec.block_size * (2 if group_kind(spec) == 'global' else 1)

    compiled_hybrid = _compile(original_hybrid, [
        ('super().__init__(', 'KVCacheCoordinator.__init__(self,'),
        ('isinstance(g.kv_cache_spec, (FullAttentionSpec, MambaSpec))',
         '_ds41_supported_group(g.kv_cache_spec)'),
    ], {'_ds41_supported_group': lambda spec: group_kind(spec) in ('global', 'swa', 'ring')})
    signature = inspect.signature(original_hybrid)
    def hybrid(self, *args, **kwargs):
        arguments = signature.bind(self, *args, **kwargs).arguments
        if arguments['dcp_world_size'] != 2:
            return original_hybrid(self, *args, **kwargs)
        validate_groups(arguments['kv_cache_config'])
        return compiled_hybrid(self, *args, **kwargs)

    def table_group():
        if _table_owner.get() == 1:
            return NS(world_size=1, rank_in_group=0)
        return tables.get_dcp_group()
    compiled_table = _compile(original_table, [], {'get_dcp_group': table_group})
    def table(self, *args, **kwargs):
        owners = _table_owners.get()
        if owners is None:
            return original_table(self, *args, **kwargs)
        owner = next(owners)
        token = _table_owner.set(owner)
        try:
            compiled_table(self, *args, **kwargs)
            assert self.dcp_world_size == owner
        finally:
            _table_owner.reset(token)

    compiled_runner = _compile(original_runner, [
        ('if kv_cache_spec_kind == KVCacheSpecKind.MAMBA:',
         "if kv_cache_spec_kind == KVCacheSpecKind.MAMBA or _ds41_group_kind(kv_cache_spec) == 'ring':"),
    ], {'_ds41_group_kind': group_kind})
    def initialize_tables(self, cache, kernel_block_sizes):
        if not is_target(self.vllm_config):
            return original_runner(self, cache, kernel_block_sizes)
        with table_ownership(cache) as owners:
            compiled_runner(self, cache, kernel_block_sizes)
        actual = self.input_batch.block_table.block_tables
        assert len(actual) == len(owners)
        for item, owner, group in zip(actual, owners, cache.kv_cache_groups):
            assert item.dcp_world_size == owner
            if group_kind(group.kv_cache_spec) == 'ring':
                assert item.slot_mapping_mode == tables.SlotMappingMode.NONE

    return [
        (interface.SlidingWindowSpec, 'max_memory_usage_bytes', memory),
        (interface.AttentionSpec, 'max_num_blocks_per_req', width),
        (utils, 'resolve_dcp_kv_block_size', block_span),
        (coordinator.HybridKVCacheCoordinator, '__init__', hybrid),
        (tables.BlockTable, '__init__', table),
        (runner.GPUModelRunner, 'may_reinitialize_input_batch', initialize_tables),
    ]


@contextmanager
def patched_cache_runtime_for_probe(config):
    hooks = make_cache_patches(config)
    with ExitStack() as stack:
        for owner, name, replacement in hooks:
            stack.enter_context(patch.object(owner, name, replacement))
        yield hooks
