# SPDX-License-Identifier: AGPL-3.0-only
# DS41 V2 adaptation for the MiaAI serving integration. The original vLLM
# block-table kernel remains Apache-2.0; it is invoked without modification.
"""V2 per-cache ownership: sharded global KV, replicated SWA and rings.

This only constructs hooks. No imports here install hooks or allocate CUDA
objects. The combined installer must select the matching config validator and
cache-name validator before committing this set atomically with attention.
"""
import hashlib
import importlib
from pathlib import Path

from . import vllm_dcp_cache as common
from .vllm_dcp import _compile

_BASE_MAKE_PATCHES = common.make_cache_patches

UPSTREAM = {
    'vllm.v1.worker.gpu.block_table':
        '61c004315d5af7e7eae4e2a9e6be92ea82c520327690a7f55a73bb9ce95f520a',
    'vllm.v1.worker.gpu.model_runner':
        '33e49f3bd99e642971cea41bbcc73677cb74b863dbf16998d9ebfc94607a64e6',
}


def validate_groups(cache, *, dspark=False):
    if type(dspark) is not bool:
        raise ValueError('DSpark cache ownership must be explicitly selected')
    expected = {
        'swa': {f'language_model.model.layers.{i}.attn.swa_cache' for i in range(40)},
        'global': {f'language_model.model.layers.{i}.attn{tail}'
                   for i in (2, 8, 14, 20) for tail in ('', '.indexer.k_cache')},
        'ring': {f'language_model.model.layers.{i}.attn.compressor.state_cache'
                 for i in (2, 8, 14)},
    }
    if dspark:
        expected['swa'].update(f'model.layers.{i}.attn.swa_cache' for i in (40, 41, 42))
    actual = {kind: set() for kind in expected}
    for group in cache.kv_cache_groups:
        kind = common.group_kind(group.kv_cache_spec)
        for name in group.layer_names:
            if name in actual[kind]:
                raise ValueError(f'Duplicate V2 cache layer: {name}')
            actual[kind].add(name)
    if actual != expected:
        missing = {k: sorted(expected[k] - actual[k]) for k in expected}
        extra = {k: sorted(actual[k] - expected[k]) for k in expected}
        raise ValueError(f'Unreviewed V2 DCP cache set: missing={missing}, extra={extra}')


def make_cache_patches(config=None, *, dspark=False):
    # Reuse exactly the four previously qualified scheduler/spec hooks. Do
    # not install the legacy V1 worker/table hooks in a V2 serving process.
    hooks = _BASE_MAKE_PATCHES(config)[:4]
    modules = {name: importlib.import_module(name) for name in UPSTREAM}
    for name, module in modules.items():
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != UPSTREAM[name]:
            raise RuntimeError(f'Unreviewed V2 cache runtime: {name}')
    tables = modules['vllm.v1.worker.gpu.block_table']
    runner = modules['vllm.v1.worker.gpu.model_runner']
    original_initialize = runner.GPUModelRunner.initialize_kv_cache
    original_compute = tables.BlockTables.compute_slot_mappings

    def construct(cache, **kwargs):
        validate_groups(cache, dspark=dspark)
        if (kwargs['cp_size'] != 2 or kwargs['cp_rank'] not in (0, 1)
                or kwargs['cp_interleave'] != 1):
            raise ValueError('V2 tables require TP2/DCP2 interleave1')
        groups = cache.kv_cache_groups
        owners = tuple(2 if common.group_kind(g.kv_cache_spec) == 'global' else 1
                       for g in groups)
        enabled = [common.group_kind(g.kv_cache_spec) != 'ring' for g in groups]
        if kwargs['slot_mapping_enabled'] != enabled:
            raise ValueError('V2 must suppress slot writes for compressor rings')
        result = tables.BlockTables(**kwargs)
        result._ds41_group_owners = owners
        return result

    compiled_initialize = _compile(original_initialize, [
        ('self.block_tables = BlockTables(',
         'self.block_tables = _ds41_block_tables(kv_cache_config,'),
    ], {'_ds41_block_tables': construct})

    def initialize(self, cache, *args, **kwargs):
        common.row_bound(self.vllm_config)
        validate_groups(cache, dspark=dspark)
        return compiled_initialize(self, cache, *args, **kwargs)

    def compute(self, idx_mapping, query_start_loc, positions, num_tokens_padded, out=None):
        owners = getattr(self, '_ds41_group_owners', None)
        if owners is None:
            return original_compute(self, idx_mapping, query_start_loc, positions,
                                    num_tokens_padded, out=out)
        if (len(owners) != self.num_kv_cache_groups or self.cp_size != 2
                or self.cp_rank not in (0, 1) or self.cp_interleave != 1
                or not 0 <= num_tokens_padded <= self.max_num_batched_tokens):
            raise ValueError('V2 DCP block-table ownership changed after construction')
        slots = self.slot_mappings if out is None else out
        if (slots.ndim != 2 or slots.shape[0] != self.num_kv_cache_groups
                or not num_tokens_padded <= slots.shape[1] <= self.max_num_batched_tokens
                or slots.dtype != self.slot_mappings.dtype
                or slots.device != self.device or slots.stride(1) != 1
                or idx_mapping.ndim != 1 or idx_mapping.shape[0] > self.max_num_reqs
                or query_start_loc.shape != (idx_mapping.shape[0] + 1,)
                or positions.ndim != 1 or positions.shape[0] > num_tokens_padded):
            raise ValueError('Invalid V2 DCP slot mapping buffers')
        # Kernel group0 now sees exactly one group's native pointers/layout.
        # In particular, we do not divide replicated SWA positions by2.
        for group, owner in enumerate(owners):
            span = slice(group, group + 1)
            tables._compute_slot_mappings_kernel[(1, idx_mapping.shape[0] + 1)](
                slots.shape[1], idx_mapping, query_start_loc, positions,
                self.block_table_ptrs[span], self.block_table_strides[span],
                self.block_sizes_tensor[span], self.kernel_block_sizes_tensor[span],
                self.slot_mapping_enabled[span], slots[span], slots.stride(0),
                self.cp_rank if owner == 2 else 0,
                CP_SIZE=owner, CP_INTERLEAVE=1, PAD_ID=tables.PAD_SLOT_ID,
                TRITON_BLOCK_SIZE=1024)
        return slots[:, :num_tokens_padded]

    return hooks + [
        (runner.GPUModelRunner, 'initialize_kv_cache', initialize),
        (tables.BlockTables, 'compute_slot_mappings', compute),
    ]
