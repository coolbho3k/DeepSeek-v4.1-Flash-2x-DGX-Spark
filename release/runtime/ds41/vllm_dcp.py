"""Coordinated DCP2 arithmetic for the pinned V4.1 SM12 runtime.

Nothing is installed on import. Probes can enter ``patched_runtime_for_probe``;
the opt-in vllm_dcp_runtime installs these with cache-ownership hooks. Actual
distributed correctness, full-model memory fit and throughput remain unproven.
"""
import ast
from contextlib import contextmanager
import hashlib
import inspect
import os
from pathlib import Path
import textwrap

import torch

from .dcp_attention import bf16_sparse_attention_with_lse, merge_outputs, partition_indices, split_sink
from .dcp_candidates import apply_candidate_mask, select_candidate_blocks
from .dcp_indexer_decode import paged_logits
from .dcp_metadata import compressed_slot_mapping, sparse_global_to_local_slots
from .dcp_key_order import canonicalize_selected_keys_

_FAST_INDEXER_MODE = os.environ.get('DS41_ENABLE_FAST_INDEXER','0')
if _FAST_INDEXER_MODE not in ('0','1'):
    raise ValueError('DS41_ENABLE_FAST_INDEXER must be exactly0 or1')
if _FAST_INDEXER_MODE == '1':
    from .dcp_indexer_decode_fast import paged_logits

_FP4_INDEXER_MODE = os.environ.get('DS41_ENABLE_FP4_INDEXER', '0')
if _FP4_INDEXER_MODE not in ('0', '1'):
    raise ValueError('DS41_ENABLE_FP4_INDEXER must be exactly0 or1')
if _FP4_INDEXER_MODE == '1':
    from .dcp_indexer_mxfp4 import paged_logits


UPSTREAM = {
    'vllm.v1.attention.backends.mla.indexer': '392d93da110ec942db3ce17895dc358bf92501edb01a8b9c19b6bdfab917bb7f',
    'vllm.models.deepseek_v4_1.sparse_mla': '27b17dbeee8916f93849c0ddcd417cdc1650be27b40742fe3ca81ba4ac85cd40',
    'vllm.model_executor.layers.sparse_attn_indexer': '5b094c4280ea615eb79db26734dd8633978bed1cc1189cf278e6a229894900a4',
    'vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse': '19c8c2ebcffacd2e8fc370c005c7a2b619585597dfc1152e8010c7fecfd215c1',
    'vllm.models.deepseek_v4_1.attention': 'ef13a8503b54172a63cca6932e2ee5a6d5d6ced81445949067f01b3f61ab6e5e',
}


def validate_config(config):
    if os.environ.get('DS41_ENABLE_FAST_INDEXER','0') != _FAST_INDEXER_MODE:
        raise ValueError('Fast indexer mode cannot change after DCP import')
    if os.environ.get('DS41_ENABLE_FP4_INDEXER', '0') != _FP4_INDEXER_MODE:
        raise ValueError('FP4 indexer mode cannot change after DCP import')
    if _FP4_INDEXER_MODE == '1' and os.environ.get('DS41_ENABLE_FP4_MAIN_KV') != '1':
        raise ValueError('FP4 indexer requires the coordinated FP4 main-cache variant')
    parallel = config.parallel_config
    if (not config.model_config.enforce_eager
            or config.model_config.hf_config.model_type != 'deepseek_v41'
            or parallel.tensor_parallel_size != 2
            or parallel.decode_context_parallel_size != 2
            or parallel.prefill_context_parallel_size != 1
            or parallel.cp_kv_cache_interleave_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.enable_expert_parallel
            or config.speculative_config is not None):
        raise ValueError('DCP probe requires eager V4.1 TP2/DCP2, interleave1, PP1, no PCP/EP/speculation')
    expected = 'mxfp4' if _FP4_INDEXER_MODE == '1' else 'fp8'
    if config.attention_config.resolve_indexer_kv_dtype('fp8') != expected:
        raise ValueError('DCP indexer mode requires explicit matching cache format: '+expected)


def _sm121_indexer_uses_fp4(config):
    """Opt-in only: native writers plus bounded SM121 unpaged scoring route."""
    from vllm.platforms import current_platform
    validate_config(config)
    capability = current_platform.get_device_capability()
    # Meta constructors supply a capability descriptor without a GPU. Actual
    # serving still rejects a missing capability or anything but SM121.
    if _FP4_INDEXER_MODE != '1' or capability is None or capability.to_int() != 121:
        raise ValueError('The qualified MXFP4 DCP indexer route requires SM121')
    return True


def _compile(function, replacements, extra_globals):
    """Compile reviewed changes against hash-checked source; never edit vendor."""
    original = inspect.unwrap(function)
    source = textwrap.dedent(inspect.getsource(original))
    for before, after in replacements:
        if source.count(before) != 1:
            raise RuntimeError(f'Pinned DCP patch anchor changed in {original.__qualname__}: {before!r}')
        source = source.replace(before, after)
    tree = ast.parse(source)
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    definition.decorator_list = []
    namespace = dict(original.__globals__)
    namespace.update(extra_globals)
    exec(compile(tree, f'<ds41-dcp-probe:{original.__qualname__}>', 'exec'), namespace)
    result = namespace[original.__name__]
    result.__ds41_patch_source__ = source
    return result


def _packed_pages(cache):
    if cache.ndim == 4 and cache.shape[-2] == 1:
        cache = cache.squeeze(-2)
    if cache.ndim != 3 or cache.dtype != torch.uint8 or cache.shape[-1] != 584:
        raise ValueError('DCP probe requires the packed FP8 DSV4 cache, not per-tensor FP8 or BF16 pages')
    return cache


def attention_forward(self, q, output, flashmla_metadata, swa_metadata,
                      self_kv_cache, swa_kv_cache, swa_only, *, group=None):
    """Head exchange + paged sparse attention + FP32 LSE/output merge.

    Keep SWA metadata's native image visibility. Small SWA rings stay replicated;
    compressed history is physically sharded. BF16 output is rounded only once.
    Collectives are chunked to32 tokens to bound extra head/output workspaces.
    """
    if group is None:
        from vllm.distributed import get_dcp_group
        group = get_dcp_group()
    if group.world_size != 2 or group.rank_in_group not in (0, 1):
        raise ValueError('The V4.1 DCP probe supports exactly two ranks')
    world, rank = group.world_size, group.rank_in_group
    if q.dtype != torch.bfloat16 or q.shape[1:] != (32, 512) or output.shape != q.shape or output.dtype != q.dtype:
        raise ValueError('Expected BF16 V4.1 TP2 query/output [tokens,32,512]')
    if self.attn_sink.shape != (32,) or bool(swa_only) != (self.compress_ratio == 0):
        raise ValueError('Invalid sink heads or SWA-only classification')
    swa = _packed_pages(swa_kv_cache)
    compressed = None if swa_only else _packed_pages(self_kv_cache)
    nd, np = swa_metadata.num_decode_tokens, swa_metadata.num_prefill_tokens
    if nd + np != q.shape[0]:
        raise ValueError('Padded/unequal DCP token batches are not supported by the eager probe')
    if not swa_only and (flashmla_metadata is None or self.topk_indices_buffer is None
                         or swa_metadata.token_to_req_indices is None or swa_metadata.is_valid_token is None):
        raise ValueError('Missing compressed attention metadata')
    sinks = self.attn_sink if swa_only else group.all_gather(self.attn_sink.contiguous(), dim=0)
    if not swa_only:
        sinks = split_sink(sinks, world)
    for base, count, indices, lengths in (
        (0, nd, swa_metadata.decode_swa_indices, swa_metadata.decode_swa_lens),
        (nd, np, swa_metadata.prefill_swa_indices, swa_metadata.prefill_swa_lens),
    ):
        if not count:
            continue
        if indices is None or lengths is None:
            raise ValueError('Missing native SWA sparse indices/lengths')
        for start in range(0, count, 32):
            end = min(start + 32, count)
            rows = slice(base + start, base + end)
            local_q = q[rows].contiguous()
            sw_ids, sw_lens = indices[start:end], lengths[start:end]
            extra = {}
            if not swa_only:
                local_q = group.all_gather(local_q, dim=1)
                sw_ids, sw_lens = partition_indices(sw_ids, sw_lens, rank, world, localize=False)
                valid = swa_metadata.is_valid_token[rows].bool()
                candidates = self.topk_indices_buffer[rows]
                candidates = torch.where(valid[:, None], candidates, -1)
                # Fixed-width candidates can contain interior -1 padding: do
                # not count valid IDs and then treat that count as a prefix.
                candidate_width = torch.full((end - start,), candidates.shape[1], device=q.device, dtype=torch.int32)
                req_ids = torch.where(valid, swa_metadata.token_to_req_indices[rows], 0)
                physical, cp_lens = sparse_global_to_local_slots(
                    candidates, candidate_width, req_ids, flashmla_metadata.block_table,
                    flashmla_metadata.block_size // self.compress_ratio, world, rank)
                extra = dict(compressed_cache=compressed, compressed_indices=physical, compressed_lengths=cp_lens)
            partial, lse = bf16_sparse_attention_with_lse(
                local_q, swa, sw_ids, sw_lens, sinks=sinks, scale=self.scale, **extra)
            if not swa_only:
                all_outputs = group.all_gather(partial.unsqueeze(0), dim=0)
                all_lses = group.all_gather(lse.unsqueeze(0), dim=0)
                partial, _ = merge_outputs(all_outputs, all_lses, lse_base=2)
                partial = partial[:, rank * 32:(rank + 1) * 32]
            output[rows].copy_(partial)


def make_probe_patches():
    """Compile ALL coordinated hooks before returning any mutation targets."""
    import importlib
    modules = {name: importlib.import_module(name) for name in UPSTREAM}
    for name, module in modules.items():
        digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if digest != UPSTREAM[name]:
            raise RuntimeError(f'Unreviewed DCP runtime source {name}: {digest}')
    index = modules['vllm.v1.attention.backends.mla.indexer']
    model_attention = modules['vllm.models.deepseek_v4_1.attention']
    mla = modules['vllm.models.deepseek_v4_1.sparse_mla']
    op = modules['vllm.model_executor.layers.sparse_attn_indexer']
    attention = modules['vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse']
    from vllm.distributed import get_dcp_group

    def slots(*args, **kwargs):
        group = get_dcp_group()
        return compressed_slot_mapping(*args, world_size=group.world_size,
                                       rank=group.rank_in_group, **kwargs)

    def select(logits, starts, ends, topk, block_size, out, row_repeat=1):
        return select_candidate_blocks(logits, starts, ends, topk, block_size, out,
                                       get_dcp_group(), row_repeat)

    def mask(logits, starts, ends, candidates, block_size, row_repeat=1):
        group = get_dcp_group()
        return apply_candidate_mask(logits, starts, ends, candidates, block_size,
                                    group.world_size, group.rank_in_group, row_repeat)

    def merge(logits, indices, topk, rank, world, interleave, row_starts=None):
        # Native local top-k may return masked (-inf) positions when fewer
        # than K candidates survive. The native global packer checks only
        # index>=0, so turn those positions into padding BEFORE exchanging
        # them; otherwise they become visible attention tokens again.
        if logits.shape[1] == 0:
            indices.fill_(-1)
        else:
            columns = indices.long()
            if row_starts is not None:
                columns = columns + row_starts.reshape(-1, 1)
            valid = (indices >= 0) & (columns >= 0) & (columns < logits.shape[1])
            scores = logits.gather(1, columns.clamp(0, logits.shape[1] - 1))
            indices.masked_fill_(~(valid & torch.isfinite(scores)), -1)
        result = op._merge_dcp_topk_global(logits, indices, topk, rank, world,
                                          interleave, row_starts=row_starts)
        # Native selection is stable, but atomic output appends can permute
        # the chosen keys. Preserve membership and use one reduction order.
        canonicalize_selected_keys_(indices)
        return result

    def schedule(lengths, block_size, num_sms, indices=None):
        if block_size not in (32, 64, 128):
            raise ValueError('Unsupported physical indexer page size')
        return index.get_paged_mqa_logits_metadata(lengths, min(block_size, 64), num_sms, indices=indices)

    cls = index.DeepseekV32IndexerMetadataBuilder
    guard = '''    if self.dcp_world_size > 1 and self.compress_ratio > 1:
        raise NotImplementedError(
            "DCP is not supported with sparse indexer KV compression "
            f"(compress_ratio={self.compress_ratio})."
        )'''
    selector_globals = {}
    selector_hooks = []
    if _FP4_INDEXER_MODE == '1':
        if model_attention.dsa_indexer_uses_fp4 is not index.dsa_indexer_uses_fp4:
            raise RuntimeError('Native indexer-format selector bindings disagree')
        selector_globals['dsa_indexer_uses_fp4'] = _sm121_indexer_uses_fp4
        selector_hooks = [(index, 'dsa_indexer_uses_fp4', _sm121_indexer_uses_fp4),
            (model_attention, 'dsa_indexer_uses_fp4', _sm121_indexer_uses_fp4)]
    init = _compile(cls.__init__, [
        ('super().__init__(*args, **kwargs)', 'AttentionMetadataBuilder.__init__(self, *args, **kwargs)\n    _ds41_validate(self.vllm_config)'),
        (guard, '    assert self.compress_ratio in (1, 2)'),
    ], {'_ds41_validate': validate_config, **selector_globals})
    localization = '''    if dcp_local_seq_lens is not None:
        seq_lens = self._dcp_localize_decode_seq_lens(
            seq_lens, num_decodes, seq_lens_is_buffer_view
        )
'''
    # The body of build has another indentation level around decode metadata.
    localization = textwrap.indent(localization, '    ')
    after_compression = '''        # DCP ownership follows completed-state compression.
        if global_seq_lens_for_decode is not None and self.compress_ratio > 1:
            global_seq_lens_for_decode = global_seq_lens_for_decode // self.compress_ratio
        if dcp_local_seq_lens is not None:
            seq_lens = self._dcp_localize_decode_seq_lens(
                seq_lens, num_decodes, seq_lens_is_buffer_view or self.compress_ratio > 1
            )

'''
    build = _compile(cls.build, [
        ('    num_reqs = common_attn_metadata.num_reqs', '    _ds41_validate(self.vllm_config)\n    num_reqs = common_attn_metadata.num_reqs'),
        (localization, ''),
        ('        # Non-MTP: deep_gemm paged MQA logits requires 2D context_lens', after_compression + '        # Non-MTP: deep_gemm paged MQA logits requires 2D context_lens'),
    ], {'get_compressed_slot_mapping': slots, '_ds41_validate': validate_config,
        'get_paged_mqa_logits_metadata': schedule})
    main_build = _compile(mla.DeepseekV4SparseMLAMetadataBuilder.build, [
        ('    cm = common_attn_metadata', '    _ds41_validate(self.vllm_config)\n    cm = common_attn_metadata'),
        ('cm.block_table_tensor.clamp_(min=0)', 'cm.block_table_tensor'),
    ], {'get_compressed_slot_mapping': slots, '_ds41_validate': validate_config})
    candidate_guard = '''        assert dcp_world_size == 1, (
            "v4.1 two-level candidate filtering is not supported with DCP."
        )'''
    empty = '''                topk_indices.fill_(-1)
            else:'''
    empty_with_collective = '''                topk_indices.fill_(-1)
                if candidate_blocks is not None and candidate_write:
                    chunk_candidates = candidate_blocks[chunk.token_start:chunk.token_end]
                    _select_candidate_blocks(logits, cu_seqlen_ks, cu_seqlen_ke,
                                             chunk_candidates.shape[1], candidate_block_size, chunk_candidates)
            else:'''
    indexer = _compile(op.sparse_attn_indexer, [
        (candidate_guard, '        assert dcp_world_size == 2 and cp_kv_cache_interleave_size == 1'),
        (empty, empty_with_collective),
    ], {'_select_candidate_blocks': select, '_apply_candidate_mask': mask,
        '_merge_dcp_topk_global': merge,
        'fp8_fp4_paged_mqa_logits': paged_logits})
    forward = _compile(op.SparseAttnIndexer.forward_cuda, [
        ('return torch.ops.vllm.sparse_attn_indexer(', 'return _ds41_indexer('),
    ], {'_ds41_indexer': indexer})
    return selector_hooks + [
        (cls, '__init__', init), (cls, 'build', build),
        (mla.DeepseekV4SparseMLAMetadataBuilder, 'build', main_build),
        (op.SparseAttnIndexer, 'forward_cuda', forward),
        (attention.DeepseekV4FlashInferSM120Attention, '_forward_sparse_impl', attention_forward),
    ]


@contextmanager
def patched_runtime_for_probe():
    patches = make_probe_patches()
    originals = [(owner, name, getattr(owner, name)) for owner, name, _ in patches]
    try:
        for owner, name, replacement in patches:
            setattr(owner, name, replacement)
        yield patches
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)
