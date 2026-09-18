"""Bound the pinned V4.1 indexer workspace without changing cached history.

Adapted from the sibling GLM recipe's request-sized indexer workspace fix.
V4.1 applies compression in its indexer constructor, so this helper returns
GLOBAL, UNCOMPRESSED rows. Retaining a whole request is conservative under
DCP2; do not divide the helper by DCP or compression a second time.
Nothing is installed on import. Registration must precede model construction
and any DCP adapter that snapshots the metadata builder's globals.
"""
import hashlib
import importlib
import math
import os
from pathlib import Path
import threading

UPSTREAM = {
    'vllm.v1.attention.backends.mla.indexer':
        '392d93da110ec942db3ce17895dc358bf92501edb01a8b9c19b6bdfab917bb7f',
    'vllm.models.deepseek_v4_1.attention':
        'ef13a8503b54172a63cca6932e2ee5a6d5d6ced81445949067f01b3f61ab6e5e',
    'vllm.model_executor.layers.sparse_attn_indexer':
        '5b094c4280ea615eb79db26734dd8633978bed1cc1189cf278e6a229894900a4',
}
_original = None
_lock = threading.Lock()
_FP4_INDEXER_MODE = os.environ.get('DS41_ENABLE_FP4_INDEXER', '0')
if _FP4_INDEXER_MODE not in ('0', '1'):
    raise ValueError('DS41_ENABLE_FP4_INDEXER must be exactly0 or1')


def row_bound(config):
    if os.environ.get('DS41_ENABLE_FP4_INDEXER', '0') != _FP4_INDEXER_MODE:
        raise ValueError('Indexer workspace format cannot change after import')
    model, scheduler, parallel = (
        config.model_config, config.scheduler_config, config.parallel_config)
    hf = model.hf_config
    if (hf.model_type != 'deepseek_v41' or not model.enforce_eager
            or scheduler.max_num_seqs != 1
            or not scheduler.enable_chunked_prefill
            or config.speculative_config is not None
            or parallel.tensor_parallel_size != 2
            or parallel.decode_context_parallel_size not in (1, 2)
            or parallel.prefill_context_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.enable_expert_parallel
            or parallel.cp_kv_cache_interleave_size != 1):
        raise ValueError('Bounded V4.1 workspace requires eager TP2, DCP1/2, '
                         'one request, chunked prefill, no PCP/PP/EP/speculation')
    length = model.max_model_len
    if (type(length) is not int or not 1 <= length <= 1048576
            or length > hf.max_position_embeddings
            or list(hf.kv_source_layer_ids) != [2, 8, 14, 20]
            or list(hf.compress_ratios[:40]) != [0, 0] + [2] * 18 + [1] * 20):
        raise ValueError('Unreviewed V4.1 context or compression layout')
    expected = 'mxfp4' if _FP4_INDEXER_MODE == '1' else 'fp8'
    if config.attention_config.resolve_indexer_kv_dtype('fp8') != expected:
        raise ValueError('Bounded indexer workspace requires matching cache format: '+expected)
    if _FP4_INDEXER_MODE == '1' and (
            os.environ.get('DS41_ENABLE_FP4_MAIN_KV') != '1'
            or os.environ.get('DS41_ENABLE_DCP2') != '1'
            or parallel.decode_context_parallel_size != 2):
        raise ValueError('MXFP4 workspace requires coordinated FP4-main/DCP2 mode')
    utilization = config.cache_config.gpu_memory_utilization
    if (isinstance(utilization, bool) or not math.isfinite(utilization)
            or not 0 < utilization <= .90
            or config.cache_config.kv_cache_memory_bytes is not None):
        raise ValueError('Keep utilization at or below0.90; no KV-byte override')
    from .vllm_vision_inputs import validate_config as validate_vision
    validate_vision(config)
    # One scheduled request has at most max_model_len keys before compression.
    # Query subchunks reuse them; async steps do not concatenate request keys.
    # Keep a whole global sequence plus alignment, even on a sharded rank.
    # Native _gather_workspace_shapes selects128+4 bytes/row for FP8 versus
    # 64+4 for MXFP4. This hook only bounds rows; it never reinterprets bytes.
    return (length + 127) // 128 * 128


def get_max_prefill_buffer_size(config):
    if config.model_config.hf_config.model_type != 'deepseek_v41':
        if _original is None:
            raise RuntimeError('Workspace hook has not been registered')
        return _original(config)
    return row_bound(config)


def register():
    global _original
    with _lock:
        modules = {name: importlib.import_module(name) for name in UPSTREAM}
        for name, module in modules.items():
            if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != UPSTREAM[name]:
                raise RuntimeError(f'Unreviewed workspace consumer: {name}')
        index = modules['vllm.v1.attention.backends.mla.indexer']
        attention = modules['vllm.models.deepseek_v4_1.attention']
        builder_globals = index.DeepseekV32IndexerMetadataBuilder.__init__.__globals__
        if _original is not None:
            if (index.get_max_prefill_buffer_size is not get_max_prefill_buffer_size
                    or attention.get_max_prefill_buffer_size is not get_max_prefill_buffer_size
                    or builder_globals.get('get_max_prefill_buffer_size') is not get_max_prefill_buffer_size):
                raise RuntimeError('Registered workspace consumer was changed')
            return
        original = index.get_max_prefill_buffer_size
        if (original.__module__ != index.__name__
                or original.__name__ != 'get_max_prefill_buffer_size'
                or attention.get_max_prefill_buffer_size is not original
                or builder_globals is not index.__dict__):
            raise RuntimeError('Register workspace before replacing indexer metadata globals')
        _original = original
        index.get_max_prefill_buffer_size = get_max_prefill_buffer_size
        attention.get_max_prefill_buffer_size = get_max_prefill_buffer_size
