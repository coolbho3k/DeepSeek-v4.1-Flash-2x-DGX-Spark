# SPDX-License-Identifier: AGPL-3.0-only
"""Explicit supported config for the assembled V2/graph/prefill candidate.

This replaces legacy eager-only selection, not the memory, cache or image
contracts. Draft execution requires explicit DSpark selection and its scoped
integration; native memory profiling remains authoritative about actual fit.
"""
import math
import os

from .launch_profile import from_environment
PROFILE = from_environment()
MAX_TOKENS = PROFILE['max_num_batched_tokens']
IO_THREADS = 96
INITIAL_UTILIZATION = PROFILE['gpu_memory_utilization']
MAX_UTILIZATION = 0.925
KV_CAP_BYTES = PROFILE['kv_cap_mib'] * 2**20
GRAPH_SIZES = (1, 2, 3, 4, 6, 8, 9, 12, 15, 16, 18, 20, 24)
DCP_COLLECTIVE_TOKENS = 512
# MiaAI-Lab start.sh, retained under vendor/miaai-serving-stack-agpl.
# Communication buffer sizing only; never disable NCCL argument/error checks.
NCCL_MEMORY_ENV = dict(NCCL_BUFFSIZE='1048576', NCCL_LL128_BUFFSIZE='262144',
                      NCCL_PROTO='^LL128', NCCL_MAX_NCHANNELS='8')


def configure_transport():
    for key, value in NCCL_MEMORY_ENV.items():
        if os.environ.get(key, value) != value:
            raise ValueError('Combined communication setting changed: ' + key)
    os.environ.update(NCCL_MEMORY_ENV)


def collective_chunk_size():
    if os.environ.get('DS41_FP4_DCP_COLLECTIVE_CHUNK') != str(DCP_COLLECTIVE_TOKENS):
        raise ValueError('Combined attention requires512-token collective batches')
    return DCP_COLLECTIVE_TOKENS


def validate_config(config):
    model, scheduler, parallel, cache = (
        config.model_config, config.scheduler_config, config.parallel_config, config.cache_config)
    if os.environ.get('VLLM_SPARSE_INDEXER_MAX_LOGITS_MB') != '128':
        raise ValueError('C6 long contexts require bounded128MiB native indexer logits')
    hf = model.hf_config
    if (hf.model_type != 'deepseek_v41' or model.quantization != 'ds41_exl3'
            or model.enforce_eager or not config.use_v2_model_runner
            or getattr(model, 'enable_sleep_mode', False)
            or (parallel.tensor_parallel_size, parallel.decode_context_parallel_size,
                parallel.prefill_context_parallel_size, parallel.pipeline_parallel_size,
                parallel.data_parallel_size, parallel.nnodes) != (2, 2, 1, 1, 1, 2)
            or parallel.enable_expert_parallel or getattr(parallel, 'enable_eplb', False)
            or parallel.cp_kv_cache_interleave_size != 1
            or scheduler.max_num_seqs != PROFILE['max_num_seqs'] or not scheduler.enable_chunked_prefill
            or scheduler.max_num_batched_tokens != MAX_TOKENS):
        raise ValueError('Combined candidate requires V2 graph EXL3 TP2/DCP2, profile-matched request and prefill budgets')
    from .combined_dspark import validate_speculation
    validate_speculation(config)
    length = model.max_model_len
    if (length != PROFILE['max_model_len'] or scheduler.long_prefill_token_threshold != PROFILE['long_prefill_token_threshold']):
        raise ValueError('CLI/config differ from the explicit serving profile')
    if (type(length) is not int or not 1 <= length <= 1048576
            or length > hf.max_position_embeddings
            or list(hf.kv_source_layer_ids) != [2, 8, 14, 20]
            or list(hf.compress_ratios[:40]) != [0, 0] + [2] * 18 + [1] * 20):
        raise ValueError('Unreviewed V4.1 context/compression layout')
    if (config.attention_config.resolve_indexer_kv_dtype('fp8') != 'mxfp4'
            or cache.cache_dtype != 'fp8_ds_mla'
            or any(os.environ.get(key) != '1' for key in (
                'DS41_ENABLE_COMBINED_MIAAI', 'DS41_ENABLE_DCP2',
                'DS41_ENABLE_FP4_MAIN_KV', 'DS41_ENABLE_FP4_INDEXER',
                'DS41_ENABLE_FUSED_SPARSE_ATTENTION', 'DS41_ENABLE_FUSED_SPARSE_SLOTS',
                'DS41_ENABLE_NATIVE_ENGRAM', 'DS41_ENABLE_GROUPED_PREFILL',
                'DS41_ENABLE_DCP_COMMUNICATION'))):
        raise ValueError('Combined candidate requires the coordinated FP4-main/MXFP4/FP8-SWA stack')
    utilization = cache.gpu_memory_utilization
    if (isinstance(utilization, bool) or not math.isfinite(utilization)
            or not 0 < utilization <= MAX_UTILIZATION
            or cache.kv_cache_memory_bytes is not None):
        raise ValueError('Keep the frozen utilization ceiling and native profiled KV admission')
    compilation = config.compilation_config
    mode = getattr(compilation.cudagraph_mode, 'name', str(compilation.cudagraph_mode))
    if (mode not in ('FULL_AND_PIECEWISE', 'FULL', 'FULL_DECODE_ONLY')
            or not compilation.cudagraph_capture_sizes
            or any(type(size) is not int or size not in GRAPH_SIZES
                   for size in compilation.cudagraph_capture_sizes)
            or compilation.max_cudagraph_capture_size > max(GRAPH_SIZES)):
        raise ValueError('Combined native graph descriptors must stay within the reviewed24-token bound')
    from .vllm_vision_inputs import validate_config as validate_vision
    validate_vision(config)
    return True


def row_bound(config):
    validate_config(config)
    # Native split_indexer_prefill_chunks packs/splits requests to this
    # shared bound. Never multiply by concurrency or divide again by DCP.
    # Each individual request still fits before compression.
    return (config.model_config.max_model_len + 127) // 128 * 128


def validate_worker(worker):
    validate_config(worker.vllm_config)
    if (worker.device_config.device_type != 'cuda' or worker.local_rank != 0
            or worker.rank not in (0, 1) or not worker.use_v2_model_runner
            or worker.cache_config.gpu_memory_utilization != INITIAL_UTILIZATION
            or worker.cache_config.num_gpu_blocks_override is not None):
        raise ValueError('Combined worker requires one GPU per node, V2 and the frozen launch profile')


def capped_budgets(config, specs, available_memory):
    validate_config(config)
    if config.cache_config.num_gpu_blocks_override is not None:
        raise ValueError('No block-count override in combined KV admission')
    if (len(specs) != 2 or len(available_memory) != 2
            or any(type(value) is not int for value in available_memory)):
        raise ValueError('Exactly two native integer profiled budgets are required')
    from .display_kv import credited_budgets
    return credited_budgets(available_memory, KV_CAP_BYTES)
