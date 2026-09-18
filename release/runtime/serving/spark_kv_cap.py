"""Reduce native profiled KV budgets; never replace or inflate admission.

The cap is the qualified native DCP2 pool for one 1,048,576-token request,
including the null block. It does not assert that physical serving fits.
Register from serve.py in both the initial interpreter and spawned children.
"""
import functools
import hashlib
import json
from pathlib import Path
import threading

CAP_BYTES = 1610612736
POOL_BLOCKS = 4382
POOL_STRIDE = 230400
NATIVE_ROOT = Path('/opt/ds41-venv/lib/python3.12/site-packages/vllm')
NATIVE_SHA256 = {
    'v1/core/kv_cache_utils.py': '0c4b312712598930d2dbdcf917c511de73fe531dd009f5b200e3c945b009bafe',
    'v1/engine/core.py': 'ef709a037077f48db65d381305a0427c25749bd5dc7997f3dec5b5539f1f0dc4',
}
_lock = threading.Lock()
_original = None
_wrapper = None
_owners = ()


def backing_bytes(config):
    # Native worker.utils.allocate_kv_cache requires a single common size
    # and allocates ONE backing tensor. These descriptors are aliased views.
    sizes = {tensor.size for tensor in config.kv_cache_tensors}
    if len(sizes) != 1:
        raise RuntimeError('Expected one shared native KV backing allocation')
    return sizes.pop()


def capped_budgets(config, specs, available_memory):
    """Return a new budget list, preserving insufficient/negative values."""
    cache, parallel = config.cache_config, config.parallel_config
    model, scheduler = config.model_config, config.scheduler_config
    if (cache.num_gpu_blocks_override is not None
            or cache.kv_cache_memory_bytes is not None):
        raise ValueError('Downward KV cap forbids block-count and KV-byte overrides')
    if (not 0 < cache.gpu_memory_utilization <= .90
            or model.quantization != 'ds41_exl3' or not model.enforce_eager
            or model.original_max_model_len == -1
            or not 0 < model.max_model_len <= 1048576
            or (parallel.tensor_parallel_size, parallel.decode_context_parallel_size,
                parallel.pipeline_parallel_size, parallel.prefill_context_parallel_size,
                parallel.data_parallel_size, parallel.nnodes) != (2, 2, 1, 1, 1, 2)
            or parallel.enable_expert_parallel or config.speculative_config is not None
            or config.use_v2_model_runner
            or scheduler.max_num_seqs != 1 or scheduler.max_num_batched_tokens != 1056):
        raise ValueError('Downward KV cap requires bounded eager V1 EXL3 TP2/DCP2 serving')
    if (len(specs) != 2 or len(available_memory) != 2
            or any(type(value) is not int for value in available_memory)):
        raise ValueError('Exactly two integer profiled worker budgets are required')
    return [min(value, CAP_BYTES) for value in available_memory]


def register():
    global _original, _wrapper, _owners
    with _lock:
        if _original is not None:
            if any(owner.get_kv_cache_configs is not _wrapper for owner in _owners):
                raise RuntimeError('Registered KV cap binding changed; refusing partial repair')
            return
        from vllm.v1.core import kv_cache_utils as utils
        from vllm.v1.engine import core
        for relative, owner in (('v1/core/kv_cache_utils.py', utils),
                                ('v1/engine/core.py', core)):
            path = Path(owner.__file__)
            if (path.resolve() != NATIVE_ROOT / relative
                    or hashlib.sha256(path.read_bytes()).hexdigest() != NATIVE_SHA256[relative]):
                raise RuntimeError(f'Unreviewed native KV cap target: {path}')
        original = utils.get_kv_cache_configs
        if (core.get_kv_cache_configs is not original
                or original.__module__ != utils.__name__
                or Path(original.__code__.co_filename).resolve() != NATIVE_ROOT/'v1/core/kv_cache_utils.py'):
            raise RuntimeError('Native KV planner binding already changed')

        @functools.wraps(original)
        def bounded(vllm_config, kv_cache_specs, available_memory):
            budgets = capped_budgets(vllm_config, kv_cache_specs, available_memory)
            print(json.dumps(dict(stage='ds41_additive_kv_pool_cap',
                profiled_budget_bytes=list(available_memory), effective_budget_bytes=budgets,
                ordinary_cap_bytes=__import__('ds41.combined_config',fromlist=['KV_CAP_BYTES']).KV_CAP_BYTES, external_display_bytes=1879048192, native_ordinary_admission_preserved=True)), flush=True)
            # No block override, modified profiler, exception swallowing, or
            # auto-fit. The unchanged native planner performs admission first.
            result = original(vllm_config, kv_cache_specs, budgets)
            allocated = [backing_bytes(item) for item in result]
            if len(result) != 2 or any(size > budget for size, budget in zip(allocated, budgets)):
                raise RuntimeError('Native KV descriptors exceeded the downward budget')
            print(json.dumps(dict(stage='ds41_additive_kv_pool_planned',
                descriptor_bytes=allocated, num_blocks=[item.num_blocks for item in result],
                max_model_len=vllm_config.model_config.max_model_len)), flush=True)
            return result

        # Both bindings are imported already; no spawned process relies on a
        # parent mutation because serve.py registers again under __mp_main__.
        utils.get_kv_cache_configs = bounded
        core.get_kv_cache_configs = bounded
        _original, _wrapper, _owners = original, bounded, (utils, core)
