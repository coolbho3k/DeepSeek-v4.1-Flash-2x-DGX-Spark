# SPDX-License-Identifier: AGPL-3.0-only
"""V2 selection with the existing UMA startup, allocator and profile guards."""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import textwrap

import guarded_worker as baseline
from ds41 import combined_config as config
from ds41.vllm_dcp import _compile

BASELINE_SHA256 = 'c8798deab9b5a47a4e651ea88844cf0d70fac107e0f7f06f81bb4b0985dd8c48'


def prepare_validator():
    if hashlib.sha256(Path(baseline.__file__).read_bytes()).hexdigest() != BASELINE_SHA256:
        raise RuntimeError('Unreviewed original host-memory/allocator guard')
    source = textwrap.dedent(inspect.getsource(baseline.validate_initial))
    definition = ast.parse(source).body[0]
    checks = [node for node in definition.body if isinstance(node, ast.If)]
    first = checks[0]
    if ('Guarded serving requires the V1 runner' not in ast.get_source_segment(source, first)
            or len(checks) != 5):
        raise RuntimeError('Original configuration/memory guard structure changed')
    # Replace ONLY the old V1/eager selection with the explicit V2 validator.
    # Preserve four host-RAM/cgroup checks; explicit256MiB additional startup reserve; native admission retained; this candidate uses the approved512MiB steady guard.
    return _compile(baseline.validate_initial, [
        (ast.get_source_segment(source, first), '_ds41_validate_worker(worker)'),
        ("memory['MemFree']<required_free or memory['MemAvailable']<required_available",
         'not _ds41_host_admitted(worker, memory, required_free, required_available)'),
        ('required_available=required+2*GIB', 'required_available=required+256*2**20'),
    ], {'_ds41_validate_worker': config.validate_worker, '_ds41_host_admitted':host_admitted,
        'INITIAL_UTILIZATION': config.INITIAL_UTILIZATION, 'STARTUP_CACHE_ALLOWANCE': 6*2**30})


from ds41.startup_headroom_override import host_admitted, install_native
validate_initial = prepare_validator()
install_native(baseline.gpu_worker, '56a605d8354010bfffb126e1577dee2514a579a5cdbf0466beeb00ab0af92d59')


def trim_before_initial(worker):
    """Return this worker's unused import heap before taking its RAM sample.

    This is the same CPU-only cleanup already used after load/profile. It
    changes no threshold or CUDA state; the original checks sample the real
    host state again immediately afterward and can still reject startup.
    """
    before, _ = baseline.runtime_sample()
    result = baseline.trim_process_heap()
    after, _ = baseline.runtime_sample()
    print(json.dumps(dict(stage='ds41_before_initial_guard_cpu_heap_trim', rank=worker.rank,
        before=before, after=after, **result, only_current_worker_unused_cpu_heap=True,
        global_cache_flush=False, gpu_limits_changed=False)), flush=True)


_init_device = _compile(baseline.InitialWorker.init_device, [
    ('memory,limits=runtime_sample()',
     '_ds41_trim_before_initial(self)\n    memory,limits=runtime_sample()'),
    ('return super().init_device()', 'return gpu_worker.Worker.init_device(self)'),
    ('required_initial_host_available_bytes=required+2*GIB',
     'required_initial_host_available_bytes=required+256*2**20'),
], {'validate_initial': validate_initial, 'INITIAL_UTILIZATION': config.INITIAL_UTILIZATION,
    '_ds41_trim_before_initial': trim_before_initial, 'STARTUP_CACHE_ALLOWANCE': 6*2**30})


def release_loaded_weight_pages(worker):
    """MiaAI post-load cache advice, restricted to the53 read-only shards.

    Only called after successful target/draft loading. No global cache flush,
    Engram advice, tensor allocation, file write or relaxed memory policy.
    Adapted from vendor/miaai-serving-stack-agpl/overlay/patch_memory_log.py.
    """
    if worker.model_config.model != '/model':
        raise ValueError('Clean weight advice requires the exact serving mount')
    names = [f'model-{i:05d}-of-00051.safetensors' for i in range(1,52)]
    if worker.vllm_config.speculative_config is not None:
        names += [f'draft/model-{i:05d}-of-00002.safetensors' for i in (1,2)]
    names += [f'/draft-exl3/draft-{i:05d}-of-00003.safetensors' for i in (1,2,3)]
    def stamp(s):
        return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
    files = []
    for name in names:
        path = Path('/model')/name
        identity = path.lstat()
        if (path.resolve() != path or not stat.S_ISREG(identity.st_mode)
                or not 0 < identity.st_size < 8*2**30
                or not os.statvfs(path).f_flag & os.ST_RDONLY):
            raise ValueError('Expected original read-only regular weight shard')
        files.append((path,stamp(identity)))
    before,_ = baseline.runtime_sample()
    for path,identity in files:
        fd = os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            if stamp(os.fstat(fd)) != identity:
                raise ValueError('Weight shard changed before cache advice')
            os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
            if stamp(os.fstat(fd)) != identity or stamp(path.stat()) != identity:
                raise ValueError('Weight shard changed during cache advice')
        finally:
            os.close(fd)
    after,_ = baseline.runtime_sample()
    print(json.dumps(dict(stage='ds41_loaded_weight_clean_pages_advised',rank=worker.rank,
        files=len(files),memory_before=before,memory_after=after,model_files_modified=False,
        files_deleted=False,engrams_touched=False,global_cache_flush=False,
        cuda_calls=False,limits_changed=False)),flush=True)


def execute_with_prefill_reclaim(worker, scheduler_output, execute):
    """Return unused CPU/CUDA blocks at sampled boundaries, never live state.

    Long prefills may never enter decode for many minutes. Reclaim after16
    successful prefill steps as well as at the original transition to decode.
    Model/KV/graph owners remain referenced; no thresholds or budgets change.
    """
    tokens = scheduler_output.total_num_scheduled_tokens
    steps = getattr(worker, '_ds41_prefill_steps_since_reclaim', 0)
    decode = 1 <= tokens <= 4*config.PROFILE['max_num_seqs']
    periodic = tokens > 32 and steps >= 16
    if (decode or periodic) and getattr(worker, '_ds41_prefill_reclaim_pending', False):
        cuda = baseline.torch.cuda
        if (not worker.use_v2_model_runner
                or worker.model_runner.execute_model_state is not None
                or cuda.is_current_stream_capturing()):
            raise RuntimeError('Prefill reclaim requires a sampled V2 state outside capture')
        cuda.synchronize()
        stage = 'ds41_prefill_interval_reclaim' if periodic else 'ds41_after_prefill_reclaim'
        worker._log_load_memory(stage+'_before')
        baseline.trim_process_heap()
        cuda.empty_cache()
        worker._log_load_memory(stage+'_after')
        worker._ds41_prefill_reclaim_pending = False
        worker._ds41_prefill_steps_since_reclaim = 0
    result = execute(scheduler_output)
    if tokens > 32:
        worker._ds41_prefill_reclaim_pending = True
        worker._ds41_prefill_steps_since_reclaim = getattr(worker, '_ds41_prefill_steps_since_reclaim', 0)+1
    return result


class CombinedWorker(baseline.InitialWorker):
    # Preserve lazy loading, malloc_trim, native profile_run/graph memory
    # accounting, successful-only CUDA observations and every fatal-error path.
    from ds41.dcp_overlap.integration import wrap_init_device as _wrap_overlap_init
    init_device = _wrap_overlap_init(_init_device)

    def initialize_from_config(self, kv_cache_config):
        from ds41.display_kv import real_allocation
        with real_allocation(kv_cache_config):
            return super().initialize_from_config(kv_cache_config)

    def execute_model(self, scheduler_output):
        return execute_with_prefill_reclaim(self, scheduler_output, super().execute_model)

    def load_model(self, *, load_dummy_weights=False):
        result = super().load_model(load_dummy_weights=load_dummy_weights)
        if not load_dummy_weights:
            release_loaded_weight_pages(self)
        return result

    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        # No CUDA observations after a failed warmup/capture. The normal
        # worker's allocator/profile/KV admission remains in control.
        from ds41.resident_storage_audit import audit
        import spark_fused_moe as moe
        inventory=audit(dict(target=self.model_runner.get_model(),dispatcher=moe._dispatcher))
        print(json.dumps(dict(stage='ds41_resident_storage_census',rank=self.rank,
            **inventory),allow_nan=False),flush=True)
        from spark_combined_ready import record_ready_worker
        record_ready_worker(self)
        # Warmup, graph construction and the census create transient CPU
        # objects after the earlier load/profile trims. Release only unused
        # process heap once successful capture has completed, never GPU state.
        self._trim_host_heap('ds41_after_capture_cpu_heap_trim')
        return result
