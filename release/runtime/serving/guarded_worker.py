"""User-ceiling0.90 guard; serve.py separately caps the profiled KV pool.

Select explicitly with --worker-cls guarded_worker.InitialWorker. This module
is a read-only serving overlay, not a change to the calibrated model/image.
NCCL and other direct CUDA allocations remain outside the PyTorch allocator.
"""
import hashlib
import json
from pathlib import Path

import torch
from vllm.v1.worker import gpu_worker

GIB=1024**3
WORKER_SHA256='6db703d5bf98bfc3de20f9c73f67cd81a8a417ca1686eb6a4a4c035c53d43427'
INITIAL_UTILIZATION=.90
STARTUP_CACHE_ALLOWANCE=2*GIB


def trim_process_heap():
    """Return only this worker's unused glibc heap; never flush OS caches."""
    import ctypes
    import gc
    collected = gc.collect()
    trim = ctypes.CDLL('libc.so.6').malloc_trim
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    return dict(gc_collected=collected, malloc_trim_result=int(trim(0)))


def runtime_sample():
    memory={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        parts=line.split()
        if parts[0] in ('MemTotal:','MemFree:','MemAvailable:'):
            if len(parts)!=3 or parts[2]!='kB':raise ValueError('Unexpected host memory units')
            memory[parts[0][:-1]]=int(parts[1])*1024
    group=Path('/sys/fs/cgroup')
    limits={name:(group/name).read_text().strip() for name in ('memory.max','memory.swap.max','cpu.max')}
    return memory,limits


def validate_initial(worker,memory,limits):
    model=worker.model_config;parallel=worker.parallel_config;cache=worker.cache_config
    if (worker.device_config.device_type!='cuda' or worker.local_rank!=0 or worker.rank not in (0,1)
            or model.hf_config.model_type!='deepseek_v41' or model.quantization!='ds41_exl3'
            or not model.enforce_eager or getattr(model,'enable_sleep_mode',False)
            or cache.gpu_memory_utilization!=INITIAL_UTILIZATION or cache.kv_cache_memory_bytes is not None
            or cache.num_gpu_blocks_override is not None
            or worker.vllm_config.speculative_config is not None
            or worker.vllm_config.use_v2_model_runner
            or getattr(worker,'use_v2_model_runner',True)
            or (parallel.tensor_parallel_size,parallel.decode_context_parallel_size,parallel.nnodes)!=(2,2,2)
            or (parallel.pipeline_parallel_size,parallel.prefill_context_parallel_size,parallel.data_parallel_size)!=(1,1,1)
            or parallel.enable_expert_parallel):
        raise ValueError('Guarded serving requires the V1 runner, eager0.90 EXL3 TP2/DCP2 on two nodes, without KV overrides or speculation')
    if set(memory)!={'MemTotal','MemFree','MemAvailable'} or any(type(v) is not int or v<0 for v in memory.values()):
        raise ValueError('Complete integer host-memory sample required')
    if not 120*GIB<=memory['MemTotal']<=128*GIB or not (memory['MemFree']<=memory['MemTotal'] and memory['MemAvailable']<=memory['MemTotal']):
        raise ValueError('Unexpected Spark host-memory accounting')
    required=int(memory['MemTotal']*INITIAL_UTILIZATION)
    required_available=required+2*GIB
    # Pinned vLLM uses MemAvailable for UMA. Permit at most 2GiB of the
    # initial ceiling to come from reclaimable startup cache, not all cache.
    # Keep the separate 2GiB available-memory reserve and allocator cap intact.
    required_free=required-STARTUP_CACHE_ALLOWANCE
    if memory['MemFree']<required_free or memory['MemAvailable']<required_available:
        raise ValueError(f'Initial serving requires free={required_free}, available={required_available}; observed {memory}')
    quota=limits['cpu.max'].split()
    if (limits['memory.max']!=str(9*GIB) or limits['memory.swap.max']!='0'
            or len(quota)!=2 or quota[0]=='max' or not 0<int(quota[0])<=6*int(quota[1])):
        raise ValueError('Initial serving requires an actual9GiB/no-swap/at-most-sixCPU cgroup')
    return required


class InitialWorker(gpu_worker.Worker):
    def _trim_host_heap(self, stage):
        self._log_load_memory(stage+'_before')
        result = trim_process_heap()
        self._log_load_memory(stage+'_after')
        print(json.dumps(dict(stage=stage,rank=self.rank,**result,
            scope='only_unused_current_worker_cpu_heap',global_cache_flush=False,
            gpu_limits_changed=False)),flush=True)

    def determine_available_memory(self):
        runner = self.model_runner
        original = runner.profile_run
        had_override = 'profile_run' in vars(runner)

        def profile_and_trim(*args, **kwargs):
            result = original(*args, **kwargs)
            # The pinned profile_run synchronizes and releases dummy outputs.
            # Reclaim real unused CPU pages BEFORE the normal memory profiler
            # measures consumption; never override its result or KV budget.
            self._trim_host_heap('ds41_after_profile_cpu_heap_trim')
            return result

        runner.profile_run = profile_and_trim
        try:
            result = super().determine_available_memory()
        finally:
            # Restore the callable even on failure; no CUDA/trim on error.
            if had_override:
                runner.profile_run = original
            else:
                del runner.profile_run
        # Parent profiling succeeded, even if the resulting cache budget is
        # negative. Log every rank; the engine will retain its normal refusal.
        self._log_load_memory('ds41_after_memory_profile')
        print(json.dumps(dict(stage='ds41_profile_budget', rank=self.rank,
            utilization=self.cache_config.gpu_memory_utilization,
            requested_memory_bytes=int(self.requested_memory),
            measured_consumed_bytes=int(self.total_consumed),
            peak_activation_headroom_bytes=int(self.peak_activation_memory),
            cudagraph_memory_estimate_bytes=int(self.cudagraph_memory_estimate),
            available_kv_before_mm_ipc_bytes=int(self.available_kv_cache_memory_bytes),
            returned_kv_budget_bytes=int(result),
            model_memory_usage_bytes=int(self.model_runner.model_memory_usage),
            initial_snapshot=vars(self.init_snapshot)), default=str), flush=True)
        return result

    def _log_load_memory(self, stage):
        memory, limits = runtime_sample()
        cg = Path('/sys/fs/cgroup')
        stats = dict(line.split() for line in (cg/'memory.stat').read_text().splitlines())
        print(json.dumps(dict(stage=stage, rank=self.rank, memory=memory, limits=limits,
            cgroup_current=int((cg/'memory.current').read_text()),
            cgroup_stats={key:int(stats[key]) for key in ('anon','file','shmem','kernel')},
            torch_allocated=torch.cuda.memory_allocated(0),
            torch_reserved=torch.cuda.memory_reserved(0))), flush=True)

    def load_model(self, *, load_dummy_weights=False):
        from streaming_loader import register
        register()
        from spark_topk import register as register_topk
        register_topk()
        from spark_fused_moe import register as register_moe
        register_moe()
        self._log_load_memory('ds41_before_model_load')
        result = super().load_model(load_dummy_weights=load_dummy_weights)
        # Deliberately no finally block: do not query CUDA after an exception.
        self._trim_host_heap('ds41_after_load_cpu_heap_trim')
        self._log_load_memory('ds41_after_model_load')
        return result

    def init_device(self):
        memory,limits=runtime_sample()
        required=validate_initial(self,memory,limits)
        if hashlib.sha256(Path(gpu_worker.__file__).read_bytes()).hexdigest()!=WORKER_SHA256:
            raise ValueError('Unreviewed vLLM worker initialization implementation')
        cuda=torch.cuda
        if cuda.device_count()!=1:raise ValueError('One visible GPU per Spark is required')
        cuda.set_device(0)
        properties=cuda.get_device_properties(0)
        # Use this node's actual CUDA total, never its sibling's total. A small
        # downward driver reservation only lowers the GPU allocator ceiling.
        if ((properties.major,properties.minor)!=(12,1)
                or not 0<=memory['MemTotal']-properties.total_memory<=256*1024**2):
            raise ValueError('Expected one SM12.1 unified-memory Spark GPU')
        if cuda.memory_allocated(0) or cuda.memory_reserved(0):
            raise ValueError('Install the serving allocator ceiling before tensor allocations')
        cuda.set_per_process_memory_fraction(INITIAL_UTILIZATION,0)
        print(json.dumps(dict(stage='ds41_initial_worker_allocator_capped',rank=self.rank,
            allocator_fraction=INITIAL_UTILIZATION,allocator_limit_bytes=int(properties.total_memory*INITIAL_UTILIZATION),
            memory=memory,limits=limits,required_initial_host_free_bytes=required-STARTUP_CACHE_ALLOWANCE,
            required_initial_host_available_bytes=required+2*GIB,
            startup_cache_allowance_bytes=STARTUP_CACHE_ALLOWANCE,
            non_pytorch_cuda_allocations_capped=False,full_serving_validated=False)),flush=True)
        # Parent initializes NCCL, then takes its normal memory snapshot and
        # performs model-runner setup. No extra CUDA calls follow an exception.
        return super().init_device()
