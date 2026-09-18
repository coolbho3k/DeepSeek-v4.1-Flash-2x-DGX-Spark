# SPDX-License-Identifier: AGPL-3.0-only
"""Explicit, process-local0.92 startup admission override.

Does not alter snapshots, allocator caps, actual profiling, KV budgets,
continuous host supervision, buffer bounds, or CUDA exception handling.
"""
import hashlib
import inspect
import json
import math
import os

ENV='DS41_ALLOW_STARTUP_MEMORY_SHORTFALL'

def enabled(utilization):
    mode=os.environ.get(ENV,'0')
    if mode not in ('0','1'):raise ValueError(ENV+' must be0 or1')
    if mode=='1' and utilization!=.92:
        raise ValueError('The explicit startup shortfall override is scoped to0.92 only')
    return mode=='1'

def host_admitted(worker,memory,required_free,required_available):
    if memory['MemFree']>=required_free and memory['MemAvailable']>=required_available:return True
    if not enabled(worker.cache_config.gpu_memory_utilization):return False
    print(json.dumps(dict(stage='ds41_explicit_startup_headroom_override',check='host_startup_reserve',
        rank=worker.rank,actual=memory,required_free=required_free,required_available=required_available,
        allocator_fraction=.92,native_memory_profile_unchanged=True,continuous_host_guard_unchanged=True)),flush=True)
    return True

def make_request_memory(original):
    def request_memory(snapshot,cache_config):
        selected=enabled(cache_config.gpu_memory_utilization)
        if not selected:return original(snapshot,cache_config)
        total,free=snapshot.total_memory,snapshot.free_memory
        if (type(total) is not int or type(free) is not int or not 0<=free<=total
                or total<=0 or cache_config.kv_cache_memory_bytes is not None):
            raise ValueError('Override requires a real unmodified memory snapshot and native KV profiling')
        requested=math.ceil(total*cache_config.gpu_memory_utilization)
        if free>=requested:return original(snapshot,cache_config)
        print(json.dumps(dict(stage='ds41_explicit_startup_headroom_override',check='native_free_memory_admission',
            actual_free_bytes=free,total_bytes=total,requested_bytes=requested,shortfall_bytes=requested-free,
            allocator_fraction=.92,snapshot_modified=False,profile_budget_forced=False)),flush=True)
        return requested
    request_memory._ds41_original=original
    return request_memory

def install_native(worker_module,expected_source_sha256):
    original=worker_module.request_memory
    if hasattr(original,'_ds41_original'):raise RuntimeError('Startup override already installed')
    source=inspect.getsource(original).strip().encode()
    if hashlib.sha256(source).hexdigest()!=expected_source_sha256:
        raise RuntimeError('Native memory-admission function changed; refuse override')
    worker_module.request_memory=make_request_memory(original)
