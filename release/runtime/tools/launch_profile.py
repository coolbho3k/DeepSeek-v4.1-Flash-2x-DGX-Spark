# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only validation shared by the public launcher and serving workers."""
import math
import os

DEFAULTS = dict(gpu_memory_utilization=.92, max_model_len=1048576,
    max_num_seqs=2, max_num_batched_tokens=3072, long_prefill_token_threshold=2816,
    kv_cap_mib=1536, prefix_cache_retention_interval=4096)
ENV = {key:'DS41_'+key.upper() for key in DEFAULTS}

def validate(values):
    if not isinstance(values, dict) or set(values) != set(DEFAULTS):
        raise ValueError('Unknown or missing serving profile fields')
    for key, default in DEFAULTS.items():
        if type(values[key]) is not type(default):
            raise ValueError('Invalid profile type: '+key)
    if not math.isfinite(values['gpu_memory_utilization']) or not .85 <= values['gpu_memory_utilization'] <= .925:
        raise ValueError('GPU utilization must be0.85..0.925')
    if not 4096 <= values['max_model_len'] <= 1048576:
        raise ValueError('Context must be4096..1048576')
    if not 1 <= values['max_num_seqs'] <= 6:
        raise ValueError('This cooperative profile supports one through six simultaneous sequences')
    if values['max_num_batched_tokens'] not in (2048,3072):
        raise ValueError('Supported prefill budgets:2048 or3072')
    threshold = values['long_prefill_token_threshold']
    if threshold != 0 and not 1056 <= threshold <= values['max_num_batched_tokens']:
        raise ValueError('Long-prefill threshold must preserve whole image spans')
    if values['kv_cap_mib'] != 0:
        raise ValueError('Display-only candidate requires zero ordinary KV; native admission still applies')
    retention = values['prefix_cache_retention_interval']
    if not 0 <= retention <= 1048576 or retention % 256:
        raise ValueError('Prefix retention must be 0 or a multiple of 256 through 1048576')
    return values

def from_environment():
    return validate({k:type(v)(os.environ.get(ENV[k],v)) for k,v in DEFAULTS.items()})

def environment(values):
    return {ENV[k]:str(v) for k,v in validate(values).items()}
