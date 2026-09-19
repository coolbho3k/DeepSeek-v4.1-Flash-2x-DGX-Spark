# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only actual native startup transaction, without model or CUDA loading."""
import json
import inspect
from pathlib import Path
from types import SimpleNamespace


def check_confidence_graphs():
    """Exercise actual native candidate generation and cache-hook composition.

    No CUDA graph is captured or replayed here: the check is deliberately
    CPU-only and must not be interpreted as GPU/full-model qualification.
    """
    from ds41 import vllm_v2_cache
    from ds41.dspark_experiment.confidence import GRAPH_MODE, SLOT_VIEW, active_query_starts
    from ds41.combined_config import GRAPH_SIZES
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    hooks=vllm_v2_cache.make_cache_patches(dspark=True)
    initializers=[fn for owner,name,fn in hooks
                  if owner is GPUModelRunner and name=='initialize_kv_cache']
    if len(initializers)!=1:raise ValueError('Missing composed cache initializer')
    compiled=inspect.getclosurevars(initializers[0]).nonlocals['compiled_initialize']
    source=compiled.__ds41_patch_source__
    if GRAPH_MODE[0] in source or source.count(GRAPH_MODE[1])!=1 or '_ds41_block_tables(kv_cache_config,' not in source:
        raise ValueError('Confidence graph mode or DCP cache ownership missing')
    import torch
    starts=torch.arange(7,dtype=torch.int32)
    table=SimpleNamespace(max_num_reqs=6,device=starts.device)
    for requests in range(1,7):
        view=active_query_starts(table,torch.arange(requests),starts)
        if view.data_ptr()!=starts.data_ptr() or view.shape!=(requests+1,):
            raise ValueError('Padded query prefix is not a zero-copy exact-length view')
    slots=[fn for owner,name,fn in hooks if name=='compute_slot_mappings']
    if len(slots)!=1 or 'active_query_starts' not in slots[0].__code__.co_names:
        raise ValueError('Actual cache hook does not normalize padded starts')
    from ds41.vllm_dcp import _compile
    prior_view=tuple(part.replace('\n        ','\n    ').strip() for part in SLOT_VIEW)
    prior=_compile(slots[0],[(prior_view[1],prior_view[0])],inspect.getclosurevars(slots[0]).nonlocals)
    if 'active_query_starts' in prior.__code__.co_names:
        raise ValueError('GPU regression did not reconstruct the pre-view guard')
    manager=object.__new__(CudaGraphManager)
    manager.compilation_config=SimpleNamespace(cudagraph_capture_sizes=list(GRAPH_SIZES),
        max_cudagraph_capture_size=max(GRAPH_SIZES))
    manager.vllm_config=SimpleNamespace(speculative_config=SimpleNamespace(
        uses_dynamic_speculative_decoding=lambda:False))
    manager.cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY
    manager.max_num_reqs=6;manager.decode_query_len=6;manager.varlen_decode=True
    manager.lora_capture_cases=[0];manager._capture_descs={};manager._candidates={}
    manager._lora_dispatch_map={};manager._graphs_captured=False
    manager._init_candidates()
    descriptors=manager._capture_descs
    if set(descriptors)!={CUDAGraphMode.FULL}:
        raise ValueError('Confidence must not request piecewise prefill graphs')
    # Simulate graph availability solely to exercise CPU dispatch selection.
    manager._graphs_captured=True
    cases=0
    for requests in range(1,7):
        for tokens in range(requests,6*requests+1):
            selected=manager.dispatch(requests,tokens,None,0,max_query_len=6)
            if (selected.cg_mode!=CUDAGraphMode.FULL
                    or selected.num_tokens<tokens or selected.num_reqs<requests
                    or selected.uniform_token_count is not None or selected.max_query_len!=6):
                raise ValueError('Missing variable-length decode descriptor')
            cases+=1
    if manager.dispatch(1,128,None,0,max_query_len=128).cg_mode!=CUDAGraphMode.NONE:
        raise ValueError('Prefill must retain eager execution')
    return dict(cpu_dispatch_cases=cases,full_target_descriptors=len(descriptors[CUDAGraphMode.FULL]),
        piecewise_descriptors=0,dcp_initializer_preserved=True,padded_starts_zero_copy_cpu_cases=6,
        gpu_graphs_tested=False)


def main():
    import torch
    if torch.cuda.is_initialized():raise ValueError('Unexpected CUDA context before CPU check')
    # Execute the real environment/import/transaction prefix only. The later
    # device-specific backend registration requires SM121 and is deliberately
    # left to GPU qualification; do not spoof capability or relax its guards.
    source=Path('/opt/ds41-serving/serve.py').read_text()
    boundary='from ds41.speculative_prefix_retention import register as register_speculative_prefix\n'
    if source.count(boundary)!=1:raise ValueError('Changed startup transaction boundary')
    exec(compile(source.split(boundary)[0],'/opt/ds41-serving/serve.py','exec'),
         {'__name__':'dspark_cpu_registration'})
    import spark_combined_miaai as combined
    result=combined.register()
    # Source attestation is CPU-only even though later backend registration
    # is SM121-specific. Include newly added/transitively changed files.
    from spark_backend_attestation import verify_sources
    verify_sources()
    if torch.cuda.is_initialized():raise ValueError('Registration created a CUDA context')
    from ds41.dspark_experiment import features
    from vllm.models.deepseek_v4_1.nvidia.dspark import DSparkDeepseekV4Model
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    def names(fn):
        while hasattr(fn,'__wrapped__'):fn=fn.__wrapped__
        return set(fn.__code__.co_names)
    if features.KV_ONLY and '_ds41_context_kv' not in names(DSparkDeepseekV4Model.precompute_and_store_context_kv):
        raise ValueError('KV projection patch missing')
    if features.MARKOV_ADD and '_ds41_sample_bias' not in names(DSparkSpeculator._sample_sequential):
        raise ValueError('Sampling patch missing')
    try:from ds41.dspark_experiment import policy
    except ImportError:policy=None
    confidence_checks=None
    if Path('/opt/ds41-serving/ds41/dspark_experiment/confidence.py').exists():
        confidence_checks=check_confidence_graphs()
    if policy is not None and policy.VERIFICATION=='ema':
        for fn,key in ((Scheduler.schedule,'_ds41_choose_prefix'),(Scheduler.update_from_output,'_ds41_prefix_policy'),
                       (CudaGraphManager._init_candidates,'_ds41_prefix_graph_lens')):
            if key not in names(fn):raise ValueError('EMA integration missing: '+key)
    if torch.cuda.is_initialized():raise ValueError('CPU checks created a CUDA context')
    report=dict(status='native_cpu_registration_passed',cuda_context_created=False,
        kv_only=features.KV_ONLY,markov_add=features.MARKOV_ADD,
        verification=policy.VERIFICATION if policy else ('confidence' if confidence_checks else 'fixed'),
        confidence_checks=confidence_checks,serving_qualified=False,
        scope='combined startup patch transaction only; GPU backend guards not executed')
    with Path('/results/complete.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
