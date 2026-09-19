# SPDX-License-Identifier: AGPL-3.0-only
"""Startup-transaction hooks for prefix-only EMA verification.

Native vLLM retains its Apache-2.0 implementation and rejection sampler.
GLM/MiaAI adaptive lessons: prepare async placeholders at schedule time,
trim synchronous draft IDs, and capture every admitted uniform query length.
No acceptance decision or unverified output token is changed here.
"""
import hashlib
import importlib
from pathlib import Path

UPSTREAM={
    'vllm.v1.core.sched.scheduler':'e5e1c18b1d7a6ea73adbb4921f64a35b00a96abe673691bf6a7f57281a524519',
    'vllm.v1.worker.gpu.cudagraph_utils':'211de2232fb71aeb761dedbd7fadb7e0832db9331eac6951a1eafb6d77f1be9c',
}
OBSERVE=('num_accepted = max(len(generated_token_ids) - num_sampled, 0)',
         'num_accepted = max(len(generated_token_ids) - num_sampled, 0)\n'
         '            if not output_is_stale and not request.use_structured_output:\n'
         '                _ds41_prefix_policy(self).observe(req_id, num_draft_tokens, num_accepted)')
SCHEDULE=('num_spec_tokens_to_schedule = self.num_spec_tokens',
          'num_spec_tokens_to_schedule = _ds41_choose_prefix(self, num_scheduled_tokens)')
GRAPHS=('decode_query_lens = [self.decode_query_len]',
        'decode_query_lens = _ds41_prefix_graph_lens(self)')


def settings():
    from . import policy
    if (policy.VERIFICATION!='ema' or policy.DRAFT_TOKENS not in (3,4,5)
            or policy.PREFIX_LENGTHS[-1]!=policy.DRAFT_TOKENS):
        raise ValueError('Unreviewed EMA boot selection')
    return policy


def prefix_policy(scheduler):
    from .adaptive import PrefixEMA
    config=settings()
    if scheduler.num_spec_tokens!=config.DRAFT_TOKENS or scheduler.dynamic_sd_lookup is not None:
        raise ValueError('EMA cannot mix with another dynamic speculation policy')
    value=getattr(scheduler,'_ds41_prefix_ema',None)
    if value is None:
        value=PrefixEMA(lengths=config.PREFIX_LENGTHS)
        scheduler._ds41_prefix_ema=value
    return value


def choose_prefix(scheduler,scheduled):
    value=prefix_policy(scheduler)
    value.prune(scheduler.requests)
    requests=[scheduler.requests[key] for key in scheduled if key in scheduler.requests
              and not scheduler.requests[key].is_prefill_chunk]
    return value.choose([r.request_id for r in requests],
        structured=[r.request_id for r in requests if r.use_structured_output])


def graph_lens(manager):
    config=settings()
    # DFlashCudaGraphManager inherits the same initializer but its backbone
    # still predicts the full K. Only target verification gains extra graphs.
    if type(manager).__name__=='DFlashCudaGraphManager':
        if manager.decode_query_len!=config.DRAFT_TOKENS:
            raise ValueError('Unexpected DSpark draft graph geometry')
        return [manager.decode_query_len]
    if type(manager).__name__!='ModelCudaGraphManager' or manager.decode_query_len!=config.DRAFT_TOKENS+1:
        raise ValueError('Unknown speculative graph owner')
    return [k+1 for k in config.PREFIX_LENGTHS]


def make_patches():
    from ds41.vllm_dcp import _compile
    modules={name:importlib.import_module(name) for name in UPSTREAM}
    for name,module in modules.items():
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=UPSTREAM[name]:
            raise ValueError('Changed native adaptive integration source: '+name)
    scheduler=modules['vllm.v1.core.sched.scheduler'].Scheduler
    graphs=modules['vllm.v1.worker.gpu.cudagraph_utils'].CudaGraphManager
    observe=_compile(scheduler.update_from_output,[OBSERVE],{'_ds41_prefix_policy':prefix_policy})
    schedule=_compile(scheduler.schedule,[SCHEDULE],{'_ds41_choose_prefix':choose_prefix})
    graph_init=_compile(graphs._init_candidates,[GRAPHS],{'_ds41_prefix_graph_lens':graph_lens})
    original_update=scheduler.update_draft_token_ids
    def update(instance,draft_token_ids):
        result=original_update(instance,draft_token_ids)
        requests=[instance.requests[rid] for rid in draft_token_ids.req_ids
            if rid in instance.requests and not instance.requests[rid].is_finished()
            and not instance.requests[rid].is_prefill_chunk and instance.requests[rid].spec_token_ids]
        value=prefix_policy(instance);value.prune(instance.requests)
        length=value.choose([r.request_id for r in requests],
            structured=[r.request_id for r in requests if r.use_structured_output])
        for request in requests:
            request.spec_token_ids=request.spec_token_ids[:length]
        return result
    return [(scheduler,'update_from_output',observe),(scheduler,'schedule',schedule),
            (scheduler,'update_draft_token_ids',update),(graphs,'_init_candidates',graph_init)]
