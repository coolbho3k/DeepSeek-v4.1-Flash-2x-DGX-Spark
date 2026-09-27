# SPDX-License-Identifier: AGPL-3.0-only
"""Confidence-capped DSpark verification without changing the verify layout.

Idea and policy from MiaAI Lab's 3x/4x DGX Spark recipe (adapter/verify_cap.py,
itself from knapcio's TP4 fork, AGPL-3.0). This is a vLLM V2 reimplementation.

The target still verifies the fixed [anchor + K drafts] block per request, so
every captured graph keeps its shape. Per request, k = number of leading drafts
whose running product of the draft confidence head's per-position acceptance
probability stays >= THRESHOLD (at least MIN_DRAFTS). Position i's confidence
depends only on drafts before i, so the cut is a stopping rule that never looks
at the token it drops.

  * Rows after anchor + k ("dead" rows) take the anchor row's expert ids and
    weights in every target MoE layer, so they add no experts to the set the
    bandwidth-bound MoE streams. Live rows only attend to earlier rows and MoE
    is per-row, so their outputs are unchanged.
  * Rejection sampling runs unchanged; afterwards each request keeps at most
    1 + k tokens. Standard speculative sampling makes the token at position k
    (accepted draft or residual resample) distributed exactly as the target's,
    so truncating there is exact for greedy and sampled requests alike.
  * Prompt-lookup proposals are verified in full (k = K).
"""
import functools

import torch
import triton
import triton.language as tl

THRESHOLD = 0.1
MIN_DRAFTS = 1
# Remap only decode-sized batches (the captured target graph sizes).
MAX_ROWS = 36
# Rank 0 decides k: a live length that differed across TP ranks would give one
# verify row two expert sets and commit different tokens per rank.
BROADCAST = True

_state = dict(live=None, src=None, ident=None, steps=None, drafts=None, verified=None, calls=0)


@triton.jit
def _dead_rows(src, query_start, cu_logits, idx_map, live_by_req, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    end = tl.load(query_start + r + 1)
    n = tl.load(cu_logits + r + 1) - tl.load(cu_logits + r)  # anchor + scheduled drafts
    live = tl.load(live_by_req + tl.load(idx_map + r))
    start = end - n  # verify rows are the last n query rows of the request
    p = tl.arange(0, BLOCK)
    tl.store(src + start + p, start.to(tl.int64), mask=(p < n) & (p >= live))


def attach(runner):
    """Allocate persistent buffers before profiling and graph capture."""
    speculator = getattr(runner, 'speculator', None)
    if speculator is None or type(speculator).__name__ != 'DSparkSpeculator':
        return False
    if _state['live'] is not None:
        raise RuntimeError('Verify cap attached twice')
    k = speculator.num_speculative_steps
    device = runner.device
    _state['k'] = k
    _state['live'] = torch.full((runner.req_states.max_num_reqs,), k + 1, dtype=torch.int32, device=device)
    _state['ident'] = torch.arange(runner.max_num_tokens, dtype=torch.int64, device=device)
    _state['src'] = _state['ident'].clone()
    _state['steps'] = torch.zeros((), dtype=torch.int64, device=device)
    _state['drafts'] = torch.zeros((), dtype=torch.int64, device=device)
    _state['verified'] = torch.zeros((), dtype=torch.int64, device=device)
    print(f'[verify_cap] attached: K={k} threshold={THRESHOLD} min={MIN_DRAFTS} max_rows={MAX_ROWS}', flush=True)
    return True


def wrap_sample(original):
    """Also compute the draft confidence head inside the captured draft graph."""
    @functools.wraps(original)
    def _sample_sequential(self, num_reqs, head_hidden):
        original(self, num_reqs, head_hidden)
        if self.model.model.confidence_head is None:
            raise RuntimeError('Verify cap needs the DSpark confidence head')
        n = self.num_speculative_steps
        hidden = head_hidden[self.sample_indices[:num_reqs * n]]
        anchor = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]].to(torch.int64)
        prev = torch.cat([anchor[:, None], self.draft_tokens[:num_reqs, :n - 1].to(torch.int64)], dim=1)
        confidence = self.model.compute_confidence(hidden, self.model.markov_embed(prev.reshape(-1)))
        self.draft_token_confidence_probs[:num_reqs] = confidence.view(num_reqs, n)
    return _sample_sequential


def wrap_propose(original):
    """Turn this step's confidences into each request's next live length."""
    @functools.wraps(original)
    def propose(self, input_batch, *args, **kwargs):
        drafts = original(self, input_batch, *args, **kwargs)
        num_reqs = input_batch.num_reqs
        live = _state['live']
        if kwargs.get('dummy_run') or kwargs.get('is_profile') or live is None or num_reqs == 0:
            return drafts
        k_max = drafts.shape[1]
        cum = torch.cumprod(self.draft_token_confidence_probs[:num_reqs].clamp(0, 1), dim=1)
        k = (cum >= THRESHOLD).to(torch.int32).cumprod(dim=1).sum(dim=1, dtype=torch.int32)
        k = k.clamp_(min=MIN_DRAFTS, max=k_max)
        best = getattr(self, '_ds41_ngram_best', None)
        if best is not None:
            k = torch.where(best[:num_reqs] >= 0, k_max, k).to(torch.int32)
        if BROADCAST:
            from vllm.distributed import get_tp_group
            group = get_tp_group()
            if group.world_size > 1:
                group.broadcast(k, src=0)
        live[input_batch.idx_mapping[:num_reqs]] = k + 1
        _state['steps'] += num_reqs
        _state['drafts'] += num_reqs * k_max
        _state['verified'] += k.sum()
        _state['calls'] += 1
        if _state['calls'] % 1024 == 0:
            from vllm.distributed import get_tp_group
            if get_tp_group().rank_in_group == 0:
                steps, verified = int(_state['steps']), int(_state['verified'])
                print(f'[verify_cap] request-steps={steps} mean_verified_drafts={verified / max(steps, 1):.3f}'
                      f' of {k_max}', flush=True)
        return drafts
    return propose


def wrap_prepare_inputs(original):
    """Point each dead verify row at its anchor row for the coming target forward."""
    @functools.wraps(original)
    def prepare_inputs(self, *args, **kwargs):
        batch = original(self, *args, **kwargs)
        src = _state['src']
        rows = batch.num_tokens_after_padding
        if src is None or rows > MAX_ROWS:
            return batch
        src[:rows].copy_(_state['ident'][:rows])
        if batch.num_draft_tokens and batch.num_reqs:
            _dead_rows[(batch.num_reqs,)](src, batch.query_start_loc, batch.cu_num_logits,
                                         batch.idx_mapping, _state['live'],
                                         BLOCK=triton.next_power_of_2(_state['k'] + 1))
        return batch
    return prepare_inputs


def wrap_apply(original):
    """Target MoE (6 routed experts per row): dead rows use the anchor's experts."""
    @functools.wraps(original)
    def apply(self, layer, x, topk_weights, topk_ids, *args, **kwargs):
        src = _state['src']
        if src is not None and topk_ids.ndim == 2 and topk_ids.shape[1] == 6 and 1 < x.shape[0] <= MAX_ROWS:
            rows = src[:x.shape[0]]
            topk_ids = topk_ids.index_select(0, rows)
            topk_weights = topk_weights.index_select(0, rows)
        return original(self, layer, x, topk_weights, topk_ids, *args, **kwargs)
    return apply


def wrap_verify(original):
    """Keep at most 1 + k tokens per request (exact truncation, see module doc)."""
    @functools.wraps(original)
    def _verify(self, logits, draft_logits, draft_sampled, pos, cu_num_logits, idx_mapping, *args):
        processed, sampled, num_sampled = original(
            self, logits, draft_logits, draft_sampled, pos, cu_num_logits, idx_mapping, *args)
        live = _state['live']
        if live is not None:
            num_sampled = torch.minimum(num_sampled, live[idx_mapping].to(num_sampled.dtype))
        return processed, sampled, num_sampled
    return _verify


def make_patches():
    import importlib
    from vllm.v1.worker.gpu import model_runner
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler
    exl3 = importlib.import_module('ds41.vllm_exl3')
    runner, sampler, method = model_runner.GPUModelRunner, rejection_sampler.RejectionSampler, exl3.DS41EXL3MoEMethod
    return [
        (runner, 'prepare_inputs', wrap_prepare_inputs(runner.prepare_inputs)),
        (sampler, '_verify', wrap_verify(sampler._verify)),
        (method, 'apply', wrap_apply(method.apply)),
    ]
