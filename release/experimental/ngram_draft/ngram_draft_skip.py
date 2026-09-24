# SPDX-License-Identifier: AGPL-3.0-only
"""Prompt-lookup proposals layered on native DSpark drafts.

After DSpark proposes K tokens, each request's token history (prompt plus
generated tokens, already updated with this step's samples) is searched for the
most recent earlier occurrence of its last NGRAM tokens. When found, the K
tokens that followed that occurrence replace DSpark's proposals for that
request, and the cached draft distribution for those positions becomes a point
mass on the copied token (logit 0, -inf elsewhere). Probabilistic rejection
sampling then accepts with probability p_target(token) and otherwise resamples
from the residual, so outputs remain distributed exactly as the target model's.
Greedy requests compare tokens directly. When every request matches, the
DSpark draft generation for that step is skipped. Target and draft weights, KV and all
other verification behavior are unchanged.
"""
import functools

import torch
import triton
import triton.language as tl

NGRAM = 3
WINDOW = 1 << 16
SEARCH_BLOCK = 1024
VOCAB_BLOCK = 4096
hits = None  # device int64 counter of overridden requests
skipped = 0  # host count of skipped DSpark generations


@triton.jit
def _search(ids, stride, total, idx_map, best, window,
            N: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    req = tl.load(idx_map + r).to(tl.int64)
    length = tl.load(total + req)
    row = ids + req * stride
    lo = tl.maximum(length - window, 0)
    i = lo + tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    # The K following tokens must exist; this also excludes the suffix itself.
    ok = (i + N + K <= length) & (length >= 2 * N + K)
    for j in tl.static_range(N):
        a = tl.load(row + i + j, mask=ok, other=-1)
        b = tl.load(row + length - N + j)
        ok = ok & (a == b)
    found = tl.max(tl.where(ok, i, -1), axis=0)
    if found >= 0:
        tl.atomic_max(best + r, found)


@triton.jit
def _apply(ids, stride, idx_map, best, drafts, dstride, logits, ls0, ls1, vocab, counter,
           N: tl.constexpr, BLOCKV: tl.constexpr):
    r = tl.program_id(0)
    j = tl.program_id(1)
    vb = tl.program_id(2)
    start = tl.load(best + r)
    if start >= 0:
        req = tl.load(idx_map + r).to(tl.int64)
        token = tl.load(ids + req * stride + start + N + j).to(tl.int64)
        if vb == 0:
            tl.store(drafts + r * dstride + j, token)
            if j == 0:
                tl.atomic_add(counter, 1)
        v = vb * BLOCKV + tl.arange(0, BLOCKV)
        value = tl.where(v == token, 0.0, float('-inf'))
        tl.store(logits + req * ls0 + j * ls1 + v, value.to(logits.dtype.element_ty), mask=v < vocab)


def search(speculator, input_batch, k):
    """Return per-request match starts (int64, -1 = none), or None if unavailable."""
    global hits
    states = getattr(speculator, '_ds41_req_states', None)
    num_reqs = input_batch.num_reqs
    if states is None or speculator.draft_logits is None or num_reqs == 0:
        return None
    ids = states.all_token_ids.gpu
    if hits is None:
        hits = torch.zeros(1, dtype=torch.int64, device=ids.device)
    best = torch.full((num_reqs,), -1, dtype=torch.int64, device=ids.device)
    _search[(num_reqs, triton.cdiv(WINDOW, SEARCH_BLOCK))](
        ids, ids.stride(0), states.total_len.gpu, input_batch.idx_mapping[:num_reqs], best, WINDOW,
        N=NGRAM, K=k, BLOCK=SEARCH_BLOCK, num_warps=4)
    return best


def apply(speculator, input_batch, drafts, best):
    states = speculator._ds41_req_states
    logits = speculator.draft_logits
    ids = states.all_token_ids.gpu
    _apply[(input_batch.num_reqs, drafts.shape[1], triton.cdiv(logits.shape[-1], VOCAB_BLOCK))](
        ids, ids.stride(0), input_batch.idx_mapping[:input_batch.num_reqs], best, drafts, drafts.stride(0),
        logits, logits.stride(0), logits.stride(1), logits.shape[-1], hits,
        N=NGRAM, BLOCKV=VOCAB_BLOCK, num_warps=4)
    return drafts


def wrap(original):
    """Wrap native DSpark propose; installed with the reviewed DSpark patch set.

    When every request in the batch has a prompt-lookup match, the DSpark draft
    generation (backbone forward, output head, Markov head) is skipped for the
    step: its proposals would all be replaced. Context KV insertion and every
    other part of the native propose still run, so draft state stays current.
    """
    if getattr(original, '_ds41_ngram', False):
        raise RuntimeError('Prompt-lookup drafting wrapped twice')

    @functools.wraps(original)
    def propose(self, input_batch, *args, **kwargs):
        if kwargs.get('dummy_run') or kwargs.get('is_profile'):
            return original(self, input_batch, *args, **kwargs)
        best = search(self, input_batch, self.num_speculative_steps)
        self._ds41_skip_draft = best is not None and bool((best >= 0).all())
        try:
            drafts = original(self, input_batch, *args, **kwargs)
        finally:
            self._ds41_skip_draft = False
        if best is not None:
            apply(self, input_batch, drafts, best)
        return drafts

    propose._ds41_ngram = True
    return propose


def wrap_generate(original):
    """Skip native draft generation only when propose decided every request is copied."""
    @functools.wraps(original)
    def generate(self, *args, **kwargs):
        if getattr(self, '_ds41_skip_draft', False):
            global skipped
            skipped += 1
            return None
        return original(self, *args, **kwargs)
    generate._ds41_ngram = True
    return generate


def attach(runner):
    """Give the loaded speculator read access to request token history."""
    speculator = getattr(runner, 'speculator', None)
    if speculator is None or speculator.draft_logits is None:
        return False
    if speculator.draft_logits.shape[1] != speculator.num_speculative_steps:
        raise RuntimeError('Unexpected draft logits cache layout')
    speculator._ds41_req_states = runner.req_states
    return True
