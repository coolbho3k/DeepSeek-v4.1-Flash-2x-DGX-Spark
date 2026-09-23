# SPDX-License-Identifier: AGPL-3.0-only
"""Experimental frequency-ranked draft vocabulary for native DSpark.

The drafter shares the target's vocabulary-parallel output head. Instead of
projecting every draft position onto all 129,280 rows, project onto a fixed
frequent subset of each rank's rows (read in place, no weight copy) and let
the native speculator scatter those logits into target-vocabulary space with
-inf elsewhere. Rejection sampling then remains exact for the target model:
only the proposal distribution changes. The Markov transition bias (a
replicated low-rank head) is restricted to the same subset.

Target logits, weights, KV, embeddings and verification are unchanged.
"""
import hashlib
import json
from pathlib import Path
import types

import torch
import triton
import triton.language as tl

VOCAB = 129280
HALF = VOCAB // 2


@triton.jit
def _gather_rows_gemv(X, W, IDX, Y, R, K, stride_x, stride_w, stride_y, N,
                      RP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # Y[r, j] = sum_k X[r, k] * W[IDX[j], k]; BF16 inputs, FP32 accumulation.
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    rows = tl.arange(0, RP)
    live = cols < N
    ids = tl.load(IDX + cols, mask=live, other=0).to(tl.int64)
    acc = tl.zeros((RP, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        x = tl.load(X + rows[:, None] * stride_x + k[None, :],
                    mask=(rows[:, None] < R) & (k[None, :] < K), other=0.)
        w = tl.load(W + ids[None, :] * stride_w + k[:, None],
                    mask=live[None, :] & (k[:, None] < K), other=0.)
        acc += tl.dot(x, w)
    tl.store(Y + rows[:, None] * stride_y + cols[None, :], acc.to(Y.dtype.element_ty),
             mask=(rows[:, None] < R) & live[None, :])


def gather_rows_gemv(x, weight, index, out_dtype):
    """x [R, K] BF16 contiguous rows; weight [V, K]; index [N] int32 rows."""
    r, k = x.shape
    if (x.dtype != weight.dtype or weight.shape[1] != k or x.stride(1) != 1
            or weight.stride(1) != 1 or not 1 <= r <= 64 or index.dtype != torch.int32):
        raise ValueError('Unsupported draft-vocabulary projection layout')
    n = index.numel()
    y = torch.empty((r, n), device=x.device, dtype=out_dtype)
    rp = 16 if r <= 16 else 32 if r <= 32 else 64
    bk = 128 if k >= 128 else 64
    _gather_rows_gemv[(triton.cdiv(n, 64),)](x, weight, index, y, r, k, x.stride(0),
        weight.stride(0), y.stride(0), n, RP=rp, BN=64, BK=bk, num_warps=4)
    return y


def load_subset(path, expected_sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Draft vocabulary subset changed')
    subset = json.loads(raw)
    ranks = subset['ranks']
    n = subset['per_rank']
    if (subset.get('format') != 'ds41_draft_vocab_v1' or subset.get('vocab') != VOCAB
            or len(ranks) != 2 or any(len(r) != n for r in ranks)
            or any(r != sorted(set(r)) for r in ranks)
            or any(not all(i * HALF <= t < (i + 1) * HALF for t in r) for i, r in enumerate(ranks))):
        raise ValueError('Invalid balanced draft vocabulary subset')
    return ranks


def install(model, ranks):
    """Override the drafter's logits hooks on this loaded instance only."""
    from vllm.distributed import (get_tensor_model_parallel_rank,
                                  get_tensor_model_parallel_world_size,
                                  tensor_model_parallel_all_gather)
    if get_tensor_model_parallel_world_size() != 2:
        raise ValueError('Draft vocabulary requires TP2')
    rank = get_tensor_model_parallel_rank()
    head = model.lm_head
    markov = model.model.markov_head.markov_w2
    processor = model.logits_processor
    weight = head.weight
    device = weight.device
    shard_rows = getattr(head, 'num_embeddings_per_partition', None)
    if (weight.shape != (HALF, 5120) or shard_rows != HALF or weight.dtype != torch.bfloat16
            or markov.weight.shape != (VOCAB, 256) or processor.soft_cap is not None
            or processor.scale != 1.0 or processor.org_vocab_size != VOCAB
            or not processor.use_all_gather):
        raise ValueError('Unexpected shared output head or Markov head layout')
    out_dtype = processor.head_dtype or torch.bfloat16
    local = torch.tensor([t - rank * HALF for t in ranks[rank]], dtype=torch.int32, device=device)
    targets = torch.tensor(ranks[0] + ranks[1], dtype=torch.int64, device=device)
    markov_rows = targets.to(torch.int32)
    positions = torch.arange(targets.numel(), dtype=torch.int64, device=device)

    def compute_draft_logits(self, hidden_states):
        x = self.model.norm(hidden_states).contiguous()
        logits = gather_rows_gemv(x, weight, local, out_dtype)
        return tensor_model_parallel_all_gather(logits)

    def markov_bias(self, markov_embed):
        return gather_rows_gemv(markov_embed.contiguous(), markov.weight, markov_rows, out_dtype)

    def map_draft_to_target(self, draft_ids):
        return targets[draft_ids]

    model.compute_draft_logits = types.MethodType(compute_draft_logits, model)
    model.markov_bias = types.MethodType(markov_bias, model)
    model.map_draft_to_target = types.MethodType(map_draft_to_target, model)
    # Native convention: target id = draft id + d2t[draft id].
    model.draft_id_to_target_id = targets - positions
    model._ds41_draft_vocab = dict(per_rank=len(ranks[0]), total=targets.numel())
    return model
