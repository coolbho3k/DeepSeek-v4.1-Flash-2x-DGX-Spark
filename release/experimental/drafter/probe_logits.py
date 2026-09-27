# SPDX-License-Identifier: AGPL-3.0-only
"""Compare serving's cached draft logits (top-32 per position) with our port's logits at the same
anchor, feeding the Markov head serving's own drafted tokens so both see identical inputs.

Near-tie numerics -> high correlation, small shifts. Structural difference -> low correlation.
"""
import json
from pathlib import Path

import torch

import dspark_torch as D
from train import load_sequence, ngram_drafts

K = 3
dev = torch.device('cuda')
with torch.device(dev):
    port = D.DSpark(D.Args())
D.load(port, '/draft', '/target', dev, experts='exl3', exl3_dir='/draft-exl3')
port.eval()
stats = {j: dict(n=0, corr=0.0, top1_same=0, port_rank_of_serving_top1=0.0, gap=0.0) for j in range(K)}
for folder in sorted(Path('/data').glob('gate1-*')):
    seq = load_sequence(folder)
    if 'serving_draft_top_ids' not in seq:
        continue
    tok = seq['tokens'].tolist()
    P = seq['meta']['prompt_len']
    log = seq['serving_drafts'].tolist()
    keep = [i for i, s in enumerate(log) if P <= s[0] < len(tok) and ngram_drafts(tok[:s[0] + 1], K) is None]
    if not keep:
        continue
    q = torch.tensor([log[i][0] for i in keep], device=dev)
    served_drafts = torch.tensor([log[i][1:] for i in keep], device=dev)
    sv = seq['serving_draft_top_values'][keep].float().to(dev)          # [S, K, 32]
    si = seq['serving_draft_top_ids'][keep].long().to(dev)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        ctx = port.context_kv(seq['aux'].to(dev), seq['positions'].to(dev))
        tokens = seq['tokens'].to(dev)
        h = port.draft_hidden(ctx, q - 1, tokens[q], q, K)
        prev = torch.cat([tokens[q][:, None], served_drafts[:, :K - 1]], 1)
        logits, _ = port.logits(h, prev)                                 # [S, K, V]
    logits = logits.float()
    for j in range(K):
        pl = logits[:, j].gather(1, si[:, j])                            # port logits at serving's top-32 ids
        a = sv[:, j] - sv[:, j].mean(1, keepdim=True)
        b = pl - pl.mean(1, keepdim=True)
        corr = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-9)
        port_top = logits[:, j].argmax(-1)
        rank = (logits[:, j] > logits[:, j].gather(1, si[:, j, :1])).sum(1).float()
        gap = logits[:, j].max(-1).values - logits[:, j].gather(1, si[:, j, :1]).squeeze(1)
        st = stats[j]
        st['n'] += len(keep)
        st['corr'] += corr.sum().item()
        st['top1_same'] += (port_top == si[:, j, 0]).sum().item()
        st['port_rank_of_serving_top1'] += rank.sum().item()
        st['gap'] += gap.sum().item()
for j, st in stats.items():
    n = max(st['n'], 1)
    print(json.dumps(dict(position=j + 1, steps=st['n'], mean_logit_corr_on_serving_top32=round(st['corr'] / n, 4),
                          top1_agreement=round(st['top1_same'] / n, 4),
                          mean_port_rank_of_serving_top1=round(st['port_rank_of_serving_top1'] / n, 2),
                          mean_port_logit_gap_to_serving_top1=round(st['gap'] / n, 3))))
