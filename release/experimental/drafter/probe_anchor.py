# SPDX-License-Identifier: AGPL-3.0-only
"""Which anchor token does serving actually feed the drafter? Compare serving drafts with port drafts
computed from the correct anchor token and from stale candidates."""
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


@torch.no_grad()
def drafts(seq, anchors, anchor_tokens):
    ctx = port.context_kv(seq['aux'].to(dev), seq['positions'].to(dev))
    q = torch.tensor(anchors, device=dev)
    at = torch.tensor(anchor_tokens, device=dev)
    h = port.draft_hidden(ctx, q - 1, at, q, K)
    last = port.layers[-1]
    base = torch.nn.functional.linear(h.to(port.head.dtype), port.head).float()
    prev, out = at, []
    for j in range(K):
        d = (base[:, j] + last.markov_head(last.markov_embed(prev)).float()).argmax(-1)
        out.append(d); prev = d
    return torch.stack(out, 1).cpu()


tot = {}
n = 0
for folder in sorted(Path('/data').glob('gate1-*')):
    seq = load_sequence(folder)
    tok = seq['tokens'].tolist()
    P = seq['meta']['prompt_len']
    log = [s for s in seq['serving_drafts'].tolist() if P < s[0] < len(tok)]
    steps = [(i, s) for i, s in enumerate(log) if ngram_drafts(tok[:s[0] + 1], K) is None and i > 0]
    if not steps:
        continue
    srv = torch.tensor([s[1:] for _, s in steps])
    anchors = [s[0] for _, s in steps]
    cands = {
        'correct_token_at_anchor': [tok[q] for q in anchors],
        'token_at_anchor-1': [tok[q - 1] for q in anchors],
        'previous_step_anchor_token': [tok[log[i - 1][0]] for i, _ in steps],
        'previous_step_first_draft': [log[i - 1][1] for i, _ in steps],
    }
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for name, at in cands.items():
            agree = (drafts(seq, anchors, at) == srv).float().sum(0)
            tot[name] = tot.get(name, 0) + agree
    n += len(steps)
for name, v in tot.items():
    print(json.dumps(dict(variant=name, steps=n, agreement=[round(x, 4) for x in (v / n).tolist()])))
