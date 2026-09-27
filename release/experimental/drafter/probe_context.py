# SPDX-License-Identifier: AGPL-3.0-only
"""Which context positions does serving's drafter actually see? Mask patterns vs serving drafts."""
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
orig = port.gather_window


def make(rule, prompt_len):
    def gw(kv, anchors):
        w, mask = orig(kv, anchors)
        pos = anchors[:, None] - (port.a.window - 1) + torch.arange(port.a.window, device=anchors.device)
        return w, mask & rule(pos, anchors[:, None], prompt_len)
    return gw


rules = {
    'all': lambda p, c, P: torch.ones_like(p, dtype=torch.bool),
    'even_positions': lambda p, c, P: p % 2 == 0,
    'odd_positions': lambda p, c, P: p % 2 == 1,
    'block32_aligned_even': lambda p, c, P: (p // 32) % 2 == 0,
    'block32_aligned_odd': lambda p, c, P: (p // 32) % 2 == 1,
    'prompt_only': lambda p, c, P: p < P,
    'generated_only': lambda p, c, P: p >= P,
    'last_32': lambda p, c, P: p > c - 32,
}
tot = {k: 0 for k in rules}
n = 0
for folder in sorted(Path('/data').glob('gate1-*')):
    seq = load_sequence(folder)
    tok = seq['tokens'].tolist()
    P = seq['meta']['prompt_len']
    steps = [s for s in seq['serving_drafts'].tolist() if P < s[0] < len(tok) and ngram_drafts(tok[:s[0] + 1], K) is None]
    if not steps:
        continue
    srv = torch.tensor([s[1:] for s in steps])
    q = torch.tensor([s[0] for s in steps], device=dev)
    for name, rule in rules.items():
        port.gather_window = make(rule, P)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            ctx = port.context_kv(seq['aux'].to(dev), seq['positions'].to(dev))
            h = port.draft_hidden(ctx, q - 1, seq['tokens'].to(dev)[q], q, K)
            last = port.layers[-1]
            base = torch.nn.functional.linear(h.to(port.head.dtype), port.head).float()
            prev = seq['tokens'].to(dev)[q]
            d = []
            for j in range(K):
                x = (base[:, j] + last.markov_head(last.markov_embed(prev)).float()).argmax(-1)
                d.append(x); prev = x
        tot[name] += (torch.stack(d, 1).cpu() == srv).float().sum(0)
    n += len(steps)
for name, v in tot.items():
    print(json.dumps(dict(variant=name, steps=n, agreement=[round(x, 4) for x in (v / n).tolist()])))
