# SPDX-License-Identifier: AGPL-3.0-only
"""Which TP-sharding mistake (if any) reproduces serving's drafts? Variants on per-head params."""
import json
from pathlib import Path
import torch
import dspark_torch as D
from probe_drafts import offline_drafts
from train import load_sequence, ngram_drafts

K = 3
dev = torch.device('cuda')
with torch.device(dev):
    port = D.DSpark(D.Args())
D.load(port, '/draft', '/target', dev, experts='exl3', exl3_dir='/draft-exl3')
port.eval()
orig = {i: l.attn.attn_sink.detach().clone() for i, l in enumerate(port.layers)}
variants = {
    'reference': lambda s: s,
    'sink_rank1_uses_rank0': lambda s: torch.cat([s[:32], s[:32]]),
    'sink_rank0_uses_rank1': lambda s: torch.cat([s[32:], s[32:]]),
    'sink_zero': lambda s: torch.zeros_like(s),
    'sink_neg_inf': lambda s: torch.full_like(s, -1e4),
}
seqs = []
for folder in sorted(Path('/data').glob('gate1-*')):
    seq = load_sequence(folder)
    tok = seq['tokens'].tolist(); P = seq['meta']['prompt_len']
    steps = [s for s in seq['serving_drafts'].tolist() if P <= s[0] < len(tok) and ngram_drafts(tok[:s[0] + 1], K) is None]
    if steps:
        seqs.append((seq, steps))
for name, f in variants.items():
    with torch.no_grad():
        for i, l in enumerate(port.layers):
            l.attn.attn_sink.copy_(f(orig[i]))
    agree, n = torch.zeros(K), 0
    for seq, steps in seqs:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            off = offline_drafts(port, seq, [s[0] for s in steps], K, dev)
        agree += (off == torch.tensor([s[1:] for s in steps])).float().sum(0); n += len(steps)
    print(json.dumps(dict(variant=name, agreement=[round(x, 4) for x in (agree / n).tolist()])), flush=True)
