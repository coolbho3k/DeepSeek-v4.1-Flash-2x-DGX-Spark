# SPDX-License-Identifier: AGPL-3.0-only
"""Diagnostic: which context/position convention reproduces serving acceptance?"""
import json, sys
from pathlib import Path
import torch
import dspark_torch as D
from train import greedy_accepts, load_sequence, walk

served = {c['label']: c['tokens_per_step'] for c in json.loads(Path('/runs/gate1-bench.json').read_text())['cases'] if c['temperature'] == 0}
dev = torch.device('cuda')
with torch.device(dev):
    m = D.DSpark(D.Args())
D.load(m, '/draft', '/target', dev)
m.eval()
orig = m.gather_window
variants = {
    'reference': dict(ctx_shift=0),
    'context_excludes_last': dict(ctx_shift=1),
    'context_excludes_last2': dict(ctx_shift=2),
}
for name, v in variants.items():
    def gw(kv, anchors, s=v['ctx_shift']):
        w, mask = orig(kv, anchors - s)
        return w, mask & ((anchors - s)[:, None] >= 0)
    m.gather_window = gw
    ratios = []
    for folder in sorted(Path('/data').glob('gate1-*')):
        label = folder.name.removeprefix('gate1-')
        if label in ('code_edit', 'json_rename'):
            continue
        seq = load_sequence(folder)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            a, acc = greedy_accepts(m, seq, 3, dev)
        p, s = walk(seq, a, acc, 3)
        ratios.append((label, round(p / s / served[label], 3)))
    print(name, ratios, 'mean', round(sum(r for _, r in ratios) / len(ratios), 3), flush=True)
