# SPDX-License-Identifier: AGPL-3.0-only
"""Same anchors, two drafters: accepted drafts (vs the generated greedy tokens) of the drafts serving
actually proposed and of our port's drafts. Also serving tokens/step from its own draft log."""
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
served = {c['label']: c['tokens_per_step'] for c in json.loads(Path('/runs/gate1-bench.json').read_text())['cases'] if c['temperature'] == 0}
for folder in sorted(Path('/data').glob('gate1-*')):
    seq = load_sequence(folder)
    tok = seq['tokens'].tolist()
    P = seq['meta']['prompt_len']
    steps = [s for s in seq['serving_drafts'].tolist() if P <= s[0] < len(tok) - K]
    def acc(anchor, drafts):
        a = 0
        while a < K and drafts[a] == tok[anchor + 1 + a]:
            a += 1
        return a
    srv = [acc(s[0], s[1:]) for s in steps]
    dsp = [s for s in steps if ngram_drafts(tok[:s[0] + 1], K) is None]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        off = offline_drafts(port, seq, [s[0] for s in dsp], K, dev).tolist() if dsp else []
    srv_d = [acc(s[0], s[1:]) for s in dsp]
    off_d = [acc(s[0], d) for s, d in zip(dsp, off)]
    print(json.dumps(dict(case=folder.name, bench=served[folder.name[6:]], serving_log_tokens_per_step=round(1 + sum(srv) / max(len(srv), 1), 3),
                          dspark_steps=len(dsp), serving_mean_accept=round(sum(srv_d) / max(len(dsp), 1), 3),
                          port_mean_accept_same_anchors=round(sum(off_d) / max(len(dsp), 1), 3))), flush=True)
