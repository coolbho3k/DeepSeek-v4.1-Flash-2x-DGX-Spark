# SPDX-License-Identifier: AGPL-3.0-only
"""Diagnostic: compare offline DSpark drafts with the drafts serving actually proposed.

Uses the serving_drafts log (anchor position + K drafts per step) of decode-time
captures. For each non-prompt-lookup step the offline model drafts greedily
from the same context (Markov head fed its own previous drafts, as serving
does). Reports exact agreement per draft position for several variants, to find
which convention reproduces serving.
"""
import argparse
import json
from pathlib import Path

import torch

import dspark_torch as D
from train import load_sequence, ngram_drafts


@torch.no_grad()
def offline_drafts(model, seq, anchors_q, k, device, ctx_shift=0):
    tokens = seq['tokens'].to(device)
    ctx = model.context_kv(seq['aux'].to(device), seq['positions'].to(device))
    q = torch.tensor(anchors_q, device=device)
    c = q - 1 - ctx_shift
    orig = model.gather_window
    model.gather_window = lambda kv, anchors: (lambda w, m: (w, m & (anchors[:, None] >= 0)))(*orig(kv, anchors))
    h = model.draft_hidden(ctx, c.clamp_min(0), tokens[q], q, k)
    model.gather_window = orig
    last = model.layers[-1]
    base = torch.nn.functional.linear(h.to(model.head.dtype), model.head).float()
    prev = tokens[q]
    out = []
    for j in range(k):
        m = last.markov_embed(prev)
        d = (base[:, j] + last.markov_head(m).float()).argmax(-1)
        out.append(d)
        prev = d
    return torch.stack(out, 1).cpu()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('/data'))
    p.add_argument('--glob', default='gate1-*')
    p.add_argument('--k', type=int, default=3)
    dev = torch.device('cuda')
    p.add_argument('--experts', default='fp4')
    variants = [('reference', dict(quant_kv=False, ctx_shift=0, causal=False, act=False, qn=False)),
                ('q_head_norm', dict(quant_kv=False, ctx_shift=0, causal=False, act=False, qn=True)),
                ('q_head_norm_fp8', dict(quant_kv=True, ctx_shift=0, causal=False, act=True, qn=True))]
    a = p.parse_args()
    with torch.device(dev):
        base = D.DSpark(D.Args())
    D.load(base, '/draft', '/target', dev, experts=a.experts, exl3_dir='/draft-exl3')
    print('experts', a.experts, flush=True)
    base.eval()
    for name, v in variants:
        for layer in base.layers:
            layer.attn.quant_kv = v['quant_kv']
            layer.attn.causal_block = v['causal']
            layer.attn.q_head_norm = v.get('qn', False)
        D.QLinear.quant_act = v['act']
        base.a.window = v.get('window', 128)
        agree = torch.zeros(a.k)
        total = 0
        for folder in sorted(a.data.glob(a.glob)):
            seq = load_sequence(folder)
            if 'serving_drafts' not in seq:
                continue
            P = seq['meta']['prompt_len']
            hist = seq['tokens'].tolist()
            steps = [s for s in seq['serving_drafts'].tolist()
                     if P <= s[0] < len(hist) and ngram_drafts(hist[:s[0] + 1], a.k) is None]
            if not steps:
                continue
            with torch.autocast('cuda', dtype=torch.bfloat16):
                off = offline_drafts(base, seq, [s[0] for s in steps], a.k, dev, v['ctx_shift'])
            srv = torch.tensor([s[1:1 + a.k] for s in steps])
            agree += (off == srv).float().sum(0)
            total += len(steps)
        print(json.dumps(dict(variant=name, steps=total,
                              agreement_per_position=[round(x, 4) for x in (agree / max(total, 1)).tolist()])), flush=True)


if __name__ == '__main__':
    main()
