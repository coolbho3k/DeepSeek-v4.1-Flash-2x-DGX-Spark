# SPDX-License-Identifier: AGPL-3.0-only
"""Gate 1: the PyTorch drafter must reproduce serving acceptance.

Compares offline greedy tokens/step (serving emulation incl. prompt lookup) on
the captured T=0 bench sequences with bench.py's measured serving tokens/step
for the same prompts. Pass: every case within ±5%, mean within ±2%.
"""
import argparse
import json
from pathlib import Path

import torch

from dspark_torch import Args, DSpark, load
from train import greedy_accepts, load_sequence, walk


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--bench', type=Path, required=True, help='bench.py report from the same kit')
    p.add_argument('--draft', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--quant-kv', action='store_true')
    p.add_argument('--checkpoint', type=Path)
    a = p.parse_args()
    device = torch.device('cuda')
    with torch.device(device):
        model = DSpark(Args(), quant_kv=a.quant_kv)
    load(model, a.draft, a.target, device)
    if a.checkpoint:
        model.load_state_dict(torch.load(a.checkpoint, map_location=device, weights_only=False)['model'], strict=False)
    model.eval()
    served = {c['label']: c for c in json.loads(a.bench.read_text())['cases'] if c['temperature'] == 0}
    rows, ratios = [], []
    for folder in sorted(a.data.glob('gate1-*')):
        label = folder.name.removeprefix('gate1-')
        seq = load_sequence(folder)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            anchors, acc = greedy_accepts(model, seq, a.k, device)
        produced, steps = walk(seq, anchors, acc, a.k)
        offline = produced / max(steps, 1)
        measured = served[label]['tokens_per_step']
        ratios.append(offline / measured)
        rows.append(dict(case=label, offline=round(offline, 3), serving=round(measured, 3),
                         ratio=round(offline / measured, 3), steps=steps,
                         text_matches=seq['tokens'][seq['meta']['prompt_len']:].numel()))
        print(json.dumps(rows[-1]), flush=True)
    mean = sum(ratios) / len(ratios)
    ok = all(abs(r - 1) <= .05 for r in ratios) and abs(mean - 1) <= .02
    print(json.dumps(dict(gate1='pass' if ok else 'fail', mean_ratio=round(mean, 4))), flush=True)


if __name__ == '__main__':
    main()
