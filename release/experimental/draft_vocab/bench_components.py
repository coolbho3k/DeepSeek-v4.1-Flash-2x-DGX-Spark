# SPDX-License-Identifier: AGPL-3.0-only
"""Component check: subset draft projections versus the full shared heads.

Synthetic BF16 weights at the real per-rank head shapes. Correctness compares
the gathered-row projection with torch's projection of the same rows. Timing
uses CUDA graphs cycling through enough weight copies to exceed L2.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from draft_vocab import gather_rows_gemv


def graph_time(fns, repeats=6):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns: f()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(repeats):
            for f in fns: f()
    g.replay(); torch.cuda.synchronize()
    out = []
    for _ in range(7):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        out.append(a.elapsed_time(b) * 1000 / (repeats * len(fns)))
    return sorted(out)[3]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.manual_seed(7)
    results = []
    with torch.inference_mode():
        for label, (v, k), subsets in (('lm_head_rank', (64640, 5120), (8192, 16384, 24576)),
                                       ('markov_head', (129280, 256), (16384, 32768, 49152))):
            copies = max(2, math.ceil(160 * 2**20 / (v * k * 2)))
            ws = [torch.randn((v, k), device='cuda', dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
            for rows in (3, 4, 12, 18):
                x = torch.randn((rows, k), device='cuda', dtype=torch.bfloat16)
                full = graph_time([lambda w=w: torch.nn.functional.linear(x, w) for w in ws])
                for n in subsets:
                    idx = torch.randperm(v, device='cuda')[:n].sort().values.to(torch.int32)
                    got = gather_rows_gemv(x, ws[0], idx, torch.bfloat16)
                    ref = torch.nn.functional.linear(x, ws[0][idx.long()])
                    diff = (got.float() - ref.float())
                    t = graph_time([lambda w=w: gather_rows_gemv(x, w, idx, torch.bfloat16) for w in ws])
                    row = dict(head=label, rows=rows, subset=n, full_us=full, subset_us=t,
                               saved_us=full - t, identical_fraction=float((got == ref).float().mean()),
                               nmse=float(diff.square().sum() / ref.float().square().sum()),
                               subset_gbps=n * k * 2 / t / 1e3, full_gbps=v * k * 2 / full / 1e3)
                    results.append(row)
                    print(json.dumps(row), flush=True)
            del ws
            torch.cuda.empty_cache()
    a.output.write_text(json.dumps(results, indent=1) + '\n')


if __name__ == '__main__':
    main()
