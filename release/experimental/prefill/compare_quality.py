# SPDX-License-Identifier: AGPL-3.0-only
"""Compare two quality_eval.py outputs (control vs candidate)."""
import argparse
import json
import math
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('control', type=Path)
    p.add_argument('candidate', type=Path)
    a = p.parse_args()
    c, d = (json.loads(x.read_bytes()) for x in (a.control, a.candidate))
    rows, deltas = [], []
    for x, y in zip(c['documents'], d['documents']):
        assert x['records'] == y['records'] and len(x['per_token']) == len(y['per_token'])
        dl = [v[0] - u[0] for u, v in zip(x['per_token'], y['per_token'])]
        deltas += dl
        nll_c = -statistics.fmean(u[0] for u in x['per_token'])
        nll_d = -statistics.fmean(v[0] for v in y['per_token'])
        rows.append(dict(source=x['source'], tokens=x['tokens'],
            control_ppl=math.exp(nll_c), candidate_ppl=math.exp(nll_d),
            ppl_change_pct=100 * (math.exp(nll_d - nll_c) - 1),
            mean_abs_dlogprob=statistics.fmean(abs(v) for v in dl),
            max_abs_dlogprob=max(abs(v) for v in dl),
            identical_logprob_fraction=sum(v == 0 for v in dl) / len(dl),
            top1_agreement=sum(u[2] == v[2] for u, v in zip(x['per_token'], y['per_token'])) / len(dl)))
    replies = []
    for x, y in zip(c['replies'], d['replies']):
        s, t = x['reply'], y['reply']
        n = next((i for i, (p, q) in enumerate(zip(s, t)) if p != q), min(len(s), len(t)))
        replies.append(dict(prompt=x['prompt'][:40], identical=s == t, first_diff_char=None if s == t else n,
                            length=len(s)))
    summary = dict(
        documents=rows,
        overall_mean_abs_dlogprob=statistics.fmean(abs(v) for v in deltas),
        overall_mean_dlogprob=statistics.fmean(deltas),
        replies_identical=sum(r['identical'] for r in replies), replies=len(replies), reply_detail=replies)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
