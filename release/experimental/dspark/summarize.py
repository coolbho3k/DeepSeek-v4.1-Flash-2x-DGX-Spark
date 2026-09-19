# SPDX-License-Identifier: AGPL-3.0-only
"""Summarize isolated serving counters without publishing prompts or replies.

CPU-only. Acceptance per position is unconditional per recorded speculative
round, not conditional on reaching that position. Native scheduler counters
count scheduled drafts, not confidence-trimmed worker verification budgets.
Zero-draft rounds are omitted. Do not infer actual confidence trimming here.
"""
import argparse
import json
import math
from pathlib import Path
import re
import statistics


def summary(report):
    if report.get('status') not in ('complete', 'complete_server_left_running'):
        raise ValueError('Require a completed serial trial')
    groups = {}
    for case in report['cases']:
        counters = case['speculative_delta']
        def count(name):
            values = [value for key, value in counters.items()
                      if key.startswith('vllm:' + name + '{')]
            if len(values) != 1 or not math.isfinite(values[0]) or values[0] < 0:
                raise ValueError('Need one isolated engine counter: ' + name)
            return values[0]
        steps = count('spec_decode_num_drafts_total')
        drafts = count('spec_decode_num_draft_tokens_total')
        accepted = count('spec_decode_num_accepted_tokens_total')
        if not steps or not drafts or accepted > drafts:
            raise ValueError('Invalid speculative counters')
        positions = {}
        for key, value in counters.items():
            if not key.startswith('vllm:spec_decode_num_accepted_tokens_per_pos_total{'):
                continue
            match = re.search(r'position="(\d+)"', key)
            if not match or not math.isfinite(value) or not 0 <= value <= steps:
                raise ValueError('Invalid position counter')
            position = int(match[1])
            if position in positions:
                raise ValueError('Mixed position counters')
            positions[position] = value
        if sorted(positions) != list(range(len(positions))) or sum(positions.values()) != accepted:
            raise ValueError('Incomplete acceptance positions')
        rate = case['timing']['decode_tokens_per_second']
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError('Invalid useful decode throughput')
        key = case['label'] + '/T' + str(float(case['request']['temperature']))
        groups.setdefault(key, []).append(dict(rate=rate, steps=steps,
            drafts=drafts, accepted=accepted, positions=positions,
            step_ms=case['wall_ms_per_step']))
    if not groups:
        raise ValueError('Empty trial')
    result = {}
    for key, rows in groups.items():
        steps = sum(row['steps'] for row in rows)
        drafts = sum(row['drafts'] for row in rows)
        accepted = sum(row['accepted'] for row in rows)
        if any(row['positions'].keys() != rows[0]['positions'].keys() for row in rows):
            raise ValueError('Changed positional envelope within a group')
        result[key] = dict(samples=len(rows),
            median_decode_tps=statistics.median(row['rate'] for row in rows),
            median_step_ms=statistics.median(row['step_ms'] for row in rows),
            mean_scheduled_drafts=drafts/steps, acceptance_fraction=accepted/drafts,
            draft_counter_scope='Scheduler proposals in nonzero-draft rounds; not the confidence-trimmed verification budget.',
            accepted_drafts_per_step=accepted/steps,
            position_survival=[sum(row['positions'][p] for row in rows)/steps
                               for p in sorted(rows[0]['positions'])])
    return result


def compare(baseline, candidate):
    if baseline.keys() != candidate.keys():
        raise ValueError('Unmatched prompt/temperature groups')
    ratios = {key: candidate[key]['median_decode_tps']/baseline[key]['median_decode_tps']
              for key in baseline}
    return dict(per_group_speed_ratio=ratios,
        geometric_mean_speed_ratio=math.exp(statistics.mean(math.log(v) for v in ratios.values())),
        all_groups_improved=all(v > 1 for v in ratios.values()),
        caveat='Descriptive matched-prompt rates, not a significance or task-quality test.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', type=Path)
    args = parser.parse_args()
    results = {path.stem: summary(json.loads(path.read_bytes())) for path in args.reports}
    baseline = next(iter(results.values()))
    print(json.dumps(dict(trials=results, comparisons={name: compare(baseline, value)
        for name, value in list(results.items())[1:]}), indent=2))


if __name__ == '__main__':
    main()
