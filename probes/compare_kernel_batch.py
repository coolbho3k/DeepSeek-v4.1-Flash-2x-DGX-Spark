# SPDX-License-Identifier: AGPL-3.0-only
"""Compare completed, request-matched kernel serving trials without GPU work."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def configuration(config):
    return {key: ([{k: v for k, v in node.items() if k != 'kit'} for node in value]
                  if key == 'nodes' else value)
            for key, value in config.items() if key not in ('run_id', 'kit_manifest_sha256')}


def pooled(rows):
    tokens = sum(row['timing']['decode_tokens'] for row in rows)
    seconds = sum(row['timing']['server_decode_seconds'] for row in rows)
    return tokens / seconds


def compare(old, new):
    if old['status'] != 'complete' or new['status'] != 'complete':
        raise ValueError('Only completed trials may be compared')
    if configuration(old['deployment']) != configuration(new['deployment']):
        raise ValueError('Weights, memory, API, image, or serving configuration changed')
    if old['source_sha256'] != new['source_sha256']:
        raise ValueError('Benchmark harness changed')
    if not old['cases'] or len(old['cases']) != len(new['cases']):
        raise ValueError('Unmatched number of requests')
    pairs = []
    for before, after in zip(old['cases'], new['cases'], strict=True):
        if before['label'] != after['label'] or before['request'] != after['request']:
            raise ValueError('Request identity/order mismatch')
        if before['usage']['completion_tokens'] != after['usage']['completion_tokens']:
            raise ValueError('Output token counts differ; not a capped matched trial')
        old_tps = before['timing']['decode_tokens_per_second']
        new_tps = after['timing']['decode_tokens_per_second']
        pairs.append(dict(label=before['label'], temperature=before['request']['temperature'],
            seed=before['request']['seed'], baseline_decode_tps=old_tps, candidate_decode_tps=new_tps,
            decode_change_percent=100 * (new_tps / old_tps - 1),
            baseline_step_ms=before['wall_ms_per_step'], candidate_step_ms=after['wall_ms_per_step'],
            step_change_percent=100 * (after['wall_ms_per_step'] / before['wall_ms_per_step'] - 1),
            baseline_acceptance=before['acceptance_fraction'], candidate_acceptance=after['acceptance_fraction'],
            acceptance_change_percentage_points=100 * (after['acceptance_fraction'] - before['acceptance_fraction']),
            baseline_tokens_per_step=before['tokens_per_step'], candidate_tokens_per_step=after['tokens_per_step'],
            reply_identical=before['reply'] == after['reply']))
    summaries = {}
    for label, temperature in sorted({(p['label'], p['temperature']) for p in pairs}):
        rows = [p for p in pairs if (p['label'], p['temperature']) == (label, temperature)]
        summary = {key: statistics.median(p[key] for p in rows) for key in (
            'baseline_decode_tps', 'candidate_decode_tps', 'baseline_step_ms', 'candidate_step_ms',
            'baseline_acceptance', 'candidate_acceptance', 'baseline_tokens_per_step', 'candidate_tokens_per_step')}
        summary['decode_change_percent'] = 100 * (summary['candidate_decode_tps'] / summary['baseline_decode_tps'] - 1)
        summary['step_change_percent'] = 100 * (summary['candidate_step_ms'] / summary['baseline_step_ms'] - 1)
        summary['reply_identical_count'] = sum(p['reply_identical'] for p in rows)
        summary['requests'] = len(rows)
        summaries[f'{label}/T{temperature}'] = summary
    before_pooled = pooled(old['cases'])
    after_pooled = pooled(new['cases'])
    return dict(status='matched_comparison_complete', baseline_run=old['deployment']['run_id'],
        candidate_run=new['deployment']['run_id'], requests=len(pairs),
        pooled_baseline_decode_tps=before_pooled, pooled_candidate_decode_tps=after_pooled,
        pooled_decode_change_percent=100 * (after_pooled / before_pooled - 1),
        unchanged_serving_configuration=True, identical_request_schedule=True,
        reply_identical_count=sum(p['reply_identical'] for p in pairs),
        statistical_significance_established=False, quality_benchmark=False,
        rows=pairs, summary=summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous comparison')
    report = compare(json.loads(args.baseline.read_bytes()), json.loads(args.candidate.read_bytes()))
    report['inputs'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (args.baseline, args.candidate)}
    report['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
