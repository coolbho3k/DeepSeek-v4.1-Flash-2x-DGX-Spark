# SPDX-License-Identifier: AGPL-3.0-only
"""Compare block verification with saved standard-verification measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports', type=Path, required=True)
    a = p.parse_args()
    names = ['baseline-serial.json', 'baseline-c6.json', 'candidate.json',
             'candidate-repeat.json', 'candidate-c6-repeat.json']
    data = [json.loads((a.reports / name).read_bytes()) for name in names]
    assert all(r['status'] == 'complete' for r in data)
    before, concurrent, first, second, third = data
    assert first['run_id'] == second['run_id'] == third['run_id']
    config = json.loads(Path(first['deployment']).read_bytes())
    assert config['serving'] == before['deployment']['serving']
    assert config['model_manifest_sha256'] == before['deployment']['model_manifest_sha256']
    for left, right in zip(before['deployment']['nodes'], config['nodes'], strict=True):
        for field in ('image', 'model', 'model_bindings', 'draft', 'cache', 'rails'):
            assert left.get(field) == right.get(field), field
    rows = []
    for old, left, right in zip(before['cases'], first['cases'], second['cases'], strict=True):
        assert old['label'] == left['label'] == right['label']
        assert old['request'] == left['request'] == right['request']
        values = [r['timing']['decode_tokens_per_second'] for r in (left, right)]
        baseline = old['timing']['decode_tokens_per_second']
        rows.append(dict(label=old['label'], baseline_tps=baseline,
            candidate_tps=values, candidate_mean_tps=statistics.mean(values),
            ratio=statistics.mean(values) / baseline,
            candidate_replies_identical=left['reply'] == right['reply'],
            candidate_tokens=[r['usage']['completion_tokens'] for r in (left, right)],
            baseline_acceptance=statistics.mean(r['acceptance_fraction'] for r in old['baseline_samples']),
            candidate_acceptance=statistics.mean(r['acceptance_fraction'] for r in (left, right)),
            baseline_tokens_per_step=statistics.mean(r['tokens_per_step'] for r in old['baseline_samples']),
            candidate_tokens_per_step=statistics.mean(r['tokens_per_step'] for r in (left, right)),
            baseline_step_ms=statistics.mean(r['wall_ms_per_step'] for r in old['baseline_samples']),
            candidate_step_ms=statistics.mean(r['wall_ms_per_step'] for r in (left, right))))
    c6 = []
    requests = lambda w: [(r['label'], r['request']) for r in w['requests']]
    baseline = concurrent['waves'][0]['aggregate_end_to_end_tps']
    for trial in (first, second, third):
        wave = trial['waves'][0]
        assert requests(wave) == requests(concurrent['waves'][0])
        c6.append(dict(tps=wave['aggregate_end_to_end_tps'],
            ratio_to_saved_warm_mean=wave['aggregate_end_to_end_tps'] / baseline,
            acceptance=wave['acceptance_fraction'], completion_tokens=wave['completion_tokens']))
    result = dict(status='matched_saved_baseline_comparison', new_baseline_measured=False,
        serial=rows, median_serial_ratio=statistics.median(r['ratio'] for r in rows),
        baseline_c6_all=concurrent['all_saved_waves'], baseline_c6_warm_mean=baseline,
        candidate_c6_all=c6, candidate_c6_warm_mean=statistics.mean(r['tps'] for r in c6[1:]),
        candidate_c6_warm_ratio=statistics.mean(r['tps'] for r in c6[1:]) / baseline,
        source_sha256={name: hashlib.sha256((a.reports / name).read_bytes()).hexdigest() for name in names},
        qualification='Four prompts, one seed, two serial repeats, three C6 waves; reused historical baseline. '
                      'Changed stochastic continuations can change routing and acceptance. No quality or long-prefill claim.')
    with (a.reports / 'comparison.json').open('x') as out:
        json.dump(result, out, indent=2); out.write('\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
