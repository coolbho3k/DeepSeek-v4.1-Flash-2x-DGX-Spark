# SPDX-License-Identifier: AGPL-3.0-only
"""Build benchmark inputs from completed probabilistic-drafting reports only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert a.output.is_dir()
    files = ['candidate.json', 'candidate-repeat.json', 'candidate-c6-repeat.json']
    trials = [json.loads((a.reports / name).read_bytes()) for name in files]
    assert all(t['status'] == 'complete' for t in trials)
    assert len({t['kit_sha256'] for t in trials}) == 1
    deployment = Path(trials[0]['deployment'])
    config = json.loads(deployment.read_bytes())
    provenance = dict(new_baseline_measured=False,
        source_sha256={str(a.reports / name): hashlib.sha256((a.reports / name).read_bytes()).hexdigest()
                       for name in files},
        baseline_kit_sha256=trials[0]['kit_sha256'],
        deployment_sha256=hashlib.sha256(deployment.read_bytes()).hexdigest(),
        serial_aggregation='Arithmetic mean of the two saved per-case decode throughputs',
        c6_aggregation='Arithmetic mean of the two saved warm waves; retain first-use wave separately')
    serial = dict(status='complete', deployment=config, cases=[], provenance=provenance)
    assert [r['label'] for r in trials[0]['cases']] == [r['label'] for r in trials[1]['cases']]
    for left, right in zip(trials[0]['cases'], trials[1]['cases'], strict=True):
        assert left['request'] == right['request']
        row = copy.deepcopy(left)
        row['timing'] = dict(decode_tokens_per_second=statistics.mean(
            r['timing']['decode_tokens_per_second'] for r in (left, right)))
        row['baseline_samples'] = [{k: r[k] for k in
            ('timing', 'acceptance_fraction', 'tokens_per_step', 'wall_ms_per_step')}
            for r in (left, right)]
        serial['cases'].append(row)
    wave = copy.deepcopy(trials[1]['waves'][0])
    waves = [t['waves'][0] for t in trials]
    requests = lambda w: [(r['label'], r['request']) for r in w['requests']]
    assert all(requests(w) == requests(wave) for w in waves)
    wave['aggregate_end_to_end_tps'] = statistics.mean(w['aggregate_end_to_end_tps'] for w in waves[1:])
    concurrent = dict(status='complete', deployment=config, waves=[wave],
        all_saved_waves=[{k: w[k] for k in ('aggregate_end_to_end_tps', 'acceptance_fraction')}
                         for w in waves], provenance=provenance)
    for name, value in [('baseline-serial.json', serial), ('baseline-c6.json', concurrent)]:
        with (a.output / name).open('x') as output:
            json.dump(value, output, indent=2); output.write('\n')
    print(json.dumps(provenance), flush=True)


if __name__ == '__main__':
    main()
