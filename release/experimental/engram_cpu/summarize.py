# SPDX-License-Identifier: AGPL-3.0-only
"""Summarize completed CPU-only reports; no serving calls or external packages."""
import argparse
import json
from pathlib import Path
import statistics


def read_report(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if not records or records[-1].get('kind') != 'complete':
        raise ValueError(f'{path}: incomplete benchmark')
    metadata = [r for r in records if r.get('kind') == 'metadata']
    checks = [r for r in records if r.get('kind') == 'correctness']
    if len(metadata) != 1 or len(checks) != 1 or not checks[0]['exact_bytes']:
        raise ValueError(f'{path}: missing correctness/metadata')
    timings = {}
    for row in records:
        if row.get('kind') != 'timing':
            continue
        key = row['rows'], row['pattern'], row['variant']
        if key in timings or len(row['samples_us']) != row['repeats']:
            raise ValueError(f'{path}: repeated/incomplete timing')
        if any(x <= 0 for x in row['samples_us']):
            raise ValueError(f'{path}: invalid timing')
        timings[key] = row
    if len(timings) not in (84, 168):
        raise ValueError(f'{path}: unexpected timing matrix size')
    return metadata[0], checks[0], records[-1], timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', type=Path, nargs='+')
    parser.add_argument('--all-shapes', action='store_true')
    args = parser.parse_args()
    for path in args.reports:
        meta, checks, complete, timings = read_report(path)
        print(f'\n## {path.name}\n')
        print(f"CPU {meta['cpu']}, SVE {meta['sve_bytes'] * 8} bits; "
              f"{checks['cases']} exact-byte cases passed; "
              f"peak RSS {complete['max_rss_kib']/1024:.1f} MiB.\n")
        print('| Rows | Pattern | Baseline µs | Direct + batched stats µs | '
              'Grouped scalar µs | Grouped SVE µs | SVE/grouped scalar throughput |')
        print('| ---: | --- | ---: | ---: | ---: | ---: | ---: |')
        shapes = sorted({key[0] for key in timings}) if args.all_shapes else (12, 48, 288, 3072)
        for n in shapes:
            for pattern in ('hot', 'rotating', 'masked'):
                if (n, pattern, 'baseline') not in timings:
                    continue
                selected = [timings[n, pattern, name] for name in
                            ('baseline', 'direct_scalar_stats', 'grouped_scalar', 'grouped_sve')]
                times = ' | '.join(f"{r['median_us']:.3f}" for r in selected)
                ratios = [a / b for a, b in zip(selected[2]['samples_us'], selected[3]['samples_us'])]
                print(f'| {n} | {pattern} | {times} | {statistics.median(ratios):.3f}× |')


if __name__ == '__main__':
    main()
