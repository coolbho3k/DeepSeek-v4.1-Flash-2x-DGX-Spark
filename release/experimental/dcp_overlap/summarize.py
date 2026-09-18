# SPDX-License-Identifier: AGPL-3.0-only
"""Summarize paired component logs and actual CUDA-kernel overlap traces."""
import argparse
import json
from pathlib import Path
import re
import statistics


def receipt(path):
    text = path.read_text()
    for line in reversed(text.splitlines()):
        if '{' not in line:
            continue
        try:
            row = json.loads(line[line.index('{'):])
        except json.JSONDecodeError:
            continue
        if row.get('status') == 'component_pass':
            return row
    # Docker splits long stdout records into 16-KiB timestamped fragments.
    text = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z ', '', text)
    begin = text.rfind('{"status": "component_pass"')
    if begin >= 0:
        return json.JSONDecoder().raw_decode(text[begin:].replace('\n', ''))[0]
    raise ValueError('Missing completed component receipt: ' + str(path))


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def trace(path):
    events = json.loads(path.read_bytes())['traceEvents']
    spans = [e for e in events if e.get('cat') == 'user_annotation' and e.get('name') in ('baseline', 'overlap')]
    result = {}
    for span in spans:
        kernels = [e for e in events if e.get('cat') == 'kernel'
                   and span['ts'] <= e['ts'] < span['ts'] + span['dur']]
        comm = merged([(e['ts'], e['ts'] + e['dur']) for e in kernels if 'nccl' in e['name'].lower()])
        compute = merged([(e['ts'], e['ts'] + e['dur']) for e in kernels if 'nccl' not in e['name'].lower()])
        intersection = sum(max(0, min(b, d) - max(a, c)) for a, b in comm for c, d in compute)
        result[span['name']] = dict(kernel_count=len(kernels),
            streams=sorted({str(e.get('args', {}).get('stream', e.get('tid'))) for e in kernels}),
            nccl_busy_us=sum(b - a for a, b in comm),
            non_nccl_busy_us=sum(b - a for a, b in compute),
            actual_concurrent_kernel_us=intersection,
            kernel_span_us=(max(e['ts'] + e['dur'] for e in kernels) - min(e['ts'] for e in kernels)) if kernels else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', type=Path)
    args = parser.parse_args()
    ranks = [receipt(args.reports / f'host{i}.log') for i in (0, 1)]
    rows = []
    for a, b in zip(ranks[0]['cases'], ranks[1]['cases'], strict=True):
        key = ('rows', 'decode', 'image_width', 'swa_only')
        if any(a[k] != b[k] for k in key):
            raise ValueError('Rank cases differ')
        item = {k: a[k] for k in key}
        if a.get('topk') != b.get('topk'):
            raise ValueError('Rank top-k differs')
        item['topk'] = a.get('topk', 1024)
        if a['timing']:
            for mode, metric in (('graph', 'gpu_samples_ms'), ('eager', 'wall_samples_ms')):
                data = [r['timing'] if mode == 'graph' else r['timing'].get('eager') for r in (a, b)]
                if any(r is None or metric not in r['baseline'] for r in data):
                    continue
                timing = {}
                for label in ('baseline', 'overlap'):
                    # Corresponding rounds on both ranks: use the slower
                    # rank per sample, not whichever rank looks favorable.
                    samples = [max(x, y) for x, y in zip(data[0][label][metric], data[1][label][metric], strict=True)]
                    timing[label + '_ms'] = statistics.median(samples)
                timing['speedup'] = timing['baseline_ms'] / timing['overlap_ms']
                item[mode] = {k: round(v, 5) for k, v in timing.items()}
        rows.append(item)
    traces = {str(p.relative_to(args.reports)): trace(p) for p in args.reports.glob('host*/trace-*.json')}
    result = dict(mode=ranks[0]['mode'], both_ranks_exact=True,
                  peak_component_tensor_bytes=[r['peak_allocated_bytes'] for r in ranks],
                  cases=rows, traces=traces, full_model_qualified=False)
    print(json.dumps(result, indent=2))
    with (args.reports / 'summary.json').open('x') as stream:
        json.dump(result, stream, indent=2)


if __name__ == '__main__':
    main()
