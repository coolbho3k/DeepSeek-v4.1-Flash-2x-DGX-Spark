# SPDX-License-Identifier: AGPL-3.0-only
"""Summarize native GPU traces; overlapping durations are never step latency."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path


def summarize(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        trace = json.load(stream)
    kernels = [event for event in trace['traceEvents']
               if event.get('cat') == 'kernel' and event.get('ph') == 'X']
    if not kernels:
        raise ValueError('No native GPU kernels in ' + str(path))
    groups = defaultdict(lambda: [0, 0.])
    for event in kernels:
        entry = groups[event['name']]
        entry[0] += 1
        entry[1] += event['dur']
    intervals = sorted((event['ts'], event['ts'] + event['dur']) for event in kernels)
    begin, end = intervals[0]
    union = 0.
    for left, right in intervals[1:]:
        if left > end:
            union += end - begin
            begin, end = left, right
        else:
            end = max(end, right)
    union += end - begin
    ranked = sorted(groups.items(), key=lambda item: item[1][1], reverse=True)
    return dict(path=str(path), kernels=len(kernels),
                kernel_span_ms=(max(right for _, right in intervals) - intervals[0][0]) / 1000,
                any_kernel_active_ms=union / 1000,
                summed_kernel_ms=sum(event['dur'] for event in kernels) / 1000,
                timing_scope='One rank; duration sums overlap and NCCL may include peer waiting.',
                top_kernels=[dict(name=name, count=value[0], total_ms=value[1] / 1000,
                                  mean_us=value[1] / value[0]) for name, value in ranked[:60]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier trace summaries')
    reports = [summarize(path) for path in args.traces]
    args.output.write_text(json.dumps(dict(status='native_traces_summarized', traces=reports), indent=2) + '\n')
    for report in reports:
        print(json.dumps({key: value for key, value in report.items() if key != 'top_kernels'}))
        for kernel in report['top_kernels'][:12]:
            print(json.dumps(dict(kernel, name=kernel['name'][:180])))


if __name__ == '__main__':
    main()
