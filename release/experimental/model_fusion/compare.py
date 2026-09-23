# SPDX-License-Identifier: AGPL-3.0-only
"""Compare matched serving evidence, keeping throughput separate from latency."""
import argparse
import json
from pathlib import Path
import statistics


def compare(root, baseline, candidate):
    def read(label, suffix):
        return json.loads((root / (label + suffix + '.json')).read_bytes())
    before = read(baseline, '-serial')
    after = read(candidate, '-serial')
    if before['status'] != 'complete' or after['status'] != 'complete':
        raise ValueError('Both serial trials must be complete')
    left, right = before['deployment'], after['deployment']
    if left['serving'] != right['serving'] or left['model_manifest_sha256'] != right['model_manifest_sha256']:
        raise ValueError('Serving settings or weights changed')
    for a, b in zip(left['nodes'], right['nodes'], strict=True):
        for field in ('image','model','model_bindings','draft','cache','rails'):
            if a.get(field) != b.get(field):
                raise ValueError('Node field changed: ' + field)
    rows = []
    for a, b in zip(before['cases'], after['cases'], strict=True):
        if a['request'] != b['request'] or a['label'] != b['label']:
            raise ValueError('Unmatched requests')
        rows.append(dict(label=a['label'], temperature=a['request']['temperature'], seed=a['request']['seed'],
            identical_reply=a['reply']==b['reply'], identical_usage=a['usage']==b['usage'],
            decode_tps_before=a['timing']['decode_tokens_per_second'], decode_tps_after=b['timing']['decode_tokens_per_second'],
            decode_tps_ratio=b['timing']['decode_tokens_per_second']/a['timing']['decode_tokens_per_second'],
            step_ms_before=a['wall_ms_per_step'], step_ms_after=b['wall_ms_per_step'],
            step_time_ratio=b['wall_ms_per_step']/a['wall_ms_per_step'],
            acceptance_before=a['acceptance_fraction'], acceptance_after=b['acceptance_fraction']))
    concurrency = []
    c_before, c_after = read(baseline,'-c6'), read(candidate,'-c6')
    if c_before['status']!='complete' or c_after['status']!='complete':
        raise ValueError('Both concurrency trials must be complete')
    for a,b in zip(c_before['waves'], c_after['waves'], strict=True):
        if a['temperature'] != b['temperature'] or a['completion_tokens'] != b['completion_tokens']:
            raise ValueError('Unmatched concurrency waves')
        concurrency.append(dict(temperature=a['temperature'],
            aggregate_tps_before=a['aggregate_end_to_end_tps'], aggregate_tps_after=b['aggregate_end_to_end_tps'],
            aggregate_tps_ratio=b['aggregate_end_to_end_tps']/a['aggregate_end_to_end_tps'],
            acceptance_before=a['acceptance_fraction'], acceptance_after=b['acceptance_fraction']))
    a,b = read(baseline,'-prefill'), read(candidate,'-prefill')
    if (any(r['status']!='uncached_prefill_measured' or r['prefix_hits_delta']!=0 for r in (a,b))
            or a['prompt_tokens'] != b['prompt_tokens']):
        raise ValueError('Unmatched uncached prefill')
    return dict(status='matched_comparison', baseline=baseline, candidate=candidate, serial=rows,
        serial_median_decode_tps_ratio=statistics.median(r['decode_tps_ratio'] for r in rows),
        serial_median_step_time_ratio=statistics.median(r['step_time_ratio'] for r in rows),
        identical_replies=sum(r['identical_reply'] for r in rows), c6=concurrency,
        prefill=dict(prompt_tokens=a['prompt_tokens'], tokens_per_second_before=a['prefill_tokens_per_second'],
            tokens_per_second_after=b['prefill_tokens_per_second'],
            tps_ratio=b['prefill_tokens_per_second']/a['prefill_tokens_per_second']),
        qualification='Single matched trial pair; small gains require a repeated control. Not a quality benchmark.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports',type=Path,required=True)
    parser.add_argument('--baseline',default='baseline')
    parser.add_argument('--candidate',default='candidate')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve earlier comparisons')
    result=compare(args.reports,args.baseline,args.candidate)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='serial'},indent=2))


if __name__=='__main__':main()
