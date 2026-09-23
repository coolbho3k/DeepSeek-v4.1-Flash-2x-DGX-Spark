# SPDX-License-Identifier: AGPL-3.0-only
"""Serial decode benchmark with copy-heavy and free-prose prompts.

Isolated native per-request counters (exactly one request delta, idle before
and after), admitted only with the independent RAM watchdog present. Each
case runs at temperature 0 and at temperature 1 (top-p 0.95, seed 41).
"""
import argparse
import fcntl
import json
from pathlib import Path
import random
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'probes'))
import check_serving_speed as speed


def code_file():
    rng = random.Random(3)
    lines = ['import math', 'from dataclasses import dataclass', '', '']
    for i in range(12):
        lines += [f'def compute_metric_{i}(values, scale={i + 1}):',
                  f'    """Return the scaled metric number {i} for a list of values."""',
                  '    total = 0.0',
                  '    for value in values:',
                  f'        total += math.sqrt(abs(value)) * scale + {rng.randint(1, 9)}',
                  '    return total / max(len(values), 1)', '', '']
    lines += ['def summarize(values):', '    return [compute_metric_3(values), compute_metric_7(values)]']
    return '\n'.join(lines)


def typo_text():
    rng = random.Random(5)
    base = ('The committee reviewed the annual budget and discussed several proposals for improving the '
            'public library, including longer opening hours, a larger children\'s section, and better '
            'access for people with disabilities. ') * 6
    words = base.split()
    for i in range(0, len(words), 7):
        w = words[i]
        if len(w) > 4:
            k = rng.randrange(1, len(w) - 2)
            words[i] = w[:k] + w[k + 1] + w[k] + w[k + 2:]
    return ' '.join(words)


def json_records():
    rng = random.Random(9)
    rows = [dict(id=i, name=f'item-{i}', price=round(rng.uniform(1, 90), 2), stock=rng.randint(0, 50))
            for i in range(24)]
    return json.dumps(rows, indent=1)


PASSAGE = '\n\n'.join(
    f'Paragraph {i}: The river town of Alder Bend grew around a mill built in {1790 + i * 7}. '
    f'Its residents traded timber, wool and grain, and every spring the flood reshaped the lower streets. '
    f'In year {1800 + i * 9} a new bridge connected the eastern farms to the market square.'
    for i in range(14))

CASES = {
    'code_edit': 'Rename the function compute_metric_3 to scaled_root_mean everywhere it appears in this '
                 'file (including calls). Output the complete updated file only.\n\n' + code_file(),
    'typo_fix': 'Fix the spelling mistakes in the following text and return the full corrected text only.\n\n'
                + typo_text(),
    'json_rename': 'Return the same JSON with the field "price" renamed to "cost". Output only JSON.\n\n'
                   + json_records(),
    'quote_explain': 'Quote Paragraph 6 verbatim, then explain in two sentences what it says.\n\n' + PASSAGE,
    'garden': 'Write a detailed description of a quiet garden in the morning. Use complete sentences and '
              'include sensory details. Write at least 800 words.',
    'python': 'Write a complete Python implementation of an LRU cache with get and put methods, type hints, '
              'and unit tests. Explain its invariants and time complexity.',
    'explanation': 'Explain how a relational database executes a SQL query, from parsing through planning, '
                   'indexes, joins, execution, and result delivery. Use a concrete example throughout.',
    'easy_prose': 'Explain photosynthesis to a curious ten-year-old. Use plain complete sentences and familiar '
                  'examples. Write at least 800 words.',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Preserve previous evidence')
    config = json.loads(a.deployment.read_bytes())
    sys.path.insert(0, str(Path(config['nodes'][0]['kit']) / 'tools'))
    import portable_pair as pair
    run = Path(config['nodes'][0]['runs']) / config['run_id'] / 'pair'
    ready = json.loads((run / 'health-ready.json').read_bytes())
    speed.BASE = 'http://127.0.0.1:' + str(config['api']['port'])
    report = dict(status='running', kit=config['kit_manifest_sha256'], cases=[])

    def hosts():
        rows = pair.results(pair.both(config, 'inspect'))
        reason = pair.stop_reason(rows, ready['containers'], ready['started_at'])
        if reason:
            raise RuntimeError(reason)
        if any(r['memory']['MemAvailable'] < 2**30 for r in rows):
            raise RuntimeError('Request admission margin unavailable')

    with (run / 'request-probe.lock').open('a') as lock, (run / 'watch.lock').open('r') as watch:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            fcntl.flock(watch, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise RuntimeError('Independent RAM watchdog missing')
        try:
            for temperature in (0.0, 1.0):
                for label, prompt in CASES.items():
                    hosts()
                    raw, before = speed.get_metrics(); speed.assert_idle(raw)
                    spec_before = speed.speculative_snapshot(raw)
                    body = dict(model=speed.MODEL, messages=[dict(role='user', content=prompt)], max_tokens=400,
                                temperature=temperature, top_p=1.0 if temperature == 0 else 0.95, seed=41,
                                chat_template_kwargs={'thinking': False})
                    req = urllib.request.Request(speed.BASE + '/v1/chat/completions', data=json.dumps(body).encode(),
                                                 headers={'Content-Type': 'application/json'})
                    with urllib.request.urlopen(req, timeout=900) as r:
                        result = json.load(r)
                    for _ in range(20):
                        raw, after = speed.get_metrics()
                        if after['request_generation_tokens_count'] != before['request_generation_tokens_count']:
                            break
                        time.sleep(.25)
                    speed.assert_idle(raw)
                    timing = speed.isolated_timing(before, after, result['usage']['completion_tokens'])
                    spec_after = speed.speculative_snapshot(raw)
                    delta = {k: spec_after[k] - v for k, v in spec_before.items()}
                    count = lambda n: sum(v for k, v in delta.items() if k.startswith('vllm:' + n + '{'))
                    drafts = count('spec_decode_num_drafts_total')
                    row = dict(label=label, temperature=temperature, tokens=result['usage']['completion_tokens'],
                               decode_tps=timing['decode_tokens_per_second'],
                               acceptance=count('spec_decode_num_accepted_tokens_total') / max(count('spec_decode_num_draft_tokens_total'), 1),
                               tokens_per_step=timing['decode_tokens'] / max(drafts, 1),
                               reply=result['choices'][0]['message']['content'])
                    report['cases'].append(row)
                    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != 'reply'}), flush=True)
            report['status'] = 'complete'
        finally:
            a.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')


if __name__ == '__main__':
    main()
