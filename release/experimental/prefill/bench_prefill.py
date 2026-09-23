# SPDX-License-Identifier: AGPL-3.0-only
"""Repeated uncached prefill timings on one pinned worker pair.

Each request starts with a unique nonce, so prefix caching cannot hit; the
three-key retrieval answer must be correct. Timing comes from isolated native
per-request counters (exactly one request delta, idle before and after).
Requests are admitted only with the independent RAM watchdog present.
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

FILLER = 'Background note: the stone path crosses the quiet garden.\n'
CODES = ('PLUM-4927', 'RIVER-8261', 'GLASS-1359')


def call(path, payload, timeout=900):
    req = urllib.request.Request(speed.BASE + path, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def build(target, nonce):
    def content(lines):
        left, middle = lines // 2, lines * 9 // 10 - lines // 2
        right = lines - left - middle
        return (f'Session {nonce}. Three record keys occur in these notes.\n'
                f'The first key is {CODES[0]}.\n' + FILLER * left +
                f'The second key is {CODES[1]}.\n' + FILLER * middle +
                f'The third key is {CODES[2]}.\n' + FILLER * right +
                'List the first, second, and third keys in that order. '
                'Reply only with the three keys separated by " | ".')

    def payload(lines):
        return dict(model=speed.MODEL, messages=[dict(role='user', content=content(lines))],
                    max_tokens=24, temperature=0, chat_template_kwargs={'thinking': False})

    base = call('/tokenize', payload(0))['count']
    per = (call('/tokenize', payload(10))['count'] - base) / 10
    lines = int((target - base) / per)
    for _ in range(4):
        count = call('/tokenize', payload(lines))['count']
        if target - 64 <= count <= target:
            return payload(lines), count
        lines += int((target - count) / per)
    raise ValueError('Could not size the prompt')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--targets', type=int, nargs='+', default=[8192, 32768])
    p.add_argument('--repeats', type=int, default=2)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Preserve previous evidence')
    config = json.loads(a.deployment.read_bytes())
    kit = Path(config['nodes'][0]['kit'])
    sys.path.insert(0, str(kit / 'tools'))
    import portable_pair as pair
    run = Path(config['nodes'][0]['runs']) / config['run_id'] / 'pair'
    ready = json.loads((run / 'health-ready.json').read_bytes())
    speed.BASE = 'http://127.0.0.1:' + str(config['api']['port'])
    report = dict(status='running', run_id=config['run_id'], kit_sha256=config['kit_manifest_sha256'],
                  serving=config['serving'], cases=[])

    def hosts():
        rows = pair.results(pair.both(config, 'inspect'))
        reason = pair.stop_reason(rows, ready['containers'], ready['started_at'])
        if reason:
            raise RuntimeError(reason)
        if any(r['memory']['MemAvailable'] < 2**30 for r in rows):
            raise RuntimeError('Request admission margin unavailable')
        return [r['memory']['MemAvailable'] for r in rows]

    def hits(text):
        return sum(float(l.rsplit(' ', 1)[1]) for l in text.splitlines()
                   if l.startswith('vllm:prefix_cache_hits_total{'))

    with (run / 'request-probe.lock').open('a') as lock, (run / 'watch.lock').open('r') as watch:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            fcntl.flock(watch, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise RuntimeError('Independent RAM watchdog missing')
        rng = random.Random()
        try:
            for target in a.targets:
                for i in range(a.repeats):
                    payload, count = build(target, f'{rng.getrandbits(64):016x}')
                    before_mem = hosts()
                    raw, before = speed.get_metrics(); speed.assert_idle(raw)
                    prior = hits(raw)
                    result = call('/v1/chat/completions', payload)
                    for _ in range(20):
                        raw, after = speed.get_metrics()
                        if after['request_generation_tokens_count'] != before['request_generation_tokens_count']:
                            break
                        time.sleep(.25)
                    speed.assert_idle(raw)
                    answer = result['choices'][0]['message']['content']
                    timing = speed.isolated_timing(before, after, result['usage']['completion_tokens'])
                    row = dict(target=target, prompt_tokens=result['usage']['prompt_tokens'],
                               prefix_hits_delta=hits(raw) - prior,
                               retrieval_pass=all(c in answer for c in CODES), answer=answer,
                               server_prefill_seconds=timing['server_prefill_seconds'],
                               prefill_tokens_per_second=result['usage']['prompt_tokens'] / timing['server_prefill_seconds'],
                               mem_before=before_mem, mem_after=hosts())
                    if row['prefix_hits_delta'] or not row['retrieval_pass']:
                        raise ValueError('Cached or failed retrieval: ' + json.dumps(row))
                    report['cases'].append(row)
                    print(json.dumps({k: row[k] for k in ('target', 'prompt_tokens', 'prefill_tokens_per_second')}), flush=True)
            report['status'] = 'complete'
        except BaseException as error:
            report.update(status='failed', error=repr(error))
            raise
        finally:
            a.output.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
