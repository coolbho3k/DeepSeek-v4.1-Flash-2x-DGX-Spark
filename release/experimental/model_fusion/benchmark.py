# SPDX-License-Identifier: AGPL-3.0-only
"""Run matched serving benchmarks against one explicitly pinned worker pair."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', type=Path, required=True)
    parser.add_argument('--deployment-sha256', required=True)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--label', choices=('candidate', 'control', 'final'), required=True)
    parser.add_argument('--profiles', action='store_true')
    args = parser.parse_args()
    raw = args.deployment.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.deployment_sha256:
        raise ValueError('Changed deployment')
    config = json.loads(raw)
    kit = Path(config['nodes'][0]['kit'])
    sys.path.insert(0, str(kit / 'tools'))
    import portable_pair as pair
    sys.path.insert(0, str(ROOT / 'probes'))
    import check_serving_speed as speed
    import check_serving_long_context as long
    reports = args.reports.absolute()
    label = args.label
    outputs = [reports / (label + suffix) for suffix in
               ('-serial.json', '-c6.json', '-prefill.json', '-prefill-raw.json')]
    outputs[3] = ROOT / 'reports' / f'model-fusion-{label}-prefill-v1.json'
    if any(p.exists() for p in outputs):
        raise ValueError('Preserve existing benchmark evidence')
    run = Path(config['nodes'][0]['runs']) / config['run_id'] / 'pair'
    identities = None
    deadline = time.monotonic() + 1200
    while True:
        observed = pair.results(pair.both(config, 'inspect'))
        current = [(r['container'], r['state']['StartedAt']) for r in observed]
        if identities is None:
            identities = current
        if current != identities or any(not r['state']['Running'] for r in observed):
            raise RuntimeError('Recorded workers stopped or changed')
        if (run / 'health-ready.json').exists():
            ready = json.loads((run / 'health-ready.json').read_bytes())
            reason = pair.stop_reason(observed, ready['containers'], ready['started_at'])
            if reason:
                raise RuntimeError(reason)
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('Readiness observation timed out; no restart attempted')
        print(json.dumps(dict(stage=label + '_loading', available=[r['memory']['MemAvailable'] for r in observed])), flush=True)
        time.sleep(15)
    def command(script, *arguments):
        subprocess.run([sys.executable, '-u', '-B', str(ROOT / script), *map(str, arguments)], check=True, cwd=ROOT)
    command('probes/benchmark_moex_campaign.py', '--deployment', args.deployment, '--output', outputs[0])
    if args.profiles:
        for n in (1, 6):
            command('release/experimental/model_fusion/profile.py', '--deployment', args.deployment,
                    '--deployment-sha256', args.deployment_sha256, '--concurrency', n,
                    '--output', reports / f'{label}-profile-c{n}.json')
    command('probes/benchmark_dspark_concurrency.py', '--deployment', args.deployment, '--output', outputs[1])
    speed.BASE = long.BASE = 'http://127.0.0.1:' + str(config['api']['port'])
    with (run / 'request-probe.lock').open('a') as lock, (run / 'watch.lock').open('r') as watch:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            fcntl.flock(watch, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise RuntimeError('Missing independent memory watchdog')
        def hosts():
            rows = pair.results(pair.both(config, 'inspect'))
            reason = pair.stop_reason(rows, ready['containers'], ready['started_at'])
            if reason:
                raise RuntimeError(reason)
            if any(r['memory']['MemAvailable'] < 2**30 for r in rows):
                raise RuntimeError('Request admission margin unavailable')
            return rows
        before_hosts = hosts()
        raw, before = speed.get_metrics(); speed.assert_idle(raw)
        def hits(text):
            return sum(float(line.rsplit(' ', 1)[1]) for line in text.splitlines()
                       if line.startswith('vllm:prefix_cache_hits_total{'))
        prior_hits = hits(raw)
        sys.argv = ['check_serving_long_context.py', '--target-tokens', '32768', '--output', str(outputs[3])]
        long.main()
        for _ in range(20):
            raw, after = speed.get_metrics()
            if after['request_generation_tokens_count'] != before['request_generation_tokens_count']:
                break
            time.sleep(.25)
        speed.assert_idle(raw)
        result = json.loads(outputs[3].read_bytes())
        if result['status'] != 'retrieval_pass' or hits(raw) != prior_hits:
            raise ValueError('Failed retrieval or unexpectedly cached prefill')
        timing = speed.isolated_timing(before, after, result['usage']['completion_tokens'])
        measured = dict(status='uncached_prefill_measured', run_id=config['run_id'],
                        prompt_tokens=result['usage']['prompt_tokens'], timing=timing,
                        prefill_tokens_per_second=result['usage']['prompt_tokens'] / timing['server_prefill_seconds'],
                        prefix_hits_delta=0, report=str(outputs[3]), before_hosts=before_hosts, after_hosts=hosts())
        outputs[2].write_text(json.dumps(measured, indent=2) + '\n')
        print(json.dumps({k:v for k,v in measured.items() if k not in ('before_hosts','after_hosts')}), flush=True)
    print(json.dumps(dict(status='matched_suite_complete', label=label, run_id=config['run_id'])), flush=True)


if __name__ == '__main__':
    main()
