# SPDX-License-Identifier: AGPL-3.0-only
"""Capture one native GPU step on an identified, isolated serving pair.

The existing controller continues RAM/identity monitoring. Profiling requires
3 GiB available per host and the native one-step, two-iteration-delay settings.
No restart, model mutation, tensor copy or throughput claim is made here.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'probes'))
import check_serving_speed as speed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', type=Path, required=True)
    parser.add_argument('--deployment-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, choices=(1, 6), default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing profile evidence')
    raw = args.deployment.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.deployment_sha256:
        raise ValueError('Changed deployment')
    config = json.loads(raw)
    if config['api']['port'] != 8889:
        raise ValueError('Use the isolated experiment port')
    kit = Path(config['nodes'][0]['kit'])
    manifest_raw = (kit / 'bundle-manifest.json').read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != config['kit_manifest_sha256']:
        raise ValueError('Changed runtime kit')
    manifest = json.loads(manifest_raw)
    sys.path.insert(0, str(kit / 'tools'))
    import portable_pair as pair
    for path in (kit / 'tools/portable_pair.py', kit / 'tools/portable_node.py'):
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['files'][str(path.relative_to(kit))]['sha256']:
            raise ValueError('Changed controller source')
    pair.node.validate_config(config)
    settings = json.loads(pair.node.PROFILE['profiler-config'])
    expected = dict(profiler='torch', max_iterations=1, delay_iterations=2,
                    warmup_iterations=0, ignore_frontend=True,
                    torch_profiler_with_stack=False, torch_profiler_record_shapes=False,
                    torch_profiler_with_memory=False, torch_profiler_with_flops=False,
                    capture_torch_profiler=False, detailed_trace_annotation=False)
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError('Require bounded native profiler settings')
    run = Path(config['nodes'][0]['runs']) / config['run_id'] / 'pair'
    ready = json.loads((run / 'health-ready.json').read_bytes())
    owner = json.loads((run / 'controller.json').read_bytes())
    if (owner['config_sha256'] != pair.node.sha(pair.node.encoded(config)) or
            ready['containers'] != owner['containers'] or
            ready['status'] != 'portable_pair_api_ready_ram_watch_continues'):
        raise ValueError('Exact pair readiness and ownership required')
    speed.BASE = 'http://127.0.0.1:8889'
    report = dict(status='running', run_id=config['run_id'],
                  deployment_sha256=args.deployment_sha256,
                  kit_sha256=config['kit_manifest_sha256'], profiler_settings=settings,
                  containers=ready['containers'], concurrency=args.concurrency,
                  throughput_measurement_valid=False,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    def inspect():
        observed = pair.results(pair.both(config, 'inspect'))
        reason = pair.stop_reason(observed, ready['containers'], ready['started_at'])
        if reason:
            raise RuntimeError(reason)
        return observed

    def control(action):
        request = urllib.request.Request(speed.BASE + '/' + action + '_profile', data=b'', method='POST')
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 200:
                raise RuntimeError('Native profiling control failed')

    def request(index):
        payload = dict(model=speed.MODEL,
                       messages=[dict(role='user', content=f'Exercise {index}: explain how a compiler optimizes a loop. Give a detailed explanation.')],
                       max_tokens=96, temperature=0, seed=41,
                       chat_template_kwargs={'thinking': False})
        query = urllib.request.Request(speed.BASE + '/v1/chat/completions',
                                       data=json.dumps(payload).encode(),
                                       headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(query, timeout=180) as response:
            result = json.load(response)
        if not result.get('choices') or result['usage']['completion_tokens'] < 16:
            raise ValueError('Profile request did not exercise multiple steps')
        return dict(usage=result['usage'], finish_reason=result['choices'][0]['finish_reason'])

    with (run / 'request-probe.lock').open('a') as lock, (run / 'watch.lock').open('r') as watch:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            fcntl.flock(watch, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise RuntimeError('Independent RAM watchdog missing')
        report['initial_hosts'] = inspect()
        if any(row['memory']['MemAvailable'] < 3 * 2**30 for row in report['initial_hosts']):
            raise RuntimeError('Preserve the 3 GiB profiler admission floor')
        text, _ = speed.get_metrics()
        speed.assert_idle(text)
        save()
        armed = False
        try:
            armed = True
            control('start')
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                report['requests'] = list(pool.map(request, range(args.concurrency)))
            control('stop')
            armed = False
            report['final_hosts'] = inspect()
            text, _ = speed.get_metrics()
            speed.assert_idle(text)
            report['status'] = 'profile_requests_complete_inspect_traces'
        except BaseException as error:
            report.update(status='failed', error=repr(error))
            raise
        finally:
            if armed:
                control('stop')
            save()
    print(json.dumps({key: report[key] for key in ('status', 'run_id', 'concurrency', 'profiler_settings')}), flush=True)


if __name__ == '__main__':
    main()
