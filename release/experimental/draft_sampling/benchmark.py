# SPDX-License-Identifier: AGPL-3.0-only
"""Replay saved temperature-one requests with no new baseline measurements.

Uses the maintainer's local probe library, as does the model-fusion campaign.
The immutable deployment supplies the sampling mode. This driver never changes
or restarts it. Request counters must isolate each serial request and C6 wave.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import statistics
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'probes'))
import check_serving_speed as speed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment', type=Path, required=True)
    p.add_argument('--baseline-serial', type=Path, required=True)
    p.add_argument('--baseline-c6', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--concurrency-only', action='store_true')
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Preserve previous evidence')
    config = json.loads(a.deployment.read_bytes())
    kit = Path(config['nodes'][0]['kit'])
    sys.path.insert(0, str(ROOT / 'release/runtime'))
    from verify import verify
    verify(kit, config['kit_manifest_sha256'])
    sys.path.insert(0, str(kit / 'tools'))
    import portable_pair as pair
    serial = json.loads(a.baseline_serial.read_bytes())
    concurrent = json.loads(a.baseline_c6.read_bytes())
    assert serial['status'] == concurrent['status'] == 'complete'
    assert config['serving'] == serial['deployment']['serving']
    assert config['model_manifest_sha256'] == serial['deployment']['model_manifest_sha256']
    cases = [r for r in serial['cases'] if r['request']['temperature'] == 1.0]
    waves = [r for r in concurrent['waves'] if r['temperature'] == 1.0]
    assert len(cases) == 4 and len(waves) == 1 and len(waves[0]['requests']) == 6
    run = Path(config['nodes'][0]['runs']) / config['run_id'] / 'pair'
    ready = json.loads((run / 'health-ready.json').read_bytes())
    speed.BASE = 'http://127.0.0.1:' + str(config['api']['port'])
    report = dict(status='running', deployment=str(a.deployment), run_id=config['run_id'],
        temperature=1.0, top_p=.95, kit_sha256=config['kit_manifest_sha256'],
        baseline_serial=str(a.baseline_serial), baseline_c6=str(a.baseline_c6),
        new_baseline_measured=False, concurrency_only=a.concurrency_only, cases=[], waves=[],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def save():
        a.output.write_text(json.dumps(report, indent=2) + '\n')

    def hosts():
        rows = pair.results(pair.both(config, 'inspect'))
        reason = pair.stop_reason(rows, ready['containers'], ready['started_at'])
        if reason:
            raise RuntimeError(reason)
        if any(r['memory']['MemAvailable'] < 2**30 for r in rows):
            raise RuntimeError('Request admission margin unavailable')
        return rows

    def request(old, barrier=None):
        payload = old['request']
        assert payload['temperature'] == 1.0 and payload['top_p'] == .95
        req = urllib.request.Request(speed.BASE + '/v1/chat/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        if barrier:
            barrier.wait(timeout=15)
        started = time.monotonic()
        with urllib.request.urlopen(req, timeout=600) as response:
            row = speed.summarize_stream(list(speed.stream_events(response, started)))
        row.update(label=old['label'], request=payload)
        return row

    def after_request(before, spec_before, count, tokens):
        for _ in range(20):
            raw, after = speed.get_metrics()
            if after['request_generation_tokens_count'] - before['request_generation_tokens_count'] >= count:
                break
            time.sleep(.25)
        speed.assert_idle(raw)
        delta = {k: after[k] - v for k, v in before.items()}
        if (any(delta[k + '_count'] != count for k in speed.METRICS)
                or delta['request_generation_tokens_sum'] != tokens or any(v < 0 for v in delta.values())):
            raise ValueError('Request counters do not isolate the trial')
        spec_after = speed.speculative_snapshot(raw)
        assert spec_after.keys() == spec_before.keys()
        spec = {k: spec_after[k] - v for k, v in spec_before.items()}
        assert all(v >= 0 for v in spec.values())
        def counter(name):
            return sum(v for k, v in spec.items() if k.startswith('vllm:' + name + '{'))
        drafts = counter('spec_decode_num_drafts_total')
        acceptance = counter('spec_decode_num_accepted_tokens_total') / counter('spec_decode_num_draft_tokens_total')
        return after, delta, spec, drafts, acceptance

    with (run / 'request-probe.lock').open('a') as lock, (run / 'watch.lock').open('r') as watch:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            fcntl.flock(watch, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise RuntimeError('Independent RAM watchdog missing')
        report['initial_hosts'] = hosts()
        save()
        try:
            # Exercise stochastic-only kernels before collecting timings.
            raw, before = speed.get_metrics(); speed.assert_idle(raw)
            spec_before = speed.speculative_snapshot(raw)
            warmup_payload = dict(cases[0]['request'])
            warmup_payload.update(messages=[dict(role='user', content=
                'List everyday objects and explain their uses in full sentences.')], max_tokens=64)
            warmup = request(dict(label='excluded_candidate_warmup', request=warmup_payload))
            after_request(before, spec_before, 1, warmup['usage']['completion_tokens'])
            report['excluded_warmup'] = warmup
            save()
            for old in ([] if a.concurrency_only else cases):
                before_hosts = hosts()
                raw, before = speed.get_metrics(); speed.assert_idle(raw)
                spec_before = speed.speculative_snapshot(raw)
                row = request(old)
                after, delta, spec, drafts, acceptance = after_request(
                    before, spec_before, 1, row['usage']['completion_tokens'])
                timing = speed.isolated_timing(before, after, row['usage']['completion_tokens'])
                row.update(timing=timing, speculative_delta=spec, acceptance_fraction=acceptance,
                    tokens_per_step=timing['decode_tokens'] / drafts,
                    wall_ms_per_step=1000 * timing['server_decode_seconds'] / drafts,
                    host_before=before_hosts, host_after=hosts(),
                    baseline_decode_tps=old['timing']['decode_tokens_per_second'],
                    decode_tps_ratio=timing['decode_tokens_per_second'] / old['timing']['decode_tokens_per_second'])
                report['cases'].append(row); save()
                print(json.dumps({k: row[k] for k in ('label', 'timing', 'acceptance_fraction', 'decode_tps_ratio')}), flush=True)
            old = waves[0]
            hosts()
            raw, before = speed.get_metrics(); speed.assert_idle(raw)
            spec_before = speed.speculative_snapshot(raw)
            barrier = threading.Barrier(7)
            with ThreadPoolExecutor(max_workers=6) as pool:
                futures = [pool.submit(request, r, barrier) for r in old['requests']]
                started = time.monotonic(); barrier.wait(timeout=15)
                rows = [f.result() for f in futures]
                elapsed = time.monotonic() - started
            tokens = sum(r['usage']['completion_tokens'] for r in rows)
            after, delta, spec, drafts, acceptance = after_request(before, spec_before, 6, tokens)
            wave = dict(temperature=1.0, requests=rows, elapsed_seconds=elapsed,
                completion_tokens=tokens, aggregate_end_to_end_tps=tokens / elapsed,
                metric_delta=delta, speculative_delta=spec, acceptance_fraction=acceptance,
                baseline_tps=old['aggregate_end_to_end_tps'],
                throughput_ratio=(tokens / elapsed) / old['aggregate_end_to_end_tps'], hosts_after=hosts())
            report['waves'].append(wave)
            report.update(status='complete', median_serial_tps_ratio=statistics.median(
                r['decode_tps_ratio'] for r in report['cases']) if report['cases'] else None, final_hosts=hosts())
        except BaseException as error:
            report.update(status='failed', error=repr(error))
            raise
        finally:
            save()
    print(json.dumps(dict(status=report['status'], median_serial_tps_ratio=report['median_serial_tps_ratio'],
        c6_tps=wave['aggregate_end_to_end_tps'], c6_ratio=wave['throughput_ratio'],
        c6_acceptance=wave['acceptance_fraction'])), flush=True)


if __name__ == '__main__':
    main()
