# SPDX-License-Identifier: AGPL-3.0-only
"""Serving A/B for searched NVFP4 scales and the indexer key format on one deployment (head node).

Modes (same deployment, weights and serving settings; only the listed paths differ):
  control      the deployment's own kit (its recorded main-KV mode, MXFP4 index keys and queries)
  kv-search    new kit, main KV nvfp4_search; MXFP4 index keys and queries as control
  nvfp4-fp8    new kit, main KV nvfp4_search, NVFP4 index keys, FP8 decode / NVFP4 prefill queries
  nvfp4-nvfp4  new kit, main KV nvfp4_search, NVFP4 index keys, NVFP4 decode and prefill queries
For each mode: stop the owned pair, launch through model_fusion/launch.py, then run the
existing harnesses: model_fusion/benchmark.py (serial decode, C6 concurrency, 32K prefill) and the
generation diagnostics plus one long-context retrieval per --retrieval target (run_probe.py, on the
deployment's own API port; the launcher's controller keeps its RAM watch). Every step's status, command and report path is
recorded in reports/indexer-ab/<tag>/summary.json. A failing step is recorded and the next mode
runs. The pair is stopped at the end unless --leave-running names a mode, which is relaunched last.
Gate 4 (KL/top-1 against an unquantized-indexer reference) is not a serving test and is not run here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'release'))
import launch as release_launch  # noqa: E402  (read_state / controller_alive)
READY_TIMEOUT = 1800
SEARCH = ['--fp4-kv-mode', 'nvfp4_search']
MODES = {
    'control': dict(label='control', kit=False, flags=[]),
    'kv-search': dict(label='candidate', kit=True, flags=SEARCH),
    'nvfp4-fp8': dict(label='candidate', kit=True, flags=SEARCH + ['--indexer-k-format', 'nvfp4']),
    'nvfp4-nvfp4': dict(label='candidate', kit=True,
                        flags=SEARCH + ['--indexer-k-format', 'nvfp4', '--indexer-decode-query', 'nvfp4']),
}


def run(command, log):
    started = time.monotonic()
    with log.open('a') as stream:
        stream.write('$ ' + ' '.join(map(str, command)) + '\n')
        stream.flush()
        result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        stream.write(result.stdout)
    return dict(command=list(map(str, command)), returncode=result.returncode,
                seconds=round(time.monotonic() - started, 1), tail=result.stdout.strip().splitlines()[-3:])


def evict(args, log):
    """GB10 page cache counts against GPU memory: drop model/source caches on both hosts before a boot."""
    worker = json.loads(args.base_deployment.read_bytes())['nodes'][1]['ssh']
    script = 'artifacts/restore/evict_models.py'
    return [run([sys.executable, '-B', script], log),
            run(['ssh', '-n', worker, f'python3 -B /home/emi/code/ds41/{script}'], log)]


def launch(args, mode, log):
    spec = MODES[mode]
    for step in evict(args, log):
        if step['returncode']:
            return dict(step, started=None)
    command = [sys.executable, '-B', 'release/experimental/model_fusion/launch.py',
               '--base-deployment', args.base_deployment, '--base-sha256', args.base_sha256, '--port', '8888']
    if spec['kit']:
        command += ['--kit', args.kit, '--kit-sha256', args.kit_sha256]
    step = run(command + spec['flags'], log)
    started = None
    for line in reversed(step['tail']):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get('status') == 'model_fusion_serving_started':
            started = record
            break
    step['started'] = started
    return step


def wait_ready(started):
    """model_fusion/launch.py returns once the controller is spawned; wait for the pair's readiness marker."""
    config = json.loads(Path(started['deployment']).read_bytes())
    ready = Path(config['nodes'][0]['runs']) / config['run_id'] / 'pair' / 'health-ready.json'
    begin = time.monotonic()
    while time.monotonic() - begin < READY_TIMEOUT:
        if ready.exists():
            return dict(returncode=0, seconds=round(time.monotonic() - begin, 1), tail=[str(ready)])
        state = release_launch.read_state()
        if not state or not release_launch.controller_alive(state):
            return dict(returncode=1, seconds=round(time.monotonic() - begin, 1),
                        tail=['controller exited before readiness; see ./start-server.sh logs --controller'])
        time.sleep(15)
    return dict(returncode=1, seconds=READY_TIMEOUT, tail=['readiness timed out; pair left as is'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-deployment', type=Path, required=True)
    p.add_argument('--base-sha256', required=True)
    p.add_argument('--kit', type=Path, required=True, help='Search/NVFP4 kit derived from the deployment kit')
    p.add_argument('--kit-sha256', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--modes', nargs='+', choices=tuple(MODES), default=list(MODES))
    p.add_argument('--retrieval', type=int, nargs='+', default=[131072, 524288])
    p.add_argument('--leave-running', choices=tuple(MODES))
    args = p.parse_args()
    if hashlib.sha256(args.base_deployment.read_bytes()).hexdigest() != args.base_sha256:
        raise ValueError('Changed base deployment')
    out = ROOT / 'reports/indexer-ab' / args.tag
    if out.exists():
        raise ValueError('Preserve earlier A/B evidence: choose a new tag')
    out.mkdir(parents=True)
    log = out / 'commands.log'
    summary = dict(tag=args.tag, base_deployment=str(args.base_deployment), base_sha256=args.base_sha256,
                   kit=str(args.kit), kit_sha256=args.kit_sha256, modes={})

    def save():
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    for mode in args.modes:
        record = summary['modes'][mode] = dict(steps=[])
        save()
        record['steps'].append(dict(stop=run(['./stop-server.sh'], log)))
        step = launch(args, mode, log)
        record['steps'].append(dict(launch=step))
        if step['returncode'] or not step['started']:
            record['status'] = 'launch_failed'
            save()
            continue
        ready = wait_ready(step['started'])
        record['steps'].append(dict(ready=ready))
        save()
        if ready['returncode']:
            record['status'] = 'not_ready'
            save()
            continue
        deployment, digest = step['started']['deployment'], step['started']['deployment_sha256']
        mode_dir = out / mode
        mode_dir.mkdir()
        label = MODES[mode]['label']
        # check_serving_long_context writes only directly under reports/: use a unique name, then file it.
        raw_prefill = ROOT / 'reports' / f'model-fusion-ab-{args.tag}-{mode}-prefill-v1.json'
        if raw_prefill.exists():
            raise ValueError(f'Refusing to overwrite existing evidence: {raw_prefill}')
        bench = run([sys.executable, '-B', 'release/experimental/model_fusion/benchmark.py', '--deployment',
                     deployment, '--deployment-sha256', digest, '--reports', mode_dir, '--label', label,
                     '--prefill-raw-output', raw_prefill], log)
        if raw_prefill.exists():
            shutil.move(str(raw_prefill), str(mode_dir / raw_prefill.name))
        record['steps'].append(dict(benchmark=bench))
        probes = [('generation', None)] + [('long-context', target) for target in args.retrieval]
        for probe, target in probes:
            name = f'indexer-ab-{args.tag}-{mode}-{probe}' + (f'-{target}' if target else '') + '.json'
            # The portable request wrapper targets the old 127.0.0.1:8041 stack; run the probe on the
            # deployment's own API port instead (the controller's RAM watch stays active).
            command = [sys.executable, '-B', 'release/experimental/indexer_fp4/run_probe.py', '--deployment',
                       deployment, '--deployment-sha256', digest, '--probe', probe, '--output', ROOT / 'reports' / name]
            if target:
                command += ['--target-tokens', target]
            if (ROOT / 'reports' / name).exists():
                raise ValueError(f'Refusing to overwrite existing evidence: reports/{name}')
            result = run(command, log)
            if (ROOT / 'reports' / name).exists():
                shutil.move(str(ROOT / 'reports' / name), str(mode_dir / name))
                result['report'] = str(mode_dir / name)
            record['steps'].append({probe + (f'-{target}' if target else ''): result})
        failures = [key for item in record['steps'] for key, value in item.items() if value['returncode']]
        record['status'] = 'complete' if not failures else 'failed_steps'
        record['failed_steps'] = failures
        record['reports'] = sorted(str(q) for q in mode_dir.iterdir())
        save()
    if args.leave_running and args.leave_running != args.modes[-1]:
        summary['final'] = launch(args, args.leave_running, log) | dict(mode=args.leave_running)
    elif not args.leave_running:
        summary['final'] = dict(stop=run(['./stop-server.sh'], log))
    save()
    print(json.dumps({mode: value.get('status') for mode, value in summary['modes'].items()}))


if __name__ == '__main__':
    main()
