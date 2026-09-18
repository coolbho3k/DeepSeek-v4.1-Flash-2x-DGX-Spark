# SPDX-License-Identifier: AGPL-3.0-only
"""Run bounded overlap probes on an explicitly stopped, recorded Spark pair."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from prepare import load_parent, encoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', type=Path, required=True)
    parser.add_argument('--deployment-sha256', required=True)
    parser.add_argument('--kit', type=Path, required=True)
    parser.add_argument('--kit-sha256', required=True)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--invalid-replay', action='store_true')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    if not 60 <= args.timeout <= 1800:
        raise ValueError('Bounded test timeout required')
    raw = args.deployment.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.deployment_sha256:
        raise ValueError('Changed serving deployment')
    config = json.loads(raw)
    load_parent(args.kit, args.kit_sha256)
    parent = Path(config['nodes'][0]['kit'])
    load_parent(parent, config['kit_manifest_sha256'])
    sys.path.insert(0, str(parent / 'tools'))
    spec = importlib.util.spec_from_file_location('overlap_parent_node', parent / 'tools/portable_node.py')
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    reports = args.reports.absolute()
    if reports.resolve() != reports or reports.exists():
        raise ValueError('Fresh unredirected report directory required')

    def command(index, argv, timeout=60, **kwargs):
        host = config['nodes'][index]['ssh']
        if host:
            argv = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, shlex.join(list(map(str, argv)))]
        return subprocess.check_output(argv, text=True, timeout=timeout, **kwargs)

    for index in (0, 1):
        if command(index, ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).strip():
            raise ValueError('GPU busy; this runner never stops an existing workload')
        if command(index, ['ss', '-H', '-ltn', 'sport = :29579']).strip():
            raise ValueError('Test rendezvous port is occupied')
        if command(index, ['docker', 'inspect', config['run_id'] + '-rank' + str(index),
                           '--format', '{{.State.Running}}']).strip() != 'false':
            raise ValueError('Recorded serving pair must be stopped first')
    reports.mkdir(mode=0o700)
    peer = config['nodes'][1]['ssh']
    # Only a fresh candidate is copied. No models, credentials or mutable
    # live overlays are read/copied, and no receiver file is overwritten.
    exists = json.loads(command(1, ['python3', '-c',
        'import json,pathlib,sys; print(json.dumps(pathlib.Path(sys.argv[1]).exists()))', str(args.kit)]))
    if not exists:
        command(1, ['mkdir', str(args.kit)])
        subprocess.run(['rsync', '-a', '--ignore-existing', '--protect-args', str(args.kit) + '/',
                        peer + ':' + str(args.kit) + '/'], check=True)
    command(1, ['python3', '-B', str(args.kit / 'verify.py'), str(args.kit),
                '--manifest-sha256', args.kit_sha256])
    command(1, ['mkdir', str(reports)])
    active = {}

    def launch(index):
        node = config['nodes'][index]
        directory = reports / ('host' + str(index))
        command(index, ['mkdir', str(directory)])
        command(index, ['mkdir', str(directory / 'cache'), str(directory / 'cache/tmp')])
        env = native.docker_command(config, index)['env']
        env.update(VLLM_PLUGINS='', PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
            WORLD_SIZE='2', LOCAL_WORLD_SIZE='1', RANK=str(index), LOCAL_RANK='0',
            MASTER_ADDR=config['nodes'][0]['fabric_ip'], MASTER_PORT='29579',
            DG_JIT_CACHE_DIR='/cache/deepgemm', FLASHINFER_DISABLE_JIT='1')
        name = 'ds41-overlap-' + reports.name + '-rank' + str(index)
        if not re.fullmatch('[a-zA-Z0-9_-]{1,120}', name):
            raise ValueError('Invalid test container name')
        argv = ['docker', 'create', '--name', name, '--label=ds41.experiment=dcp-overlap',
            '--runtime=runc', '--gpus=all', '--network=host', '--restart=no', '--pull=never', '--read-only',
            '--cap-drop=ALL', '--cap-add=IPC_LOCK', '--security-opt=no-new-privileges',
            '--device=/dev/infiniband', '--ulimit=memlock=-1:-1', '--memory=4g', '--memory-swap=4g',
            '--cpus=4', '--pids-limit=256', '--shm-size=128m', '--user=' + str(node['uid']) + ':' + str(node['gid']),
            '--tmpfs=/tmp:rw,exec,nosuid,nodev,size=256m', '--workdir=/cache',
            '--mount', f'type=bind,src={args.kit}/serving,dst=/opt/ds41-serving,readonly',
            '--mount', f'type=bind,src={args.kit}/experiments/dcp_overlap/probe_gpu.py,dst=/opt/probe_gpu.py,readonly',
            '--mount', f'type=bind,src={args.kit}/aot/{native.AOT_NAME},dst={native.AOT_TARGET},readonly',
            '--mount', f'type=bind,src={directory}/cache,dst=/cache',
            '--mount', f'type=bind,src={directory},dst=/results']
        argv += ['--env=' + key + '=' + value for key, value in env.items()]
        argv += ['--entrypoint=/opt/ds41-venv/bin/python', node['image'], '-u', '-B', '/opt/probe_gpu.py',
                 '--acknowledge-idle-gpus', '--invalid-replay' if args.invalid_replay else '--timing']
        if args.trace:
            argv.append('--trace')
        cid = command(index, argv).strip()
        if not re.fullmatch('[0-9a-f]{64}', cid):
            raise ValueError('Invalid created container identity')
        active[index] = cid
        (reports / f'host{index}-launch.json').write_bytes(encoded(dict(container=cid, command=argv)))
        command(index, ['docker', 'start', cid])
        return index, cid

    states = {}
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(launch, (0, 1)))
        print(json.dumps(dict(stage='paired_component_test_started', containers=active, reports=str(reports))), flush=True)
        deadline = time.monotonic() + args.timeout
        while active:
            for index, cid in list(active.items()):
                state = json.loads(command(index, ['docker', 'inspect', '--format', '{{json .State}}', cid]))
                if state['Running']:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Bounded paired component deadline reached')
                    continue
                states[index] = state
                text = command(index, ['docker', 'logs', '--timestamps', cid], stderr=subprocess.STDOUT)
                (reports / f'host{index}.log').write_text(text)
                del active[index]
                print(json.dumps(dict(rank=index, state=state, tail=text[-2200:])), flush=True)
                if state['ExitCode'] or state['OOMKilled']:
                    raise RuntimeError('One probe failed; stop only its paired test peer')
            if active:
                time.sleep(2)
    finally:
        for index, cid in list(active.items()):
            command(index, ['docker', 'stop', '-t', '5', cid])
            states[index] = json.loads(command(index, ['docker', 'inspect', '--format', '{{json .State}}', cid]))
            (reports / f'host{index}.log').write_text(command(index, ['docker', 'logs', '--timestamps', cid], stderr=subprocess.STDOUT))
        (reports / 'states.json').write_bytes(encoded(states))
    expected = 'negative_component_pass' if args.invalid_replay else 'component_pass'
    for index in (0, 1):
        if '"status": "' + expected + '"' not in (reports / f'host{index}.log').read_text():
            raise RuntimeError('Missing component success receipt')
    (reports / 'complete.json').write_bytes(encoded(dict(status=expected, both_ranks=True,
        candidate=str(args.kit), manifest_sha256=args.kit_sha256, serving_qualified=False)))
    print(json.dumps(dict(status=expected, reports=str(reports))), flush=True)


if __name__ == '__main__':
    main()
