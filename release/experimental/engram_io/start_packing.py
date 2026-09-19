# SPDX-License-Identifier: AGPL-3.0-only
"""Start bounded CPU packing on an explicitly stopped, pinned serving pair.

Copies only this small experiment, builds its CPU reader, and starts one
identified CPU-only container per host. Does not read env files or credentials,
stop workloads, overwrite any artifacts, or change the public launch profile.
Inspect the recorded container identities to monitor; never restart blindly.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment', type=Path, required=True)
    p.add_argument('--deployment-sha256', required=True)
    p.add_argument('--reports', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    raw = a.deployment.read_bytes()
    if hashlib.sha256(raw).hexdigest() != a.deployment_sha256:
        raise ValueError('Reference deployment changed')
    config = json.loads(raw)
    reports, output = a.reports.absolute(), a.output.absolute()
    if reports.resolve() != reports or output.resolve() != output:
        raise ValueError('Unredirected fresh paths required')

    def command(index, argv, **kwargs):
        host = config['nodes'][index]['ssh']
        argv = list(map(str, argv))
        if host:
            argv = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, shlex.join(argv)]
        return subprocess.check_output(argv, text=True, timeout=120, **kwargs)

    for rank in (0, 1):
        if command(rank, ['docker', 'inspect', config['run_id']+'-rank'+str(rank),
                          '--format', '{{.State.Running}}']).strip() != 'false':
            raise ValueError('Recorded server must be stopped first')
        command(rank, ['python3', '-c',
            'import pathlib,sys; assert all(not pathlib.Path(x).exists() and not pathlib.Path(x).is_symlink() for x in sys.argv[1:])',
            reports, output])
    command(0, ['mkdir', '-m', '700', reports, output])
    code = reports / 'code'
    code.mkdir(mode=0o700)
    source = Path(__file__).resolve().parent
    names = ('packing.py', 'pack_rank.py', 'native.py', 'test_cpu.py', 'row_store.cpp', 'row_store_core.cpp')
    hashes = {}
    for name in names:
        data = (source/name).read_bytes()
        with (code/name).open('xb') as f:
            f.write(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    command(1, ['mkdir', '-m', '700', reports, output, code])
    subprocess.run(['rsync', '-a', '--ignore-existing', '--protect-args', str(code)+'/',
                    config['nodes'][1]['ssh']+':'+str(code)+'/'], check=True)

    def launch(rank):
        node = config['nodes'][rank]
        results = reports / ('host'+str(rank))
        packed = output / ('rank'+str(rank))
        command(rank, ['mkdir', '-m', '700', results, packed])
        binary = results / 'librow_store.so'
        command(rank, ['g++', '-O3', '-std=c++17', '-shared', '-fPIC', '-pthread',
                       code/'row_store.cpp', '-o', binary])
        metadata = command(rank, ['python3', '-c',
            'import hashlib,json,pathlib,sys; print(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in map(pathlib.Path,sys.argv[1:])}))',
            binary, *[code/name for name in names]])
        observed = json.loads(metadata)
        if any(observed[str(code/name)] != value for name, value in hashes.items()):
            raise ValueError('Copied code differs')
        # Model views may contain empty bind targets, not symlinks. Resolve
        # explicit per-file deployment bindings before considering model paths.
        model = node['model']
        bindings = node.get('model_bindings', {})
        sources = [bindings.get(f'engrams/engram-layer-{layer:02}.safetensors',
                    str(Path(model)/f'engrams/engram-layer-{layer:02}.safetensors')) for layer in (1, 14)]
        resolved = json.loads(command(rank, ['python3', '-c',
            'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); tables=[pathlib.Path(x).resolve(strict=True) for x in sys.argv[2:]]; assert all(x.stat().st_size>4096 for x in tables) and tables[0].parent==tables[1].parent; print(json.dumps(dict(config=str((p/"config.json").resolve(strict=True)),sources=str(tables[0].parent))))', model, *sources]))
        name = 'ds41-engram-pack-'+reports.name+'-rank'+str(rank)
        args = ['docker', 'create', '--name', name, '--label=ds41.experiment=engram-packing',
            '--runtime=runc', '--network=none', '--restart=no', '--pull=never', '--read-only',
            '--cap-drop=ALL', '--security-opt=no-new-privileges', '--memory=512m', '--memory-swap=512m',
            '--cpus=3', '--pids-limit=192', '--user='+str(node['uid'])+':'+str(node['gid']),
            '--tmpfs=/tmp:rw,nosuid,nodev,size=16m', '--workdir=/work',
            '--env=OPENBLAS_NUM_THREADS=1', '--env=PYTHONDONTWRITEBYTECODE=1',
            '--mount', f'type=bind,src={code},dst=/work,readonly',
            '--mount', f'type=bind,src={results},dst=/results',
            '--mount', f'type=bind,src={packed},dst={packed}',
            '--mount', f'type=bind,src={resolved["sources"]},dst={resolved["sources"]},readonly',
            '--mount', f'type=bind,src={resolved["config"]},dst=/model-config.json,readonly',
            '--entrypoint=python3', node['image'], '-u', '-B', '-c',
            'import sys,unittest,pack_rank; result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName("test_cpu")); assert result.wasSuccessful(); sys.argv=["pack_rank.py",*sys.argv[1:]]; pack_rank.main()',
            '--config', '/model-config.json', '--sources', resolved['sources'], '--rank', str(rank), '--output', str(packed)]
        cid = command(rank, args).strip()
        record = dict(rank=rank, container=cid, command=args, code_sha256=observed,
                      original_source_read_only=True, output=str(packed))
        with (reports/('host'+str(rank)+'-launch.json')).open('x') as f:
            json.dump(record, f, indent=2); f.write('\n')
        command(rank, ['docker', 'start', cid])
        return record
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(launch, (0, 1)))
    print(json.dumps(dict(stage='packing_started', containers={r['rank']:r['container'] for r in records},
                          reports=str(reports), output=str(output))), flush=True)


if __name__ == '__main__':
    main()
