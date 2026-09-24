# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded standalone probes on one idle host; no serving/driver configuration."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
import uuid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', required=True)
    p.add_argument('--directory', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--full-gpu', action='store_true', help='Full GPU pattern and CUDA graph checks')
    p.add_argument('--ordinary-mib', type=int, default=0, help='Native allocator prefix, 0..1024 MiB')
    p.add_argument('probe_args', nargs=6)
    a = p.parse_args()
    if not re.fullmatch('[a-zA-Z0-9@_.-]+', a.host):
        p.error('plain SSH host required')
    if not re.fullmatch('/tmp/ds41-display-595[.][a-zA-Z0-9]+', a.directory):
        p.error('use a dedicated mktemp probe directory')
    if not re.fullmatch('sha256:[0-9a-f]{64}', a.image):
        p.error('immutable installed image required')
    if not re.fullmatch('[a-z0-9-]{1,60}', a.tag):
        p.error('simple unique report tag required')
    if not 0 <= a.ordinary_mib <= 1024 or (a.ordinary_mib and a.probe_args[0] != 'native'):
        p.error('ordinary prefix requires native backend and 0..1024 MiB')
    out = Path(__file__).resolve().parents[3]/'reports/display-driver-595'
    out.mkdir(parents=True, exist_ok=True)
    report = out/(a.tag+'.local.json')
    if report.exists():
        raise ValueError('Do not overwrite a prior report')

    def call(argv, timeout=30, check=True):
        return subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',
            a.host,shlex.join(argv)],text=True,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,timeout=timeout,check=check)

    jobs = call(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).stdout.strip()
    if jobs:
        raise ValueError('GPU busy; no concurrent allocations allowed')
    identity = json.loads(call(['python3','-c',
        'import os,json; print(json.dumps([os.getuid(),os.getgid(),os.stat("/dev/dri/card0").st_gid]))']).stdout)
    args = ['docker','create','--name','ds41-display595-'+uuid.uuid4().hex[:16],
        '--pull=never','--runtime=runc','--gpus=all','--network=none','--read-only',
        '--cap-drop=ALL','--cap-add=IPC_LOCK','--security-opt=no-new-privileges',
        '--memory=6g','--memory-swap=6g','--cpus=2','--pids-limit=128',
        '--ulimit=core=0','--ulimit=memlock=-1:-1',
        '--device=/dev/dri/card0','--group-add',str(identity[2]),
        '--user',f'{identity[0]}:{identity[1]}',
        '--env=NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display',
        '--tmpfs=/tmp:rw,nosuid,nodev,size=128m',
        '--mount',f'type=bind,src={a.directory}/probe,dst=/probe,readonly',
        '--mount',f'type=bind,src={a.directory}/original.so,dst=/original.so,readonly',
        '--env',f'DS41_PROBE_ORDINARY_MIB={a.ordinary_mib}']
    if a.full_gpu:
        args += ['--env=DS41_PROBE_FULL_GPU=1','--mount',
            f'type=bind,src={a.directory}/kernels.cubin,dst=/kernels.cubin,readonly']
    args += ['--entrypoint=/probe',a.image,*a.probe_args]
    cid = call(args).stdout.strip()
    if not re.fullmatch('[0-9a-f]{64}',cid):
        raise ValueError('Invalid newly created container ID')
    row = dict(host=a.host,container=cid,command=args,tag=a.tag)
    report.write_text(json.dumps(row,indent=2)+'\n')
    begin = time.monotonic()
    try:
        run = call(['docker','start','-a',cid], timeout=120, check=False)
        row['attach_code'] = run.returncode
    except subprocess.TimeoutExpired:
        row['timeout'] = True
        call(['docker','stop','--time','5',cid],timeout=15)
    row['elapsed_seconds'] = time.monotonic()-begin
    row['state'] = json.loads(call(['docker','inspect','--format','{{json .State}}',cid]).stdout)
    row['log'] = call(['docker','logs',cid]).stdout
    report.write_text(json.dumps(row,indent=2)+'\n')
    print(row['log'],flush=True)
    print(json.dumps(dict(report=str(report),exit_code=row['state']['ExitCode'],
        oom_killed=row['state']['OOMKilled'],elapsed_seconds=row['elapsed_seconds'])),flush=True)
    if row['state']['Running']:
        raise RuntimeError('Probe still running; inspect its exact container')


if __name__ == '__main__':
    main()
