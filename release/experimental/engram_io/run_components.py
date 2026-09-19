# SPDX-License-Identifier: AGPL-3.0-only
"""Start identified, bounded Engram component tests on the stopped Spark pair.

No workload is stopped here. No env/credential files are read. Uses the pinned
deployment's exact images and frozen parent overlay, with fresh experiment
code/results. Containers have no networking and no writable model mounts.
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
    p.add_argument('--packing-reports', type=Path, required=True)
    p.add_argument('--packed', type=Path, required=True)
    p.add_argument('--reports', type=Path, required=True)
    p.add_argument('--kind', choices=('gpu', 'ssd'), required=True)
    a = p.parse_args()
    raw = a.deployment.read_bytes()
    if hashlib.sha256(raw).hexdigest() != a.deployment_sha256:
        raise ValueError('Reference deployment changed')
    config = json.loads(raw)
    reports, packed, packing = [p.absolute() for p in (a.reports, a.packed, a.packing_reports)]
    def command(rank, argv, **kwargs):
        argv = list(map(str, argv))
        if config['nodes'][rank]['ssh']:
            argv = ['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',
                    config['nodes'][rank]['ssh'],shlex.join(argv)]
        return subprocess.check_output(argv,text=True,timeout=120,**kwargs)
    for rank in (0,1):
        if command(rank,['docker','inspect',config['run_id']+'-rank'+str(rank),'--format','{{.State.Running}}']).strip()!='false':
            raise ValueError('Stop the recorded serving pair before component tests')
        if a.kind=='gpu' and command(rank,['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip():
            raise ValueError('GPU is occupied; no workload will be stopped')
        if a.kind=='ssd':
            command(rank,['python3','-c',
                'import json,pathlib,sys; assert json.loads(pathlib.Path(sys.argv[1]).read_text())["status"]=="complete"',
                packed/('rank'+str(rank))/'rank-manifest.json'])
        command(rank,['python3','-c','import pathlib,sys; p=pathlib.Path(sys.argv[1]); assert p.resolve()==p and not p.exists()',reports])
    for rank in (0,1):
        command(rank,['mkdir','-m','700',reports,reports/'code'])
    source=Path(__file__).resolve().parent
    names=('packing.py','native.py','test_cpu.py','stage_transform.py','overlap.py','probe_gpu.py','bench_ssd.py')
    for name in names:
        with (reports/'code'/name).open('xb') as f:
            f.write((source/name).read_bytes())
    subprocess.run(['rsync','-a','--ignore-existing','--protect-args',str(reports/'code')+'/',
        config['nodes'][1]['ssh']+':'+str(reports/'code')+'/'],check=True)
    def launch(rank):
        node=config['nodes'][rank]
        kit=Path(node['kit'])
        library=packing/('host'+str(rank))/'librow_store.so'
        code=reports/'code'
        out=reports/('host'+str(rank))
        command(rank,['mkdir','-m','700',out,out/'results',out/'cache'])
        if a.kind=='gpu':
            command(rank,['python3','-B',code/'stage_transform.py','--parent',kit/'serving/miaai_engram.py',
                          '--library',library,'--output',code/'engram_candidate_stage.py'])
        argv=['docker','create','--name','ds41-engram-'+a.kind+'-'+reports.name+'-rank'+str(rank),
            '--label=ds41.experiment=engram-'+a.kind,'--runtime=runc','--network=none','--restart=no','--pull=never',
            '--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--cpus=20','--pids-limit=256',
            '--memory='+('8g' if a.kind=='gpu' else '1g'),'--memory-swap='+('8g' if a.kind=='gpu' else '1g'),
            '--user='+str(node['uid'])+':'+str(node['gid']),'--tmpfs=/tmp:rw,exec,nosuid,nodev,size=256m',
            '--workdir=/cache','--env=OPENBLAS_NUM_THREADS=1','--env=OMP_NUM_THREADS=1',
            '--env=PYTHONDONTWRITEBYTECODE=1','--env=VLLM_PLUGINS=',
            '--env=PYTHONPATH=/work:/opt/ds41-serving:/opt/ds41-dcp-v3:/opt/exllamav3',
            '--env=TRITON_CACHE_DIR=/cache/triton','--env=TORCHINDUCTOR_CACHE_DIR=/cache/inductor',
            '--mount',f'type=bind,src={code},dst=/work,readonly',
            '--mount',f'type=bind,src={library},dst=/native/librow_store.so,readonly',
            '--mount',f'type=bind,src={out}/results,dst=/results',
            '--mount',f'type=bind,src={out}/cache,dst=/cache',
            '--mount',f'type=bind,src={kit}/serving,dst=/opt/ds41-serving,readonly']
        if a.kind=='gpu':
            argv+=['--gpus=all','--cap-add=IPC_LOCK','--ulimit=memlock=-1:-1',
                   '--env=NVIDIA_DRIVER_CAPABILITIES=compute,utility']
            probe=['/work/probe_gpu.py','--rank',str(rank),'--library','/native/librow_store.so','--results','/results']
        else:
            rank_dir=packed/('rank'+str(rank))
            data=json.loads(command(rank,['python3','-c','import pathlib,sys; print(pathlib.Path(sys.argv[1]).read_text())',rank_dir/'rank-manifest.json']))
            source_dirs={str(Path(s['source']['path']).parent) for s in data['shards']}
            argv+=['--mount',f'type=bind,src={rank_dir},dst={rank_dir},readonly']
            for folder in sorted(source_dirs):
                argv+=['--mount',f'type=bind,src={folder},dst={folder},readonly']
            probe=['/work/bench_ssd.py','--manifest',str(rank_dir/'rank-manifest.json'),
                   '--library','/native/librow_store.so','--output','/results/ssd.json']
        argv+=['--entrypoint=/opt/ds41-venv/bin/python',node['image'],'-u','-B',*probe]
        cid=command(rank,argv).strip()
        record=dict(container=cid,rank=rank,kind=a.kind,command=argv)
        with (reports/('host'+str(rank)+'-launch.json')).open('x') as f:
            json.dump(record,f,indent=2);f.write('\n')
        command(rank,['docker','start',cid])
        return record
    with ThreadPoolExecutor(max_workers=2) as pool:
        records=list(pool.map(launch,(0,1)))
    print(json.dumps(dict(kind=a.kind,reports=str(reports),containers={x['rank']:x['container'] for x in records})),flush=True)


if __name__=='__main__':
    main()
