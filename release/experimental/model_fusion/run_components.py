# SPDX-License-Identifier: AGPL-3.0-only
"""Start paired component tests only after the recorded server is stopped."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment',type=Path,required=True)
    parser.add_argument('--deployment-sha256',required=True)
    parser.add_argument('--reports',type=Path,required=True)
    parser.add_argument('--probe',choices=('wo_a','wo_a_tuning','gather','mhc'),default='wo_a')
    parser.add_argument('--binary-dir',type=Path)
    args=parser.parse_args()
    raw=args.deployment.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.deployment_sha256:raise ValueError('Changed deployment')
    config=json.loads(raw);reports=args.reports.absolute()
    if (args.probe=='gather') != bool(args.binary_dir):raise ValueError('Gather requires its explicit binary directory')
    names=('probe_'+args.probe+'.py','packed_wo_a_rows.py') if args.probe in ('wo_a','wo_a_tuning') else ('probe_gather.py',) if args.probe=='gather' else ('probe_mhc.py','post_prenorm.py')
    if reports.resolve()!=reports or reports.exists():raise ValueError('Fresh component output required')
    def command(rank,argv):
        if config['nodes'][rank]['ssh']:
            argv=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',config['nodes'][rank]['ssh'],shlex.join(list(map(str,argv)))]
        return subprocess.check_output(argv,text=True,timeout=120)
    for rank,node in enumerate(config['nodes']):
        if command(rank,['docker','inspect',config['run_id']+f'-rank{rank}','--format','{{.State.Running}}']).strip()!='false':
            raise ValueError('Stop the recorded serving pair first')
        if command(rank,['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip():raise ValueError('GPU occupied')
        command(rank,['python3','-B',str(Path(node['kit'])/'verify.py'),node['kit'],'--manifest-sha256',config['kit_manifest_sha256']])
        check='import pathlib,sys; p=pathlib.Path(sys.argv[1]); assert p.resolve()==p and not p.exists(); a=next(int(s.split()[1])*1024 for s in pathlib.Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:")); assert a>=32*2**30; print(a)'
        command(rank,['python3','-c',check,str(reports)])
    for rank in (0,1):command(rank,['mkdir','-m','700',str(reports),str(reports/'code')])
    source=Path(__file__).resolve().parent
    for name in names:
        (reports/'code'/name).write_bytes((source/name).read_bytes())
    subprocess.run(['rsync','-a','--ignore-existing','--protect-args',str(reports/'code')+'/',config['nodes'][1]['ssh']+':'+str(reports/'code')+'/'],check=True)
    if args.binary_dir:
        args.binary_dir=args.binary_dir.absolute()
        receipt=json.loads((args.binary_dir/'complete.json').read_bytes())
        digest=receipt['binary_sha256']
        if hashlib.sha256((args.binary_dir/'dual_gather.so').read_bytes()).hexdigest()!=digest:raise ValueError('Changed binary')
        command(1,['mkdir','-p',str(args.binary_dir)])
        subprocess.run(['rsync','-a','--ignore-existing','--protect-args',str(args.binary_dir/'dual_gather.so'),str(args.binary_dir/'complete.json'),config['nodes'][1]['ssh']+':'+str(args.binary_dir)+'/'],check=True)
        for rank in (0,1):
            if command(rank,['sha256sum',str(args.binary_dir/'dual_gather.so')]).split()[0]!=digest:raise ValueError('Peer binary differs')
    def launch(rank):
        node=config['nodes'][rank];out=reports/f'host{rank}';kit=Path(node['kit'])
        command(rank,['mkdir','-m','700',str(out),str(out/'results'),str(out/'cache'),str(out/'cache/tmp')])
        env=dict(NVIDIA_VISIBLE_DEVICES='all',CUDA_VISIBLE_DEVICES='0',NVIDIA_DRIVER_CAPABILITIES='compute,utility',
                 PYTHONPATH='/work:/opt/ds41-serving:/opt/ds41-dcp-v3:/opt/exllamav3',PYTHONDONTWRITEBYTECODE='1',
                 OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MAX_JOBS='1',
                 TORCHINDUCTOR_COMPILE_THREADS='1',HF_HUB_OFFLINE='1',HF_HOME='/tmp/hf',
                 XDG_CACHE_HOME='/cache/xdg',TORCH_EXTENSIONS_DIR='/cache/torch',TMPDIR='/cache/tmp',
                 TRITON_CACHE_DIR='/cache/triton',CUDA_CACHE_PATH='/cache/cuda',VLLM_PLUGINS='',
                 TILELANG_CACHE_DIR='/cache/tilelang',DG_JIT_CACHE_DIR='/cache/deepgemm',
                 VLLM_CACHE_ROOT='/cache/vllm',FLASHINFER_WORKSPACE_BASE='/cache/flashinfer',
                 FLASHINFER_DISABLE_JIT='1',TORCH_CUDA_ARCH_LIST='12.1a',FLASHINFER_CUDA_ARCH_LIST='12.1a')
        argv=['docker','create','--name',f'ds41-{reports.name}-rank{rank}','--label=ds41.experiment=model-fusion',
              '--runtime=runc','--gpus=all','--network=none','--restart=no','--pull=never','--read-only',
              '--cap-drop=ALL','--security-opt=no-new-privileges','--memory=8g','--memory-swap=8g','--cpus=6',
              '--pids-limit=256','--shm-size=256m','--user',f'{node["uid"]}:{node["gid"]}',
              '--tmpfs=/tmp:rw,exec,nosuid,nodev,size=256m','--workdir=/cache']
        mounts=[(reports/'code','/work'),(kit/'serving','/opt/ds41-serving'),(node['model'],'/model')]
        if args.binary_dir:mounts.append((args.binary_dir,'/candidate'))
        mounts += [(src,'/model/'+name) for name,src in sorted(node.get('model_bindings',{}).items())]
        for src,dst in mounts:argv+=['--mount',f'type=bind,src={src},dst={dst},readonly']
        for name in ('results','cache'):argv+=['--mount',f'type=bind,src={out/name},dst=/{name}']
        argv+=['--env='+key+'='+value for key,value in env.items()]
        argv+=['--entrypoint=/opt/ds41-venv/bin/python',node['image'],'-u','-B','/work/probe_'+args.probe+'.py',
               '--rank',str(rank),'--output','/results/'+args.probe+'.json']
        cid=command(rank,argv).strip()
        record=dict(rank=rank,container=cid,command=argv,kit_manifest_sha256=config['kit_manifest_sha256'],
                    sources={name:hashlib.sha256((reports/'code'/name).read_bytes()).hexdigest()
                             for name in names})
        (reports/f'host{rank}-launch.json').write_text(json.dumps(record,indent=2)+'\n')
        command(rank,['docker','start',cid]);return record
    with ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(launch,(0,1)))
    print(json.dumps(dict(status='model_fusion_components_started',reports=str(reports),containers=[r['container'] for r in records])),flush=True)


if __name__=='__main__':main()
