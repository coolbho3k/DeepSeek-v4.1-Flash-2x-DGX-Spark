# SPDX-License-Identifier: AGPL-3.0-only
"""Run bounded MoE components on both idle Sparks; never stop a server.

Uses a pinned deployment's frozen image, overlay and read-only model bindings.
No systemd, credential access, environment files or external container network.
Returned container IDs are the authority for monitoring; do not blindly retry.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import runpy
import shlex
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment',type=Path,required=True)
    p.add_argument('--deployment-sha256',required=True)
    p.add_argument('--reports',type=Path,required=True)
    p.add_argument('--prefill-sweep',action='store_true')
    p.add_argument('--native-candidate',type=Path)
    p.add_argument('--native-sha256')
    a=p.parse_args()
    raw=a.deployment.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=a.deployment_sha256:raise ValueError('Changed deployment')
    config=json.loads(raw);reports=a.reports.absolute()
    if bool(a.native_candidate)!=bool(a.native_sha256):raise ValueError('Supply candidate path and digest together')
    if a.native_candidate:
        a.native_candidate=a.native_candidate.absolute()
        if a.prefill_sweep or a.native_candidate.resolve()!=a.native_candidate:raise ValueError('One explicit candidate at a time')
        candidate=json.loads((a.native_candidate/'complete.json').read_bytes())
        if (candidate['status']!='native_experiment_built_cpu_only' or candidate['abi']!=2
                or candidate['binary_sha256']!=a.native_sha256
                or hashlib.sha256((a.native_candidate/'cooperative_moe.so').read_bytes()).hexdigest()!=a.native_sha256):
            raise ValueError('Changed candidate binary')
    def command(rank,argv,**kwargs):
        argv=list(map(str,argv));worker=config['nodes'][rank]['ssh']
        if worker:argv=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',worker,shlex.join(argv)]
        return subprocess.check_output(argv,text=True,timeout=120,**kwargs)
    for rank,node in enumerate(config['nodes']):
        if command(rank,['docker','inspect',config['run_id']+f'-rank{rank}','--format','{{.State.Running}}']).strip()!='false':
            raise ValueError('Explicitly stop the recorded serving pair first')
        if command(rank,['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip():
            raise ValueError('GPU occupied; no workload will be stopped')
        command(rank,['python3','-B',Path(node['kit'])/'verify.py',node['kit'],
                      '--manifest-sha256',config['kit_manifest_sha256']])
        admission='import pathlib,sys; p=pathlib.Path(sys.argv[1]); assert p.resolve()==p and not p.exists(); available=next(int(s.split()[1])*1024 for s in pathlib.Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:")); assert available>=32*2**30; print(available)'
        command(rank,['python3','-c',admission,reports])
    for rank in (0,1):command(rank,['mkdir','-m','700',reports,reports/'code'])
    if a.native_candidate:
        command(1,['mkdir','-p',a.native_candidate])
        subprocess.run(['rsync','-a','--ignore-existing','--protect-args',
            str(a.native_candidate/'cooperative_moe.so'),str(a.native_candidate/'complete.json'),
            config['nodes'][1]['ssh']+':'+str(a.native_candidate)+'/'],check=True)
        for rank in (0,1):
            actual=command(rank,['sha256sum',a.native_candidate/'cooperative_moe.so']).split()[0]
            if actual!=a.native_sha256:raise ValueError('Peer candidate mismatch')
    source=Path(__file__).resolve().parent
    for name in ('probe_geometry.py','geometry.py','prefill_sweep.py','probe_native.py'):
        with (reports/'code'/name).open('xb') as out:out.write((source/name).read_bytes())
    if a.native_candidate and candidate['experiment']==401:
        with (reports/'code/dspark_contracts.py').open('xb') as out:
            out.write((source.parent/'dspark/contracts.py').read_bytes())
    subprocess.run(['rsync','-a','--ignore-existing','--protect-args',str(reports/'code')+'/',
        config['nodes'][1]['ssh']+':'+str(reports/'code')+'/'],check=True)
    def launch(rank):
        node=config['nodes'][rank];kit=Path(node['kit']);output=reports/f'host{rank}'
        command(rank,['mkdir','-m','700',output,output/'results',output/'cache',output/'cache/tmp'])
        env=dict(NVIDIA_VISIBLE_DEVICES='all',CUDA_VISIBLE_DEVICES='0',NVIDIA_DRIVER_CAPABILITIES='compute,utility',
            PYTHONPATH='/work:/opt/ds41-serving:/opt/ds41-dcp-v3:/opt/exllamav3',
            DS41_ENABLE_DCP2='1',VLLM_USE_V2_MODEL_RUNNER='1',DSV41_IO_THREADS='96',
            PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            MAX_JOBS='1',TORCHINDUCTOR_COMPILE_THREADS='1',HUMMING_DISABLE_PARALLEL_BUILD='1',
            HF_HUB_OFFLINE='1',HF_HOME='/tmp/hf',XDG_CACHE_HOME='/cache/xdg',
            TORCH_EXTENSIONS_DIR='/cache/torch',TMPDIR='/cache/tmp',DG_JIT_CACHE_DIR='/cache/deepgemm',
            FLASHINFER_WORKSPACE_BASE='/cache/flashinfer',VLLM_CACHE_ROOT='/cache/vllm',
            TRITON_CACHE_DIR='/cache/triton',TILELANG_CACHE_DIR='/cache/tilelang',CUDA_CACHE_PATH='/cache/cuda',
            VLLM_PLUGINS='',FLASHINFER_DISABLE_JIT='1',TORCH_CUDA_ARCH_LIST='12.1a',FLASHINFER_CUDA_ARCH_LIST='12.1a',
            DS41_ENABLE_DSPARK='1',DS41_ENABLE_SSD_VOCAB='1',DS41_ENABLE_COOPERATIVE_MOE='1')
        # Registration validates the current profile even though this probe
        # never allocates KV. Do not inherit obsolete defaults or shell env.
        profile=runpy.run_path(str(kit/'tools/launch_profile.py'))
        env.update(profile['environment'](config['serving']))
        env['VLLM_SPARSE_INDEXER_MAX_LOGITS_MB']='128'
        argv=['docker','create','--name',f'ds41-{reports.name}-rank{rank}',
            '--label=ds41.experiment=moe-critical-path','--runtime=runc','--gpus=all','--network=none',
            '--restart=no','--pull=never','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges',
            '--memory=8g','--memory-swap=8g','--cpus=6','--pids-limit=256','--shm-size=256m',
            '--user',f'{node["uid"]}:{node["gid"]}','--tmpfs=/tmp:rw,exec,nosuid,nodev,size=256m','--workdir=/cache']
        mounts=[(reports/'code','/work'),(kit/'serving','/opt/ds41-serving'),(node['model'],'/model'),
            (kit/'aot/mxfp8_gemm_cutlass_sm120','/usr/local/lib/python3.12/dist-packages/flashinfer/data/aot/mxfp8_gemm_cutlass_sm120')]
        mounts += [(source,'/model/'+name) for name,source in sorted(node.get('model_bindings',{}).items())]
        if a.native_candidate:mounts.append((a.native_candidate,'/candidate'))
        for source,target in mounts:argv+=['--mount',f'type=bind,src={source},dst={target},readonly']
        for name,target in (('results','/results'),('cache','/cache')):
            argv+=['--mount',f'type=bind,src={output/name},dst={target}']
        argv+=['--env='+key+'='+value for key,value in env.items()]
        argv+=['--entrypoint=/opt/ds41-venv/bin/python',node['image'],'-u','-B','/work/probe_geometry.py',
            '--maintenance','--model','/model','--serving','/opt/ds41-serving','--rank',str(rank),
            '--experts','384',
            '--prefill-sweep' if a.prefill_sweep else '--prefill-profile',
            '--output','/results/prefill-sweep.json' if a.prefill_sweep else '/results/geometry.json']
        if a.native_candidate:argv+=['--native-candidate','/candidate']
        cid=command(rank,argv).strip()
        record=dict(rank=rank,container=cid,command=argv,kit_sha256=config['kit_manifest_sha256'])
        with (reports/f'host{rank}-launch.json').open('x') as out:json.dump(record,out,indent=2)
        command(rank,['docker','start',cid]);return record
    with ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(launch,(0,1)))
    print(json.dumps(dict(status='component_pair_started',reports=str(reports),
        containers={r['rank']:r['container'] for r in records})),flush=True)


if __name__=='__main__':main()
