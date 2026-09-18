"""Verify the exact seed-only transform and real Ninja command expansion.

CPU only: `ninja -t commands` lists commands; no compiler is executed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import re
import subprocess
import sys
import tempfile

import generate_mxfp8_graph_cpu as graph
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_clean_mxfp8_bootstrap as controller


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container-name',help='Explicit fresh CPU-only pinned-image Ninja diagnostic')
    args=parser.parse_args()
    if args.container_name and not re.fullmatch(r'ds41-seeded-graph-contract-v[1-9][0-9]*',args.container_name):
        raise ValueError('Use a fresh task-specific diagnostic container name')
    root=Path(__file__).resolve().parents[1]
    native=(root/'artifacts/ds41-mxfp8-clean-bootstrap-v1-cache'/controller.MODULE/'build.ninja').read_bytes()
    transformed=graph.seeded_graph(native)
    assert hashlib.sha256(transformed).hexdigest()==controller.SEEDED_GRAPH_SHA
    assert transformed.replace(b' --frandom-seed=$in',b'')==native
    assert hashlib.sha256(Path(graph.__file__).read_bytes()).hexdigest()==controller.PROBE_SHA
    for bad in (b'',native+b'\n',native.replace(b'-O3',b'-O0'),transformed):
        try:graph.seeded_graph(bad)
        except ValueError:pass
        else:raise AssertionError('Modified/unpinned graph accepted')
    with tempfile.TemporaryDirectory(prefix='ds41-seeded-graph-') as temp:
        path=Path(temp)/'build.ninja';path.write_bytes(transformed)
        argv=['ninja','-f',str(path),'-t','commands']
        if args.container_name:
            argv=['docker','run','--name',args.container_name,'--runtime=runc','--pull=never',
                  '--restart=no','--network=none','--memory=256m','--memory-swap=256m','--cpus=1',
                  '--pids-limit=32','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges',
                  '--user',f'{os.getuid()}:{os.getgid()}',
                  '--env=NVIDIA_VISIBLE_DEVICES=void','--env=CUDA_VISIBLE_DEVICES=',
                  '--mount',f'type=bind,src={path},dst=/graph.ninja,readonly',
                  '--entrypoint=/usr/bin/env',controller.IMAGES[0],
                  'ninja','-f','/graph.ninja','-t','commands']
        rows=subprocess.check_output(argv,text=True,timeout=30).splitlines()
        if args.container_name:
            state=json.loads(subprocess.check_output(['docker','inspect','--format','{{json .State}}',
                                                      args.container_name],text=True,timeout=10))
            assert not state['Running'] and not state['OOMKilled'] and state['ExitCode']==0
    commands=[shlex.split(row) for row in rows]
    assert len(commands)==12
    seeds=[]
    for argv in commands[:-1]:
        assert argv[0]=='/usr/local/cuda/bin/nvcc'
        seed=[x for x in argv if x.startswith('--frandom-seed=')]
        source=argv[argv.index('-c')+1]
        assert seed==['--frandom-seed='+source] and source.endswith('.cu')
        assert '$' not in seed[0]
        seeds.append(seed[0])
    assert len(set(seeds))==11
    assert not any('frandom-seed' in x for x in commands[-1])
    for host in (0,1):
        for stage in ('graph','compile'):
            plain=controller.create_command(host,'mxfp8-clean-bootstrap-v2',stage)
            seeded=controller.create_command(host,'mxfp8-clean-bootstrap-v2',stage,True)
            assert seeded==(plain+['--deterministic'] if stage=='graph' else plain)
    assert 'torch' not in sys.modules and 'flashinfer' not in sys.modules
    if args.container_name:
        report=root/'reports'/(args.container_name+'.json')
        with report.open('x') as stream:
            json.dump(dict(status='native_ninja_seed_expansion_pass',command=argv,state=state,
                           probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           graph_sha256=controller.SEEDED_GRAPH_SHA,seeds=seeds,
                           compiler_executed=False,gpu_execution_tested=False),stream,indent=2)
    print('PASS: seed-only graph transform,4 unpinned-graph refusals,11 unique actual Ninja-expanded input seeds, '
          'both hosts/stages retain CPU limits; no compilation/GPU or serving qualification')


if __name__=='__main__':main()
