# SPDX-License-Identifier: AGPL-3.0-only
"""Prepare corresponding sources for an isolated decode native experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from fused_gateup import fragment,transform

ROOT=Path(__file__).resolve().parents[3]


def sha(raw):return hashlib.sha256(raw).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parallel-gateup',action='store_true')
    parser.add_argument('--persistent-pipeline',action='store_true')
    parser.add_argument('--persistent-resident-blocks',type=int,choices=(1,2),default=1)
    parser.add_argument('--dequant-pipeline',action='store_true')
    parser.add_argument('--speculative-capacity',action='store_true',help='Unchanged target math, 36 rows for K5/C6')
    args=parser.parse_args();out=args.output.absolute()
    if sum((args.parallel_gateup,args.persistent_pipeline,args.dequant_pipeline,args.speculative_capacity))>1:raise ValueError('One native experiment at a time')
    if args.persistent_resident_blocks!=1 and not args.persistent_pipeline:raise ValueError('Occupancy option requires the persistent experiment')
    if out.resolve()!=out or out.exists():raise ValueError('Use a fresh non-symlink output directory')
    runtime=ROOT/'release/runtime';deps=runtime/'vendor/miaai-cooperative-dependencies-agpl'
    mia=runtime/'vendor/miaai-cooperative-moe-agpl';native=mia/'extensions/cooperative_moe/native'
    files={};parents={}
    for directory,key in ((deps,'sha256'),(mia,'local_sha256')):
        manifest=json.loads((directory/'UPSTREAM.json').read_bytes())
        for name,row in manifest['files'].items():
            data=(directory/name).read_bytes()
            if sha(data)!=row[key]:raise ValueError('Changed vendored source: '+name)
        parents[directory.name]=sha((directory/'UPSTREAM.json').read_bytes())
    for path in (deps/'exllamav3/exllamav3_ext').rglob('*'):
        if path.is_file():files['source/include/'+path.relative_to(deps/'exllamav3/exllamav3_ext').as_posix()]=path.read_bytes()
    kernel=(native/'cooperative_moe_kernel.cuh').read_text()
    for before,after in (('p.slots_max = 48; p.rows_max = 8;','p.slots_max = 144; p.rows_max = 24;'),
                         ('p.ctr_a_len = 432; p.ctr_b_len = 320;','p.ctr_a_len = 1296; p.ctr_b_len = 960;')):
        if kernel.count(before)!=1:raise ValueError('Changed C6 parent adaptation')
        kernel=kernel.replace(before,after)
    kernel='// Local DS41 C6 adaptation: 24 physical rows / 144 routed slots, ABI 2.\n'+kernel
    extra={}
    if args.speculative_capacity:
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'dspark'))
        from native_capacity import transform as capacity_transform
        wrapper,kernel=capacity_transform((runtime/'sources/cooperative24.cu').read_bytes(),kernel.encode())
        gateup=None
        extra={'native_capacity.py':Path(__file__).resolve().parent.parent.joinpath('dspark/native_capacity.py').read_bytes()}
    elif args.dequant_pipeline:
        from dequant_pipeline import transform as dequant_transform
        wrapper,kernel,pipeline=dequant_transform((runtime/'sources/cooperative24.cu').read_bytes(),kernel.encode())
        gateup=None
        extra={'source/include/quant/dequant_pipeline.cuh':pipeline,
            'dequant_pipeline.py':Path(__file__).with_name('dequant_pipeline.py').read_bytes()}
    elif args.persistent_pipeline:
        from persistent_pipeline import transform as persistent_transform
        wrapper,kernel,gateup,queue=persistent_transform((runtime/'sources/cooperative24.cu').read_bytes(),kernel.encode(),args.persistent_resident_blocks)
        extra={'source/include/quant/persistent_pipeline.cuh':queue,
            'persistent_pipeline.py':Path(__file__).with_name('persistent_pipeline.py').read_bytes()}
    else:
        wrapper,kernel=transform((runtime/'sources/cooperative24.cu').read_bytes(),kernel.encode(),args.parallel_gateup)
        gateup=fragment(args.parallel_gateup)
    files.update({'source/cooperative.cu':wrapper,'source/include/quant/goal50_fixed_coop_kernel.cuh':kernel,
        'source/include/quant/exl3_moe_coop.cuh':(native/'exl3_moe_coop.cuh').read_bytes(),
        'LICENSE':(mia/'LICENSE').read_bytes(),'LICENSE.MIT':(mia/'LICENSE.MIT').read_bytes(),
        'LICENSE.exllamav3':(native/'LICENSE.exllamav3').read_bytes(),
        'compile_native.py':Path(__file__).with_name('compile_native.py').read_bytes(),
        'prepare_native.py':Path(__file__).read_bytes(),'fused_gateup.py':Path(__file__).with_name('fused_gateup.py').read_bytes()})
    files.update(extra)
    if gateup is not None:files['source/include/quant/fused_gateup.cuh']=gateup
    if sum(map(len,files.values()))>4*2**20:raise ValueError('Unexpected native source size')
    out.mkdir(parents=True)
    for name,data in files.items():
        path=out/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(data)
    receipt=dict(status='native_experiment_prepared',
        variant='speculative_capacity36' if args.speculative_capacity else 'dequant_pipeline' if args.dequant_pipeline else ('persistent_pipeline_two_blocks' if args.persistent_resident_blocks==2 else 'persistent_pipeline') if args.persistent_pipeline else 'fused_gateup_parallel' if args.parallel_gateup else 'fused_gateup',
        experiment=401 if args.speculative_capacity else 301 if args.dequant_pipeline else 201 if args.persistent_pipeline else 102 if args.parallel_gateup else 101,abi=2,
        license='AGPL-3.0-only',upstream_commit='b9c49e90bdcc6f1e0192feb57214df11b67d36aa',
        parent_manifests=parents,files={name:sha(data) for name,data in sorted(files.items())},
        additional_persistent_gpu_bytes=0,serving_qualified=False,
        cuda_flags=['-std=c++17','-O3','--use_fast_math','-lineinfo','--expt-relaxed-constexpr',
            '-gencode','arch=compute_121a,code=sm_121a','-shared','-Xcompiler','-fPIC','--ptxas-options=-v'])
    data=(json.dumps(receipt,indent=2,sort_keys=True)+'\n').encode()
    with (out/'prepared.json').open('xb') as stream:stream.write(data)
    print(json.dumps(dict(status=receipt['status'],output=str(out),prepared_sha256=sha(data))))


if __name__=='__main__':main()
