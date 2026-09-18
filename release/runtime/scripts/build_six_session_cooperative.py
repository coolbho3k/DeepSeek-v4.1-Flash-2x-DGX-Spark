# SPDX-License-Identifier: AGPL-3.0-only
"""Build the attributed MiaAI cooperative specialization for 24 physical rows.

Only capacity constants change in the native kernels. The eight-slot MMA tile,
quantization, reductions, and completion protocol remain the upstream ones.
Generated copies are private build artifacts; pinned vendor files stay intact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def sha(data):return hashlib.sha256(data).hexdigest()
def once(source,old,new):
    if source.count(old)!=1:raise ValueError('Changed native anchor: '+old)
    return source.replace(old,new)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/work'))
    p.add_argument('--output',type=Path,default=Path('/build'))
    a=p.parse_args();root=a.root;out=a.output
    if list(out.iterdir()):raise ValueError('Fresh build output required')
    vendor=root/'vendor/miaai-cooperative-moe-agpl'
    deps=root/'vendor/miaai-cooperative-dependencies-agpl'
    for directory,key in ((vendor,'local_sha256'),(deps,'sha256')):
        receipt=json.loads((directory/'UPSTREAM.json').read_bytes())
        for name,row in receipt['files'].items():
            if sha((directory/name).read_bytes())!=row[key]:raise ValueError('Source changed: '+name)
    ext=out/'upstream/exllamav3/exllamav3_ext'
    shutil.copytree(deps/'exllamav3/exllamav3_ext',ext)
    native=vendor/'extensions/cooperative_moe/native'
    source=(native/'cooperative_moe.cu').read_text()
    source=once(source,'ROWS_MAX = 8;','ROWS_MAX = 24;')
    source=once(source,'goal50_coop_abi() { return 1; }','goal50_coop_abi() { return 2; }')
    source=once(source,'capacity 48 routed slots and 8 output rows','capacity 144 routed slots and 24 output rows')
    kernel=(native/'cooperative_moe_kernel.cuh').read_text()
    kernel=once(kernel,'p.slots_max = 48; p.rows_max = 8;','p.slots_max = 144; p.rows_max = 24;')
    kernel=once(kernel,'p.ctr_a_len = 432; p.ctr_b_len = 320;','p.ctr_a_len = 1296; p.ctr_b_len = 960;')
    note='// Local DS41 C6 adaptation: 24 physical rows / 144 routed slots, ABI 2.\n'
    (out/'goal50_fixed_coop.cu').write_text(note+source)
    (ext/'quant/goal50_fixed_coop_kernel.cuh').write_text(note+kernel)
    shutil.copyfile(native/'exl3_moe_coop.cuh',ext/'quant/exl3_moe_coop.cuh')
    command=['/usr/local/cuda/bin/nvcc','-std=c++17','-O3','--use_fast_math','-lineinfo',
        '--expt-relaxed-constexpr','-gencode','arch=compute_121a,code=sm_121a','-shared',
        '-Xcompiler','-fPIC','--ptxas-options=-v','-I',str(ext),str(out/'goal50_fixed_coop.cu'),
        '-o',str(out/'cooperative_moe.so')]
    started=time.monotonic()
    with (out/'compiler.log').open('x') as stream:
        subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=900)
    result=dict(status='cooperative_moe_built_cpu_only',abi=2,maximum_rows=24,
        routed_slots=144,counter_elements=2547,additional_persistent_gpu_bytes=0,
        upstream_commit='b9c49e90bdcc6f1e0192feb57214df11b67d36aa',license='AGPL-3.0-only',
        gpu_qualified=False,binary_sha256=sha((out/'cooperative_moe.so').read_bytes()),
        source_sha256=sha((out/'goal50_fixed_coop.cu').read_bytes()),
        kernel_sha256=sha((ext/'quant/goal50_fixed_coop_kernel.cuh').read_bytes()),
        elapsed_seconds=time.monotonic()-started,command=command)
    (out/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
