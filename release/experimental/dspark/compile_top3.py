# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only nvcc build inside the pinned 768MiB/no-swap/no-network container."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    source=Path('/input');build=Path('/build')
    prepared=json.loads((source/'prepared.json').read_bytes())
    if prepared['status']!='draft_top3_prepared' or prepared['abi']!=1:raise ValueError('Wrong source')
    if any(build.iterdir()) or os.environ.get('NVIDIA_VISIBLE_DEVICES')!='void':raise ValueError('Fresh CPU-only build required')
    limits={k:(Path('/sys/fs/cgroup')/k).read_text().strip() for k in ('memory.max','memory.swap.max','cpu.max')}
    if limits['memory.max']!=str(768*2**20) or limits['memory.swap.max']!='0':raise ValueError('Changed build limits')
    quota,period=map(int,limits['cpu.max'].split())
    if not 0<quota<=2*period:raise ValueError('At most two CPU cores')
    for name,digest in prepared['files'].items():
        path=source/name
        if path.resolve()!=path or not path.is_relative_to(source):raise ValueError('Unsafe path')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Changed source')
    binary=build/'dspark_draft_top3.so'
    command=['/usr/local/cuda/bin/nvcc','-std=c++17','-O3','--use_fast_math','-lineinfo',
        '--expt-relaxed-constexpr','-gencode','arch=compute_121a,code=sm_121a',
        '-shared','-Xcompiler','-fPIC','--ptxas-options=-v','-I','/input/source/include',
        '/input/source/draft_top3.cu','-o',str(binary)]
    started=time.monotonic()
    with (build/'compiler.log').open('x') as f:result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=300)
    if result.returncode:raise RuntimeError('See preserved compiler.log')
    library=ctypes.CDLL(str(binary))
    if library.ds41_draft_top3_abi()!=1:raise ValueError('Changed compiled ABI')
    record=dict(status='draft_top3_built_cpu_only',abi=1,command=command,
        prepared_sha256=hashlib.sha256((source/'prepared.json').read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),binary_bytes=binary.stat().st_size,
        source_files=prepared['files'],elapsed_seconds=time.monotonic()-started,cgroup_limits=limits,
        gpu_qualified=False,serving_qualified=False)
    with (build/'complete.json').open('x') as f:json.dump(record,f,indent=2)
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
