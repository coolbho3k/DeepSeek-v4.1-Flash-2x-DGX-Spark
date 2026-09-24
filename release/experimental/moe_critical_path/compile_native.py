# SPDX-License-Identifier: AGPL-3.0-only
"""Compile the prepared plain-CUDA experiment with no GPU or network access."""
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
    if prepared['status']!='native_experiment_prepared' or prepared['experiment'] not in (101,102,201,301,401):
        raise ValueError('Unexpected experiment identity')
    if any(build.iterdir()):raise ValueError('Preserve earlier build results')
    if os.environ.get('NVIDIA_VISIBLE_DEVICES')!='void':raise ValueError('CPU-only build required')
    limits={name:(Path('/sys/fs/cgroup')/name).read_text().strip()
        for name in ('memory.max','memory.swap.max','cpu.max')}
    if limits['memory.max']!=str(768*2**20) or limits['memory.swap.max']!='0':
        raise ValueError('Use the bounded 768 MiB/no-swap build container')
    quota,period=map(int,limits['cpu.max'].split())
    if not 0<quota<=2*period:raise ValueError('Use at most two CPU cores')
    for name,digest in prepared['files'].items():
        path=source/name
        if path.resolve()!=path or not path.is_relative_to(source):raise ValueError('Unsafe source path')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Changed build source: '+name)
    command=['/usr/local/cuda/bin/nvcc',*prepared['cuda_flags'],'-I',str(source/'source/include'),
        str(source/'source/cooperative.cu'),'-o',str(build/'cooperative_moe.so')]
    begin=time.monotonic()
    with (build/'compiler.log').open('x') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
    if result.returncode:raise RuntimeError('Native compilation failed; see compiler.log')
    binary=build/'cooperative_moe.so';module=ctypes.CDLL(str(binary))
    if module.goal50_coop_abi()!=2 or module.goal50_coop_experiment()!=prepared['experiment']:raise ValueError('Changed compiled ABI')
    receipt=dict(status='native_experiment_built_cpu_only',variant=prepared['variant'],experiment=prepared['experiment'],abi=2,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),binary_bytes=binary.stat().st_size,
        prepared_sha256=hashlib.sha256((source/'prepared.json').read_bytes()).hexdigest(),
        source_files=prepared['files'],command=command,cgroup_limits=limits,
        elapsed_seconds=time.monotonic()-begin,gpu_qualified=False,serving_qualified=False,
        additional_persistent_gpu_bytes=0,license=prepared['license'])
    with (build/'complete.json').open('x') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
