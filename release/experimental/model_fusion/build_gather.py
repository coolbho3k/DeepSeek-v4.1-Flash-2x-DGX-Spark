# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only, resource-bounded build of the dual-gather experiment."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

source, output = Path('/input'), Path('/build')
if os.environ.get('NVIDIA_VISIBLE_DEVICES') != 'void' or any(output.iterdir()):
    raise ValueError('Fresh output and GPU-free compiler required')
limits = {key: (Path('/sys/fs/cgroup') / key).read_text().strip()
          for key in ('memory.max', 'memory.swap.max', 'cpu.max')}
quota, period = map(int, limits['cpu.max'].split())
if limits['memory.max'] != str(768 * 2**20) or limits['memory.swap.max'] != '0' or not 0 < quota <= 2 * period:
    raise ValueError('Require the bounded 768 MiB/two-core/no-swap compiler')
prepared = json.loads((source / 'prepared.json').read_bytes())
for name, expected in prepared['files'].items():
    path = source / name
    if path.resolve() != path or not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Changed compiler input')
command = ['/usr/local/cuda/bin/nvcc', '-std=c++17', '-O3', '--use_fast_math', '-lineinfo',
           '-gencode', 'arch=compute_121a,code=sm_121a', '-shared', '-Xcompiler', '-fPIC',
           '--ptxas-options=-v', '-I', str(source / 'include'),
           str(source / 'dual_gather.cu'), '-o', str(output / 'dual_gather.so')]
started = time.monotonic()
with (output / 'compiler.log').open('x') as stream:
    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=180, check=True)
binary = output / 'dual_gather.so'
module = ctypes.CDLL(str(binary))
if module.ds41_dual_gather_abi() != 1:
    raise ValueError('Unexpected native ABI')
receipt = dict(status='dual_gather_built_cpu_only', gpu_qualified=False, serving_qualified=False,
               binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), binary_bytes=binary.stat().st_size,
               source_sha256=prepared['files']['dual_gather.cu'],
               prepared_sha256=hashlib.sha256((source / 'prepared.json').read_bytes()).hexdigest(),
               source_files=prepared['files'], command=command, cgroup_limits=limits,
               elapsed_seconds=time.monotonic() - started,
               build_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({key: receipt[key] for key in ('status', 'binary_sha256', 'binary_bytes', 'elapsed_seconds')}), flush=True)
