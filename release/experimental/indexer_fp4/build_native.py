# SPDX-License-Identifier: AGPL-3.0-only
"""GPU-free, bounded build of the NVFP4 prefill indexer kernel (sm_121a).

Run in the serving image with release/runtime/kernels at /input (read-only) and a fresh
/build; copy nvfp4_indexer.so and complete.json to release/runtime/serving/nvfp4-indexer-native/.
  docker run --rm --network none -e NVIDIA_VISIBLE_DEVICES=void --memory 2g --memory-swap 2g \
    --cpus 2 -v KERNELS:/input:ro -v OUT:/build -v THIS_DIR:/script:ro \
    --entrypoint /opt/ds41-venv/bin/python IMAGE /script/build_native.py
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

source, output = Path('/input/nvfp4_indexer.cu'), Path('/build')
if os.environ.get('NVIDIA_VISIBLE_DEVICES') != 'void' or any(output.iterdir()):
    raise ValueError('Fresh output and GPU-free compiler required')
command = ['/usr/local/cuda/bin/nvcc', '-std=c++17', '-O3', '-lineinfo',
           '-gencode', 'arch=compute_121a,code=sm_121a', '-shared', '-Xcompiler', '-fPIC',
           '--ptxas-options=-v', str(source), '-o', str(output / 'nvfp4_indexer.so')]
started = time.monotonic()
with (output / 'compiler.log').open('x') as stream:
    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=300, check=True)
binary = output / 'nvfp4_indexer.so'
if ctypes.CDLL(str(binary)).ds41_nvfp4_indexer_abi() != 2:
    raise ValueError('Unexpected native ABI')
receipt = dict(status='built_not_gpu_qualified', abi=2,
               binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), binary_bytes=binary.stat().st_size,
               source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), command=command,
               ptxas=[line.strip() for line in (output / 'compiler.log').read_text().splitlines()
                      if 'registers' in line or 'spill' in line],
               seconds=round(time.monotonic() - started, 2),
               build_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({k: receipt[k] for k in ('status', 'binary_sha256', 'ptxas', 'seconds')}), flush=True)
