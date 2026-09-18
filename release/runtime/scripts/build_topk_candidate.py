# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only native build, no serving replacement or GPU access."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

def main():
    out=Path('/build');source=Path('/work/kernels/length_aware_topk.cu')
    if list(out.iterdir()):raise ValueError('Fresh build directory required')
    command=['/usr/local/cuda/bin/nvcc','-std=c++17','-O3','-lineinfo',
        '--expt-relaxed-constexpr','-gencode','arch=compute_121a,code=sm_121a',
        '-shared','-Xcompiler','-fPIC','--ptxas-options=-v',str(source),'-o',str(out/'topk.so')]
    started=time.monotonic()
    with (out/'compiler.log').open('x') as log:
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    result=dict(status='built_not_gpu_qualified',command=command,seconds=time.monotonic()-started,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256((out/'topk.so').read_bytes()).hexdigest())
    (out/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
