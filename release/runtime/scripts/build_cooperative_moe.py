# SPDX-License-Identifier: AGPL-3.0-only
"""Compile pinned MiaAI kernels with no GPU access; emit an unqualified binary."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/work')
OUT = Path('/build')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    if list(OUT.iterdir()):
        raise ValueError('Build directory must be empty')
    vendor = ROOT/'vendor/miaai-cooperative-moe-agpl'
    deps = ROOT/'vendor/miaai-cooperative-dependencies-agpl'
    identities = {}
    for directory, key in ((vendor, 'local_sha256'), (deps, 'sha256')):
        receipt = json.loads((directory/'UPSTREAM.json').read_bytes())
        identities[directory.name] = sha((directory/'UPSTREAM.json').read_bytes())
        for name, row in receipt['files'].items():
            if sha((directory/name).read_bytes()) != row[key]:
                raise ValueError('Pinned source changed: '+name)
    ext = OUT/'upstream/exllamav3/exllamav3_ext'
    shutil.copytree(deps/'exllamav3/exllamav3_ext', ext)
    native = vendor/'extensions/cooperative_moe/native'
    shutil.copyfile(native/'cooperative_moe.cu', OUT/'goal50_fixed_coop.cu')
    shutil.copyfile(native/'cooperative_moe_kernel.cuh', ext/'quant/goal50_fixed_coop_kernel.cuh')
    shutil.copyfile(native/'exl3_moe_coop.cuh', ext/'quant/exl3_moe_coop.cuh')
    command = ['/usr/local/cuda/bin/nvcc', '-std=c++17', '-O3', '--use_fast_math', '-lineinfo',
        '--expt-relaxed-constexpr', '-gencode', 'arch=compute_121a,code=sm_121a', '-shared',
        '-Xcompiler', '-fPIC', '--ptxas-options=-v', '-I', str(ext), str(OUT/'goal50_fixed_coop.cu'),
        '-o', str(OUT/'cooperative_moe.so')]
    start = time.monotonic()
    compiler = subprocess.check_output([command[0], '--version'], text=True)
    with (OUT/'compiler.log').open('x') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=900)
    result = dict(status='cooperative_moe_built_cpu_only', gpu_qualified=False, abi=1,
        upstream_commit='b9c49e90bdcc6f1e0192feb57214df11b67d36aa',
        license='AGPL-3.0-only', binary_sha256=sha((OUT/'cooperative_moe.so').read_bytes()),
        compiler=compiler, command=command, source_manifests=identities,
        elapsed_seconds=time.monotonic()-start, additional_persistent_gpu_bytes=0,
        numerical_parity_pending=True, performance_measurement_pending=True)
    (OUT/'complete.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
