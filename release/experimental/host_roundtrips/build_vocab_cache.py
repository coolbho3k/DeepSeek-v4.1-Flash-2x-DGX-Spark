# SPDX-License-Identifier: AGPL-3.0-only
"""Build and CPU-check the cached vocabulary store inside the runtime image.

Same compiler flags and bounds as scripts/build_vocab_row_store_cpu.py:
512 MiB / no swap / one CPU, no network, no GPU devices, read-only sources.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
IMAGE = 'sha256:a5ef1cecb16259d16e49578c334b05f60dc58873eeac94cc4a29c5c246d0bcbf'
SOURCE = 'release/experimental/host_roundtrips/ds41_vocab_row_store.cpp'
CHECK = 'release/experimental/host_roundtrips/check_vocab_cache.py'
CHECKPOINT = ROOT / 'artifacts/ds41-exl3-3bpw-candidate-v1/model-00001-of-00051.safetensors'

INNER = f'''
set -eu
cp /work/{SOURCE} /results/ds41_vocab_row_store.cpp
g++ -std=c++17 -O2 -Wall -Wextra -Werror -shared -fPIC -pthread \\
    /results/ds41_vocab_row_store.cpp -o /results/libds41_vocab_rows.so
if readelf -d /results/libds41_vocab_rows.so | grep -Ei 'libcuda|libtorch|libcublas|libnvrtc'; then exit 3; fi
python3 -B /work/{CHECK} --checkpoint /checkpoint.safetensors \\
    --library /results/libds41_vocab_rows.so --output /results/cpu-parity.json
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.absolute()
    out.mkdir(parents=True)
    name = 'ds41-' + out.name
    command = ['docker', 'run', '--rm', '--name', name, '--runtime=runc', '--pull=never',
               '--network=none', '--read-only', '--memory=512m', '--memory-swap=512m', '--cpus=1',
               '--pids-limit=64', '--tmpfs=/tmp:rw,nosuid,nodev,size=67108864', '--user=1000:1000',
               '--cap-drop=ALL', '--security-opt=no-new-privileges', '--env=NVIDIA_VISIBLE_DEVICES=void',
               '--env=CUDA_VISIBLE_DEVICES=-1', '--env=PYTHONDONTWRITEBYTECODE=1',
               '--mount', f'type=bind,src={ROOT},dst=/work,readonly',
               '--mount', f'type=bind,src={CHECKPOINT},dst=/checkpoint.safetensors,readonly',
               '--mount', f'type=bind,src={out},dst=/results',
               '--entrypoint=/bin/sh', IMAGE, '-c', INNER]
    subprocess.run(command, check=True, timeout=900)
    binary = out / 'libds41_vocab_rows.so'
    parity = json.loads((out / 'cpu-parity.json').read_bytes())
    assert parity['status'] == 'vocab_cache_cpu_parity_pass' and parity['binary_sha256'] == sha(binary)
    receipt = dict(status='vocab_cache_build_and_cpu_parity_pass', image=IMAGE,
                   source_sha256=sha(ROOT / SOURCE), check_sha256=sha(ROOT / CHECK),
                   binary_sha256=sha(binary), binary_bytes=binary.stat().st_size,
                   parity_sha256=sha(out / 'cpu-parity.json'), gpu_qualified=False)
    (out / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
