# SPDX-License-Identifier: AGPL-3.0-only
"""Read-only loaded-driver qualification; never initializes CUDA or changes a host."""
import json
from pathlib import Path
import subprocess

QUALIFIED_DRIVER = '580.173.02'


def observe():
    # Check the running kernel module, not an installed package or modinfo,
    # which can describe a different driver awaiting a reboot.
    loaded = Path('/sys/module/nvidia/version').read_text().strip()
    reported = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
        text=True, stderr=subprocess.PIPE, timeout=15)
    return dict(loaded=loaded, reported=reported.strip().splitlines())


def validate(sample, host):
    if (not isinstance(sample, dict) or sample.get('loaded') != QUALIFIED_DRIVER
            or sample.get('reported') != [QUALIFIED_DRIVER]):
        loaded = sample.get('loaded', 'unknown') if isinstance(sample, dict) else 'unknown'
        reported = sample.get('reported', []) if isinstance(sample, dict) else []
        raise ValueError(
            f'{host}: display-KV requires the qualified loaded driver {QUALIFIED_DRIVER}; '
            f'loaded={loaded!r}, nvidia-smi={reported!r}. '
            '595.84 was reported to fail CUDA registration of the DRM buffer '
            '(register display IO: CUDA_ERROR_INVALID_VALUE). Other versions are '
            'unqualified, not necessarily broken. No driver settings were changed; '
            'see docs/display-memory.md before changing an idle host.')
    return sample


if __name__ == '__main__':
    print(json.dumps(observe()))
