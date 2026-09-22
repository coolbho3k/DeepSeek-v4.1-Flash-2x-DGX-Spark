# SPDX-License-Identifier: AGPL-3.0-only
"""Read-only driver consistency check, not a version allowlist or CUDA probe."""
import json
from pathlib import Path
import re
import subprocess
import sys

# Observations only: 595.84 passed a local mixed-driver serving canary, not
# every firmware/kernel combination. Other versions are allowed with a warning.
OBSERVED_DRIVERS = frozenset(('580.173.02', '595.84'))


def observe():
    # Check the running kernel module, not an installed package or modinfo,
    # which can describe a different driver awaiting a reboot.
    loaded = Path('/sys/module/nvidia/version').read_text().strip()
    reported = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
        text=True, stderr=subprocess.PIPE, timeout=15)
    return dict(loaded=loaded, reported=reported.strip().splitlines())


def validate(sample, host):
    loaded = sample.get('loaded') if isinstance(sample, dict) else None
    reported = sample.get('reported') if isinstance(sample, dict) else None
    if (not isinstance(loaded, str) or not re.fullmatch(r'[0-9]+(?:\.[0-9]+){1,3}', loaded)
            or reported != [loaded]):
        raise ValueError(
            f'{host}: NVIDIA loaded-driver/NVML mismatch or incomplete observation; '
            f'loaded={loaded!r}, nvidia-smi={reported!r}. '
            'Check that the running kernel module and NVIDIA userspace match. '
            'No driver version is required by this recipe; see docs/display-memory.md.')
    if loaded not in OBSERVED_DRIVERS:
        print(f'Warning: {host}: driver {loaded} has no local display-KV serving evidence; '
              'continuing. Compatibility also depends on kernel/firmware and the DRM mapping. '
              'See docs/display-memory.md.', file=sys.stderr)
    return sample


if __name__ == '__main__':
    print(json.dumps(observe()))
