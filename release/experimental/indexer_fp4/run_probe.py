# SPDX-License-Identifier: AGPL-3.0-only
"""Run one serial serving probe against a started deployment's own API port (in-process).

The probes in probes/ default to the portable stack's 127.0.0.1:8041. model_fusion/benchmark.py
re-points them at the deployment's port the same way; this does it for the generation diagnostics
and long-context retrieval. The launcher's controller keeps its continuous RAM watch meanwhile.
  run_probe.py --deployment D --deployment-sha256 SHA --probe generation --output reports/X.json
  run_probe.py --deployment D --deployment-sha256 SHA --probe long-context --target-tokens N --output reports/Y.json
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
MODULES = {'generation': 'check_serving_generation', 'long-context': 'check_serving_long_context'}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--deployment', type=Path, required=True)
    p.add_argument('--deployment-sha256', required=True)
    p.add_argument('--probe', choices=tuple(MODULES), required=True)
    p.add_argument('--target-tokens', type=int)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    raw = a.deployment.read_bytes()
    if hashlib.sha256(raw).hexdigest() != a.deployment_sha256:
        raise ValueError('Changed deployment')
    if (a.probe == 'long-context') != (a.target_tokens is not None):
        raise ValueError('--target-tokens is required for, and only for, long-context')
    port = json.loads(raw)['api']['port']
    sys.path.insert(0, str(ROOT / 'probes'))
    module = importlib.import_module(MODULES[a.probe])
    module.BASE = f'http://127.0.0.1:{port}'
    argv = [module.__file__, '--output', str(a.output.absolute())]
    if a.target_tokens is not None:
        argv += ['--target-tokens', str(a.target_tokens)]
    sys.argv = argv
    module.main()


if __name__ == '__main__':
    main()
