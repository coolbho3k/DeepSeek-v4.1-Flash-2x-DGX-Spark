# SPDX-License-Identifier: AGPL-3.0-only
"""One local mixed-driver experiment, retaining the saved recipe and lifecycle.

No environment-file or credential access, public policy edits, driver changes,
downloads, memory-limit changes, or automatic stops/restarts. Explicit execute
is required. The exception accepts only head580.173.02 + worker595.84.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'release'))
import launch
from config import load


def check_drivers(config):
    source = (ROOT / 'release/runtime/tools/display_driver.py').read_bytes()
    samples = []
    for node, expected in zip(config['nodes'], ('580.173.02', '595.84'), strict=True):
        sample = launch.json_run(['python3', '-B', '-'], node['ssh'], input=source, timeout=30)
        if sample != dict(loaded=expected, reported=[expected]):
            raise ValueError('Mixed-driver test requires precisely head580.173.02 + worker595.84')
        samples.append(sample)
    print(json.dumps(dict(experimental_mixed_drivers=samples, public_driver_policy_unchanged=True)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment', type=Path, required=True)
    p.add_argument('--deployment-sha256', required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    path = a.deployment.absolute()
    if path.resolve() != path or not path.is_file() or path.stat().st_size > 65536:
        raise ValueError('Expected a small unredirected saved deployment')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != a.deployment_sha256:
        raise ValueError('Reference deployment changed')
    config = json.loads(raw)
    values = dict(WORKER_HOST=config['nodes'][1]['ssh'], FABRIC_NETWORK=config['fabric_network'],
        ROCE_GID_INDEX=str(config['nodes'][0]['gid_index']),
        API_HOST=config['api']['host'], API_PORT=str(config['api']['port']),
        MASTER_PORT=str(config['api']['master_port']), SERVED_MODEL_NAME=config['api']['model_name'],
        ALLOW_STARTUP_MEMORY_SHORTFALL=str(int(config['startup_memory_override'])),
        EXISTING_DEPLOYMENT=str(path), EXISTING_DEPLOYMENT_SHA256=a.deployment_sha256)
    if config['serving']['kv_cap_mib'] != 0:
        raise ValueError('Keep the existing display-only KV budget')
    values.update({k.upper():str(v) for k,v in config['serving'].items() if k != 'kv_cap_mib'})
    for prefix, node in zip(('HEAD', 'WORKER'), config['nodes'], strict=True):
        for field in ('fabric_ip', 'ifname', 'hca', 'drm_card'):
            values[prefix+'_'+field.upper()] = str(node[field])
        if len(node['rails']) == 2:
            for suffix, key in (('IP', 'fabric_ip'), ('IFNAME', 'ifname'), ('HCA', 'hca')):
                values[prefix+'_SECONDARY_'+suffix] = node['rails'][1][key]
    settings = load(ROOT, environ=values)
    if settings['serving'] != config['serving'] or settings['api'] != config['api']:
        raise ValueError('Test launcher changed serving settings')
    check_drivers(config)
    if not a.execute:
        print(json.dumps(dict(status='plan_only', serving=config['serving'], api=config['api'])))
        return
    launch.STATE.mkdir(parents=True, exist_ok=True)
    with (launch.STATE/'operation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = launch.read_state()
        if state and launch.controller_alive(state):
            raise RuntimeError('An owned server is already running; no automatic stop')
        prepared, candidate = launch.prepare_existing(settings)
        check_drivers(candidate)
        if candidate['serving'] != config['serving'] or candidate['api'] != config['api']:
            raise ValueError('Prepared serving settings changed')
        launch.start(prepared, candidate, no_wait=True)
        print(json.dumps(dict(deployment=str(prepared), configuration_unchanged=True,
            mixed_driver_experiment=True, public_driver_policy_unchanged=True)))


if __name__ == '__main__':
    main()
