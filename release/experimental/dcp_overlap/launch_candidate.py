# SPDX-License-Identifier: AGPL-3.0-only
"""Launch an explicitly pinned local candidate using the public lifecycle.

Only runtime-kit paths/pin change. Existing weights, caches, images, network,
KV allocation and safety boundaries are reused. Does not read environment
files, credentials or secrets; does not stop any workload. Experimental only.
"""
import argparse
import fcntl
import json
from pathlib import Path
import sys

from prepare import bounded_read, encoded, load_parent, sha

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'release'))
import launch
from config import load


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment', type=Path, required=True)
    p.add_argument('--deployment-sha256', required=True)
    p.add_argument('--kit', type=Path, required=True)
    p.add_argument('--kit-sha256', required=True)
    a = p.parse_args()
    raw = bounded_read(a.deployment.absolute())
    if sha(raw) != a.deployment_sha256:
        raise ValueError('Changed reference deployment')
    config = json.loads(raw)
    manifest, _ = load_parent(a.kit, a.kit_sha256)
    if manifest['parent_manifest_sha256'] != config['kit_manifest_sha256']:
        raise ValueError('Candidate is not based on the recorded parent')
    for node in config['nodes']:
        node['kit'] = str(a.kit.absolute())
    config['kit_manifest_sha256'] = a.kit_sha256
    raw = encoded(config)
    values = dict(WORKER_HOST=config['nodes'][1]['ssh'], FABRIC_NETWORK=config['fabric_network'],
        ROCE_GID_INDEX=str(config['nodes'][0]['gid_index']),
        API_HOST=config['api']['host'], API_PORT=str(config['api']['port']),
        MASTER_PORT=str(config['api']['master_port']), SERVED_MODEL_NAME=config['api']['model_name'],
        ALLOW_STARTUP_MEMORY_SHORTFALL=str(int(config['startup_memory_override'])))
    if config['serving']['kv_cap_mib'] != 0:
        raise ValueError('This launcher preserves the current display-only KV profile')
    values.update({k.upper(): str(v) for k, v in config['serving'].items() if k != 'kv_cap_mib'})
    for prefix, node in zip(('HEAD', 'WORKER'), config['nodes'], strict=True):
        for field in ('fabric_ip', 'ifname', 'hca', 'drm_card'):
            values[prefix + '_' + field.upper()] = str(node[field])
        if len(node['rails']) == 2:
            for suffix, key in (('IP', 'fabric_ip'), ('IFNAME', 'ifname'), ('HCA', 'hca')):
                values[prefix + '_SECONDARY_' + suffix] = node['rails'][1][key]
    launch.STATE.mkdir(parents=True, exist_ok=True)
    with (launch.STATE / 'operation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = launch.read_state()
        if state and launch.controller_alive(state):
            raise RuntimeError('Stop the existing owned server explicitly first')
        source = launch.STATE / ('candidate-' + a.kit_sha256[:16] + '.json')
        with source.open('xb') as out:
            out.write(raw)
        values.update(EXISTING_DEPLOYMENT=str(source), EXISTING_DEPLOYMENT_SHA256=sha(raw))
        settings = load(ROOT, environ=values)
        if settings['serving'] != config['serving'] or settings['api'] != config['api']:
            raise ValueError('Candidate launcher changed serving settings')
        path, candidate = launch.prepare_existing(settings)
        launch.start(path, candidate, no_wait=True)
        print(json.dumps(dict(deployment=str(path), candidate_manifest_sha256=a.kit_sha256,
                              configuration_unchanged=True, existing_environment_file_changed=False)))


if __name__ == '__main__':
    main()
