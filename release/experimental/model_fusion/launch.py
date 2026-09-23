# SPDX-License-Identifier: AGPL-3.0-only
"""Explicit campaign launcher: same settings, port 8889 unless requested otherwise.

Reuses the public ownership/lifecycle checks. Never reads .env or credentials,
downloads weights, changes networking, stops a worker or edits a live kit.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'release'))
import launch
from config import load

PORT=8889


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-deployment',type=Path,required=True)
    p.add_argument('--base-sha256',required=True)
    p.add_argument('--kit',type=Path)
    p.add_argument('--kit-sha256')
    p.add_argument('--port',type=int,choices=(8888,8889),default=PORT)
    a=p.parse_args()
    raw=a.base_deployment.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=a.base_sha256:raise ValueError('Changed baseline deployment')
    config=json.loads(raw);parent=config['kit_manifest_sha256']
    kit=a.kit.absolute() if a.kit else Path(config['nodes'][0]['kit'])
    digest=a.kit_sha256 if a.kit else parent
    if bool(a.kit)!=bool(a.kit_sha256):raise ValueError('Supply both candidate path and digest')
    verifier=launch.module(ROOT/'release/runtime/verify.py','model_fusion_verify')
    verifier.verify(kit,digest)
    manifest=json.loads((kit/'bundle-manifest.json').read_bytes())
    if digest!=parent and manifest.get('parent_manifest_sha256')!=parent:
        raise ValueError('Candidate must derive from the specified baseline')
    for node in config['nodes']:node['kit']=str(kit)
    config['kit_manifest_sha256']=digest
    config['api']['port']=a.port
    values=dict(WORKER_HOST=config['nodes'][1]['ssh'],FABRIC_NETWORK=config['fabric_network'],
        ROCE_GID_INDEX=str(config['nodes'][0]['gid_index']),API_HOST=config['api']['host'],
        API_PORT=str(a.port),MASTER_PORT=str(config['api']['master_port']),
        SERVED_MODEL_NAME=config['api']['model_name'],
        ALLOW_STARTUP_MEMORY_SHORTFALL=str(int(config['startup_memory_override'])))
    values.update({(('DS41_' if k in ('fp4_kv_mode','swa_kv_group_size') else '') + k.upper()):str(v)
                   for k,v in config['serving'].items() if k!='kv_cap_mib'})
    for prefix,node in zip(('HEAD','WORKER'),config['nodes'],strict=True):
        for field in ('fabric_ip','ifname','hca','drm_card'):values[prefix+'_'+field.upper()]=str(node[field])
        if len(node['rails'])==2:
            for suffix,key in (('IP','fabric_ip'),('IFNAME','ifname'),('HCA','hca')):
                values[prefix+'_SECONDARY_'+suffix]=node['rails'][1][key]
    raw=launch.encoded(config)
    with (launch.STATE/'operation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=launch.read_state()
        if state and launch.controller_alive(state):raise RuntimeError('Explicitly stop the current owned pair first')
        source=launch.STATE/f'model-fusion-{digest[:16]}-{hashlib.sha256(raw).hexdigest()[:12]}-port{a.port}.json'
        if source.exists():
            if source.read_bytes()!=raw:raise ValueError('Preserve changed campaign source')
        else:
            with source.open('xb') as out:out.write(raw)
        values.update(EXISTING_DEPLOYMENT=str(source),EXISTING_DEPLOYMENT_SHA256=hashlib.sha256(raw).hexdigest())
        settings=load(ROOT,environ=values)
        if settings['api']!=config['api'] or settings['serving']!=config['serving']:
            raise ValueError('Unexpected serving or API change')
        path,candidate=launch.prepare_existing(settings)
        launch.start(path,candidate,no_wait=True)
        print(json.dumps(dict(status='model_fusion_serving_started',deployment=str(path),port=a.port,
            deployment_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),kit_manifest_sha256=digest,
            serving_memory_settings_unchanged=True)),flush=True)


if __name__=='__main__':main()
