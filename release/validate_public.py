# SPDX-License-Identifier: AGPL-3.0-only
"""Fail-closed publication check; optional small HTTP metadata requests only."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import registry

ROOT=Path(__file__).resolve().parents[1]


def validate(lock,online=False):
    runtime=lock.get('runtime')
    if runtime is None:
        raise ValueError('NOT READY FOR PUBLIC LAUNCH: GHCR runtime build/publication and immutable digest pin are incomplete. Weight downloads are public; no runtime URL is invented.')
    if not re.fullmatch('[0-9a-f]{64}',lock.get('kit_manifest_sha256','')):
        raise ValueError('Missing runtime source manifest pin')
    if hashlib.sha256((ROOT/'release/runtime/bundle-manifest.json').read_bytes()).hexdigest()!=lock['kit_manifest_sha256']:
        raise ValueError('Runtime source manifest does not match lock')
    for spec in (lock['model'],lock['draft']) + (() if runtime.get('transport') == 'ghcr' else (runtime,)):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',spec['repo']) or not re.fullmatch('[0-9a-f]{40}',spec['revision']):
            raise ValueError('Repository and immutable revision required')
    if runtime.get('transport') == 'ghcr':registry.validate(runtime)
    for name in (registry.ASSETS if runtime.get('transport') == 'ghcr' else ('runtime-image.tar.gz','kernel-cache.tar','runtime-source.tar.gz')):
        row=runtime['files'][name]
        if type(row['bytes']) is not int or row['bytes']<=0 or not re.fullmatch('[0-9a-f]{64}',row['sha256']):
            raise ValueError('Invalid runtime payload pin: '+name)
    if online:
        for name in ('model','draft'):
            spec=lock[name]
            url=f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/release-manifest.json"
            with urllib.request.urlopen(url,timeout=30) as response:raw=response.read(4*2**20+1)
            if hashlib.sha256(raw).hexdigest()!=spec['manifest_sha256']:
                raise ValueError(name+' public manifest hash mismatch')
        if runtime.get('transport') == 'ghcr':
            registry.require_public(runtime)
        else:
            url=f"https://huggingface.co/datasets/{runtime['repo']}/resolve/{runtime['revision']}/assets.json"
            with urllib.request.urlopen(url,timeout=30) as response:assets=json.loads(response.read(2**20))
            if assets['files']!=runtime['files'] or assets['cache_manifest_sha256']!=runtime['cache_manifest_sha256']:
                raise ValueError('Published runtime inventory differs from lock')
    return dict(status='asset_pins_valid',online_metadata_checked=online,anonymous_public_access_verified=online,gpu_tested=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--online',action='store_true');a=p.parse_args()
    try:print(json.dumps(validate(json.loads((ROOT/'recipe-lock.json').read_bytes()),a.online)))
    except (ValueError,KeyError,OSError) as e:p.exit(1,str(e)+'\n')
