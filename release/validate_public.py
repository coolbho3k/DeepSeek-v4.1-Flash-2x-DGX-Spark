# SPDX-License-Identifier: AGPL-3.0-only
"""Fail-closed publication check; optional small HTTP metadata requests only."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import urllib.request
import registry

ROOT=Path(__file__).resolve().parents[1]



# Engram tables of the packed page15 source (original weights, manifest 6d79a9ae); every later
# weight revision must carry these exact bytes.
ENGRAM_SOURCE_TABLES = {
    'engrams/engram-layer-01.safetensors': {'bytes': 101377628616, 'sha256': '9ecd9bc00c8c045104d639dfdd8d6de4fe0457ce05ad8885baab062d059b0c9e'},
    'engrams/engram-layer-14.safetensors': {'bytes': 101380404312, 'sha256': '0c33910e5a59c2787d1bc1891ca5961248e1db94e7919c562afbf4cedba2f5b2'},
}

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
    packed=lock['engram']
    if (not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',packed['repo'])
            or not re.fullmatch('[0-9a-f]{40}',packed['revision'])
            or packed['manifest_path']!='engram-page15-v1/manifest.json'):
        raise ValueError('Immutable packed Engram publication pin required')
    raw=(ROOT/'release/engram-release-manifest.json').read_bytes()
    module_spec=importlib.util.spec_from_file_location('validate_engram_pin',ROOT/'release/runtime/tools/engram_assets.py')
    helper=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(helper)
    manifest=helper.parse_manifest(raw)
    # The packed tables were built from the original weights (source_model). Later weight
    # revisions keep those engram tables byte for byte, so the pinned model manifest must
    # carry exactly the source's engram rows.
    model_raw=(ROOT/'release/model-release-manifest.json').read_bytes()
    model_rows={name:json.loads(model_raw)['files'].get(name) for name in ENGRAM_SOURCE_TABLES}
    if (hashlib.sha256(raw).hexdigest()!=packed['manifest_sha256']
            or helper.MANIFEST_SHA!=packed['manifest_sha256']
            or hashlib.sha256(model_raw).hexdigest()!=lock['model']['manifest_sha256']
            or manifest['source_model']['repo']!=lock['model']['repo'] or manifest['repo_id']!=packed['repo']
            or model_rows!=ENGRAM_SOURCE_TABLES):
        raise ValueError('Packed reader, data and canonical source pins do not agree')
    if runtime.get('transport') == 'ghcr':registry.validate(runtime)
    for name in (registry.ASSETS if runtime.get('transport') == 'ghcr' else ('runtime-image.tar.gz','kernel-cache.tar','runtime-source.tar.gz')):
        row=runtime['files'][name]
        if type(row['bytes']) is not int or row['bytes']<=0 or not re.fullmatch('[0-9a-f]{64}',row['sha256']):
            raise ValueError('Invalid runtime payload pin: '+name)
    if online:
        base=f"https://huggingface.co/{packed['repo']}/resolve/{packed['revision']}/"
        with urllib.request.urlopen(base+packed['manifest_path'],timeout=30) as response:
            remote=response.read(65537)
        if remote!=raw:raise ValueError('Public packed inventory differs')
        # Verify every referenced part exists at this immutable revision,
        # without downloading weights or using publisher authentication.
        url=f"https://huggingface.co/api/models/{packed['repo']}/revision/{packed['revision']}?blobs=true"
        with urllib.request.urlopen(url,timeout=30) as response:info=json.loads(response.read(2**20))
        files={row['rfilename']:row for row in info['siblings']}
        if info.get('private') or info['sha']!=packed['revision']:raise ValueError('Packed assets must be public and pinned')
        for table in manifest['files'].values():
            for part in table['parts']:
                remote=files.get(part['path'],{});lfs=remote.get('lfs',{})
                if remote.get('size')!=part['bytes'] or lfs.get('sha256')!=part['sha256']:
                    raise ValueError('Missing or mismatched immutable public packed part')
        source=manifest['source_model']
        url=f"https://huggingface.co/{source['repo']}/resolve/{source['revision']}/release-manifest.json"
        with urllib.request.urlopen(url,timeout=30) as response:source_raw=response.read(4*2**20+1)
        if (hashlib.sha256(source_raw).hexdigest()!=source['manifest_sha256']
                or {name:json.loads(source_raw)['files'].get(name) for name in ENGRAM_SOURCE_TABLES}!=ENGRAM_SOURCE_TABLES):
            raise ValueError('Packed Engram source tables differ from the pinned model')
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
