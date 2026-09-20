# SPDX-License-Identifier: AGPL-3.0-only
"""Refresh reviewed source pins and make a new immutable private test kit.

Never edits a mounted kit, launches a process, accesses weights or reads secrets.
Private Engram asset bindings are retained; public bindings remain portable.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'release/experimental/dcp_overlap'))
from prepare import load_parent, bounded_read, encoded, sha, safe_name

REPLACEMENTS = ('ds41/vllm_prompt.py', 'serving/ds41/vllm_prompt.py',
                'serving/ds41/launch_profile.py', 'tools/launch_profile.py')
REVIEWED = (*REPLACEMENTS, 'tools/portable_node.py')
PROVENANCE = dict(upstream='MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks',
    authors=['MiaAI Lab / Wesley Young', 'mrexodia'], license='AGPL-3.0-only',
    commits=['a2c8d28a2355f193c4008430e07061293fb47a6b',
             '14ebc3937d4cef76c3f7369607df817f703e21a1',
             '0b5654dfbbfb8aa88c7c4d70e21104403e1fb236'],
    changes=['Responses input_text/output_text normalization', 'prompt token details',
             'configurable periodic prefix retention; default4096, rollback0'],
    native_kernels_unchanged=True, weights_unchanged=True,
    gpu_allocation_and_safety_boundaries_unchanged=True,
    inherited_qualification_applies_to_parent_only=True)


def repin(payload, old_hashes):
    payload = dict(payload)
    history = {n:{h} for n,h in old_hashes.items() if n.endswith('.py')}
    all_updates = {}
    for _ in range(32):
        for name, values in history.items():
            values.add(sha(payload[name]))
        updates = {old:sha(payload[name]) for name,values in history.items()
                   for old in values if old != sha(payload[name])}
        all_updates.update(updates)
        changed = False
        for name, raw in list(payload.items()):
            if not name.endswith('.py') or not name.startswith(('serving/', 'tools/', 'ds41/')):
                continue
            for old,new in updates.items():
                raw = raw.replace(old.encode(), new.encode())
            if raw != payload[name]:
                payload[name], changed = raw, True
        if not changed:
            break
    else:
        raise ValueError('Source pin cycle')
    def refresh(value):
        if isinstance(value, dict):
            return {k:refresh(v) for k,v in value.items()}
        if isinstance(value, list):
            return [refresh(v) for v in value]
        return all_updates.get(value, value) if isinstance(value, str) else value
    requirements = refresh(json.loads(payload['runtime-requirements.json']))
    requirements['upstream_api_cache'] = PROVENANCE
    payload['runtime-requirements.json'] = encoded(requirements)
    payload['serving/overlay-manifest.json'] = encoded({n.removeprefix('serving/'):sha(raw)
        for n,raw in payload.items() if n.startswith('serving/') and n != 'serving/overlay-manifest.json'})
    for n,raw in payload.items():
        if n.endswith('.py'):
            compile(raw, n, 'exec')
    return payload


def refresh_public():
    kit = ROOT/'release/runtime'
    manifest = json.loads(bounded_read(kit/'bundle-manifest.json'))
    payload = {n:bounded_read(kit/safe_name(n)) for n in manifest['files']}
    changed = {n for n,raw in payload.items() if sha(raw) != manifest['files'][n]['sha256']}
    if changed != set(REVIEWED):
        raise ValueError('Unexpected public edits before refresh: '+repr(changed))
    updated = repin(payload, {n:r['sha256'] for n,r in manifest['files'].items()})
    # Mechanical regeneration only; source edits were applied separately.
    for n,raw in updated.items():
        if raw != payload[n]:
            (kit/n).write_bytes(raw)
    sys.path.insert(0, str(ROOT/'release'))
    from freeze import freeze
    freeze()


def candidate(parent, digest, output):
    manifest, source = load_parent(parent, digest)
    if output.exists() or output.resolve() != output or not output.parent.is_dir():
        raise ValueError('Use a fresh canonical candidate directory')
    payload = dict(source)
    for name in REPLACEMENTS:
        payload[name] = bounded_read(ROOT/'release/runtime'/name)
    name = 'tools/portable_node.py'
    old = "'--reasoning-parser','deepseek_v41']"
    new = "'--reasoning-parser','deepseek_v41','--enable-prompt-tokens-details']"
    if payload[name].decode().count(old) != 1:
        raise ValueError('Unexpected private launcher anchor')
    payload[name] = payload[name].decode().replace(old, new).encode()
    payload = repin(payload, {n:sha(raw) for n,raw in source.items()})
    output.mkdir(mode=0o700)
    for name,raw in payload.items():
        path = output/safe_name(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            handle.write(raw)
    result = dict(format=manifest['format'], standalone_runtime=False,
        clean_rebuild_qualified=False, publication_approved=False, serving_qualified=False,
        variant='miaai_api_cache_v1', parent_manifest_sha256=digest,
        files={n:dict(bytes=len(raw), sha256=sha(raw)) for n,raw in sorted(payload.items())})
    raw = encoded(result)
    with (output/'bundle-manifest.json').open('xb') as handle:
        handle.write(raw)
    load_parent(output, sha(raw))
    print(json.dumps(dict(candidate=str(output), manifest_sha256=sha(raw))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-public', action='store_true')
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--parent-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.refresh_public:
        refresh_public()
    else:
        if not all((args.parent, args.parent_sha256, args.output)):
            parser.error('candidate requires parent, parent-sha256 and output')
        candidate(args.parent, args.parent_sha256, args.output)
