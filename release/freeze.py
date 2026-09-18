# SPDX-License-Identifier: AGPL-3.0-only
"""Maintainer-only manifest refresh. Never invoked by the end-user launcher."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def encoded(value):return (json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
def freeze():
    kit=ROOT/'release/runtime'
    manifest=json.loads((kit/'bundle-manifest.json').read_bytes())
    manifest['files']={}
    for path in sorted(kit.rglob('*')):
        if path.is_symlink():raise ValueError('No redirected release files')
        if not path.is_file() or path.name=='bundle-manifest.json':continue
        if '__pycache__' in path.parts:raise ValueError('Remove generated Python bytecode before freezing')
        raw=path.read_bytes()
        manifest['files'][path.relative_to(kit).as_posix()]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    raw=encoded(manifest)
    (kit/'bundle-manifest.json').write_bytes(raw)
    lock=json.loads((ROOT/'recipe-lock.json').read_bytes())
    lock['kit_manifest_sha256']=hashlib.sha256(raw).hexdigest()
    lock['format']='ds41_public_recipe_lock_v2'
    (ROOT/'recipe-lock.json').write_bytes(encoded(lock))
    print(json.dumps(dict(files=len(manifest['files']),kit_manifest_sha256=lock['kit_manifest_sha256'])))
if __name__=='__main__':freeze()
