# SPDX-License-Identifier: AGPL-3.0-only
"""Download pinned public assets and import prebuilt image; never compile code.

Standard library only. Downloads are resumable, bounded-memory and SHA256
verified. Do this before loading the model, not beside a resident GPU job.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import urllib.error
import urllib.request

def sha(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(2**20):
            result.update(block)
            os.posix_fadvise(stream.fileno(),stream.tell()-len(block),len(block),os.POSIX_FADV_DONTNEED)
    return result.hexdigest()

def relative(name):
    p=PurePosixPath(name)
    if not name or name=='.' or any(c in name for c in ('\0','\n','\r')) or p.is_absolute() or '..' in p.parts or '\\' in name or str(p)!=name:
        raise ValueError('Unsafe asset path')
    return name

def download(url,path,digest,size=None):
    if not re.fullmatch('[0-9a-f]{64}',digest):raise ValueError('SHA256 pin required')
    if path.exists():
        if path.is_symlink() or (size is not None and path.stat().st_size!=size) or sha(path)!=digest:
            raise ValueError('Existing asset differs; preserve and inspect: '+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_name(path.name+'.download-part')
    if partial.is_symlink():raise ValueError('Redirected partial download')
    offset=partial.stat().st_size if partial.exists() else 0
    headers={'Range':f'bytes={offset}-'} if offset else {}
    request=urllib.request.Request(url,headers=headers)
    with urllib.request.urlopen(request,timeout=120) as response:
        append=offset>0 and response.status==206
        if append and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
            raise ValueError('Unexpected resume range')
        with partial.open('ab' if append else 'wb') as stream:
            while block:=response.read(2**20):
                stream.write(block)
                stream.flush()
                os.posix_fadvise(stream.fileno(),max(0,stream.tell()-len(block)),len(block),os.POSIX_FADV_DONTNEED)
    if (size is not None and partial.stat().st_size!=size) or sha(partial)!=digest:
        raise ValueError('Downloaded asset hash/size mismatch: '+path.name)
    if path.exists():raise ValueError('Asset appeared during download')
    partial.replace(path)
    print(json.dumps(dict(stage='download_verified',file=path.name,bytes=path.stat().st_size)),flush=True)

def snapshot(spec,destination):
    base=f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/"
    download(base+'release-manifest.json',destination/'release-manifest.json',spec['manifest_sha256'])
    manifest=json.loads((destination/'release-manifest.json').read_bytes())
    if manifest['repo_id']!=spec['repo']:raise ValueError('HF manifest repository mismatch')
    for name,row in sorted(manifest['files'].items()):
        relative(name);download(base+name,destination/name,row['sha256'],row['bytes'])

def extract(archive,destination):
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():raise ValueError('Unsafe existing extraction directory')
        return
    staging=destination.with_name(destination.name+'.extracting')
    if staging.exists():raise ValueError('Interrupted extraction exists; inspect it before continuing')
    staging.mkdir()
    with tarfile.open(archive,'r|*') as stream:
        for item in stream:
            relative(item.name)
            if not (item.isdir() or item.isfile()):raise ValueError('No links/devices in runtime archives')
            output=staging/item.name
            if item.isdir():output.mkdir(parents=True,exist_ok=True);continue
            output.parent.mkdir(parents=True,exist_ok=True)
            with output.open('xb') as target:shutil.copyfileobj(stream.extractfile(item),target,2**20)
            os.chmod(output,item.mode&0o777)
            os.utime(output,(item.mtime,item.mtime))
    staging.rename(destination)

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj);return obj

def fingerprint(path):
    info=path.stat()
    return [info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns]

def runtime_storage(root,runtime):
    """Keep runtime versions separate while reusing large model downloads."""
    digest=hashlib.sha256(json.dumps(runtime,sort_keys=True).encode()).hexdigest()
    return Path(root)/'runtime-assets'/digest

def bootstrap(lock,root,kit):
    runtime=lock['runtime']
    if runtime is None:raise ValueError('Prebuilt runtime publication is not complete; no source-build fallback')
    jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if jobs:raise ValueError('Prepare assets before starting GPU serving')
    root.mkdir(parents=True,exist_ok=True)
    root = root.resolve()
    assets_root=runtime_storage(root,runtime)
    assets_root.mkdir(parents=True,exist_ok=True)
    kit = kit.resolve()
    checker=module(kit/'verify.py','verify_source_runtime')
    checker.verify(kit,lock['kit_manifest_sha256'])
    frozen=root/'kits'/lock['kit_manifest_sha256']
    if not frozen.exists():
        frozen.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(kit,frozen)
    checker.verify(frozen,lock['kit_manifest_sha256'])
    kit=frozen
    lock_sha=hashlib.sha256(json.dumps(lock,sort_keys=True).encode()).hexdigest()
    prepared=root/'prepared.json'
    if prepared.exists():
        previous=json.loads(prepared.read_bytes())
        if previous.get('lock_sha256')==lock_sha:
            verifier=module(kit/'tools/verify_public_download.py','reuse_weights')
            manifest,summary=verifier.load_manifest(root/'model/release-manifest.json',lock['model']['manifest_sha256'])
            verifier.check_receipt(root/'model',manifest,summary,json.loads((root/'model-verified.json').read_bytes()))
            draft_files=json.loads((root/'draft-verified.json').read_bytes())
            for name,row in draft_files.items():
                p=root/'draft-exl3'/relative(name)
                if p.is_symlink() or fingerprint(p)!=row:raise ValueError('Draft changed after verification: '+name)
            print(json.dumps(dict(stage='reuse_verified_assets',cache=str(root))),flush=True)
            return previous
    image_check=module(kit/'tools/verify_runtime_image.py','verify_image')
    expected=json.loads((kit/'runtime-image-identity.json').read_bytes())
    if runtime.get('transport') == 'ghcr':
        registry=module(Path(__file__).with_name('registry.py'),'public_registry')
        registry.require_public(runtime)
        image=registry.prepare(runtime,assets_root,expected,image_check,sha)
    else:
        image=None
    base=None if runtime.get('transport') == 'ghcr' else f"https://huggingface.co/datasets/{runtime['repo']}/resolve/{runtime['revision']}/"
    if base is not None:
        for name,row in runtime['files'].items():
            download(base+relative(name),assets_root/'downloads'/name,row['sha256'],row['bytes'])
    cache=assets_root/'kernel-cache'
    extract(assets_root/'downloads/kernel-cache.tar',cache)
    checker=module(kit/'verify.py','verify_runtime');checker.verify(kit,lock['kit_manifest_sha256'])
    image_check=module(kit/'tools/verify_runtime_image.py','verify_image')
    expected=json.loads((kit/'runtime-image-identity.json').read_bytes())
    if image is None:
        for candidate in subprocess.check_output(['docker','image','ls','-q','--no-trunc'],text=True).split():
            data=image_check.inspect(candidate)
            if image_check.identity(data)==expected:image=data['Id'];break
    if image is None:
        subprocess.run(['docker','load','--input',str(assets_root/'downloads/runtime-image.tar.gz')],check=True)
        for candidate in subprocess.check_output(['docker','image','ls','-q','--no-trunc'],text=True).split():
            data=image_check.inspect(candidate)
            if image_check.identity(data)==expected:image=data['Id'];break
    if image is None:raise ValueError('Imported image differs from the pinned runtime identity')
    snapshot(lock['model'],root/'model');snapshot(lock['draft'],root/'draft-exl3')
    weights=module(kit/'tools/verify_public_download.py','verify_public_weights')
    manifest,summary=weights.load_manifest(root/'model/release-manifest.json',lock['model']['manifest_sha256'])
    receipt_path=root/'model-verified.json'
    if receipt_path.exists():weights.check_receipt(root/'model',manifest,summary,json.loads(receipt_path.read_bytes()))
    else:receipt_path.write_text(json.dumps(weights.verify_full(root/'model',manifest,summary),indent=2)+'\n')
    draft_manifest=json.loads((root/'draft-exl3/release-manifest.json').read_bytes())
    draft_names=list(draft_manifest['files'])+['release-manifest.json']
    (root/'draft-verified.json').write_text(json.dumps({n:fingerprint(root/'draft-exl3'/n) for n in draft_names})+'\n')
    # Restore exact cache mtimes, including nanoseconds, from the pinned manifest.
    cache_manifest=json.loads((cache/'cache-manifest.json').read_bytes())
    for name,row in cache_manifest['files'].items():
        if 'mtime_ns' in row:os.utime(cache/name,ns=(row['mtime_ns'],row['mtime_ns']))
    result=dict(lock_sha256=hashlib.sha256(json.dumps(lock,sort_keys=True).encode()).hexdigest(),kit=str(kit),model=str(root/'model'),draft=str(root/'draft-exl3'),
        model_receipt=str(receipt_path),cache=str(cache),runs=str(root/'runs'),image=image,
        uid=os.getuid(),gid=os.getgid())
    (root/'runs').mkdir(exist_ok=True)
    (root/'prepared.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--kit',type=Path,required=True);a=p.parse_args()
    result=bootstrap(json.loads(a.lock.read_bytes()),a.cache_dir.resolve(),a.kit.resolve())
    print(json.dumps(dict(status='public_assets_prepared',**result)),flush=True)
