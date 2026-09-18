# SPDX-License-Identifier: AGPL-3.0-only
"""Maintainer upload of reviewed runtime assets; never used by end users.

Uses HF_TOKEN_WRITE only for Hugging Face authentication, never logs it.
Bounded-memory file reads; no CUDA, image import, build or server lifecycle calls.
"""
import argparse
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import resource
import tarfile
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
REPO='coolbho3k/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-DGX-Spark-runtime'
IMAGE_SHA='dd6b4604be3985ad2413e40123d3a74415c4e63b38d0caeff2e54b47eb7fe5a3'
CACHE_SHA='9a5c9c563317cd9205b5e640a9f082e326a875497a2a742c2e9d34faa5cd1bad'


class BoundedFile(io.BufferedReader):
    def read(self,size=-1):
        # All Hub/LFS callers must stream, not materialize a 21-GiB image.
        if size<0:raise ValueError('Unbounded upload read refused')
        result=super().read(size)
        os.posix_fadvise(self.fileno(),max(0,self.tell()-len(result)),len(result),os.POSIX_FADV_DONTNEED)
        return result


def digest(path):
    value=hashlib.sha256()
    with BoundedFile(io.FileIO(path,'r')) as f:
        while block:=f.read(2**20):value.update(block)
    return value.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--publish',action='store_true',required=True)
    p.add_argument('--audit',type=Path,required=True)
    a=p.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(768*2**20,768*2**20))
    token=os.environ.pop('HF_TOKEN_WRITE','')
    if not token.startswith('hf_'):raise ValueError('HF_TOKEN_WRITE is required for publication only')
    audit=json.loads(a.audit.read_bytes())
    if audit.get('archive_sha256')!=IMAGE_SHA or audit.get('layers')!=90:
        raise ValueError('Exact image audit required')
    if audit.get('reviewed_for_publication') is not True:
        raise ValueError('Manual review of redacted scanner findings required')
    os.environ['HF_HUB_DISABLE_XET']='1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    logging.disable(logging.CRITICAL)
    def guard():
        while True:
            available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
            if available<1024*2**20:
                print('Upload stopped to preserve headroom; the serving process was not touched.',flush=True)
                os._exit(75)
            time.sleep(2)
    threading.Thread(target=guard,daemon=True).start()
    from huggingface_hub import HfApi, CommitOperationAdd
    api=HfApi(token=token)
    if api.whoami().get('name')!='coolbho3k':raise ValueError('Unexpected publisher identity')
    paths={'runtime-image.tar.gz':ROOT/'artifacts/ds41-runtime-image-v1.tar.gz',
           'kernel-cache.tar':ROOT/'artifacts/ds41-runtime-cache-v1.tar'}
    hashes={}
    for name,path in paths.items():
        expected=IMAGE_SHA if name.startswith('runtime-image') else CACHE_SHA
        if digest(path)!=expected:raise ValueError('Asset differs from reviewed bytes: '+name)
        hashes[name]=dict(bytes=path.stat().st_size,sha256=expected)
        print('Verified '+name,flush=True)
    source=ROOT/'artifacts/ds41-public-runtime-source-v1.tar.gz'
    if not source.exists():
        with tarfile.open(source,'w:gz',compresslevel=1) as archive:
            for path in sorted((ROOT/'release/runtime').rglob('*')):
                if path.is_file():archive.add(path,arcname='runtime/'+str(path.relative_to(ROOT/'release/runtime')),recursive=False)
            for name in ('LICENSE','CREDITS.md','THIRD_PARTY_NOTICES.md'):
                archive.add(ROOT/name,arcname=name,recursive=False)
    paths['runtime-source.tar.gz']=source
    hashes['runtime-source.tar.gz']=dict(bytes=source.stat().st_size,sha256=digest(source))
    card='''---
license: agpl-3.0
tags: [deepseek, exl3, dgx-spark, runtime]
---
# DeepSeek V4.1 Flash EXL3 3bpw — two-Spark runtime assets

Prebuilt ARM64/SM121 runtime and cache for the companion two-DGX-Spark recipe.
Not model weights and not a standalone generic vLLM Docker image. Use the
recipe's pinned image identity, runtime overlay, verified target/draft weights
and configuration together. The complete new fresh-clone launcher has not yet
received its two-node GPU boot test; the underlying serving payload completed
the six-session 3.15M-token test. No general portability guarantee is made.

Full credit to **MiaAI Lab / Wesley Young** and contributors for the Engram,
grouped-prefill, cooperative-MoE and EXL3 serving foundation:
https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks

MiaAI-derived code and our adaptations are **AGPL-3.0-only**. Corresponding
runtime source, native-kernel sources, build helpers, provenance and notices
are included in runtime-source.tar.gz; source and notices are also present
inside the image. Separately licensed upstream packages and NVIDIA CUDA
components retain their respective licenses. Model weights retain their own
licenses; this dataset does not relicense them or the proprietary components.

Target: https://huggingface.co/coolbho3k/DeepSeek-V4.1-Flash-EXL3-3bpw
Draft: https://huggingface.co/coolbho3k/DeepSeek-V4.1-Flash-DSpark-EXL3-3bpw

The recipe uses experimental display-reserve KV: both hosts should be headless
with nvidia_drm modeset=1 fbdev=0. Never change drivers while serving is active.
Archives were streamed and scanned layer-by-layer. Recorded matches were
reviewed as upstream program symbols/checksums/parser markers/test fixtures
and an empty .ssh directory, not campaign credentials. This is not a general
security certification. Preserve upstream licenses and source when sharing.
'''
    api.create_repo(REPO,repo_type='dataset',private=False,exist_ok=True)
    info=api.dataset_info(REPO)
    if info.private:raise ValueError('Expected public runtime dataset')
    # One file per commit makes partial uploads recoverable without another
    # source build. Existing matching LFS payloads are deduplicated by the Hub.
    for name,path in paths.items():
        print('Uploading '+name,flush=True)
        with BoundedFile(io.FileIO(path,'r')) as stream:
            op=CommitOperationAdd(path_in_repo=name,path_or_fileobj=stream)
            api.create_commit(REPO,repo_type='dataset',operations=[op],num_threads=1,
                commit_message='Publish pinned '+name+' for EXL3 3bpw Spark recipe')
        print('Uploaded '+name,flush=True)
    receipt=dict(format='ds41_public_runtime_assets_v1',files=hashes,
        image_identity_sha256='03c151b169249d413dc64a365d3fa8f5c104561d1eb3bf2bd30138b9010c3e3b',
        cache_manifest_sha256='939a5317880b3003e76657db9a349ab509225c463686604ca536c39e34b1e1ee',
        full_clean_clone_gpu_tested=False)
    result=api.create_commit(REPO,repo_type='dataset',num_threads=1,
        commit_message='Add runtime source attribution and immutable asset inventory',operations=[
            CommitOperationAdd(path_in_repo='README.md',path_or_fileobj=card.encode()),
            CommitOperationAdd(path_in_repo='assets.json',path_or_fileobj=(json.dumps(receipt,indent=2)+'\n').encode())])
    output=ROOT/'reports/public-runtime-upload-v1.json'
    output.write_text(json.dumps(dict(repo=REPO,revision=result.oid,**receipt),indent=2)+'\n')
    print(json.dumps(dict(status='runtime_assets_published',repo=REPO,revision=result.oid)),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        # Do not echo signed upload URLs, authorization headers or token values.
        print('Publication failed: '+type(error).__name__+'; no serving lifecycle action was performed.',flush=True)
        raise SystemExit(1)
