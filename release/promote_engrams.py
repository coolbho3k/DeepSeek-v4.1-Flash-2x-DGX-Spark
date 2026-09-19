# SPDX-License-Identifier: AGPL-3.0-only
"""Promote fully staged Engrams in one additive Hub main-branch commit.

Requires both completed rank receipts. Does not update Git, launch serving,
change original weight bytes, or print credentials/signed storage URLs.
"""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import urllib.request

from publish_engrams import PREFIX, REPO


def encoded(value): return (json.dumps(value, indent=2, sort_keys=True)+'\n').encode()
def digest(raw): return hashlib.sha256(raw).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--rank0',type=Path,required=True)
    p.add_argument('--rank1',type=Path,required=True)
    p.add_argument('--readme',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--publish',action='store_true',required=True)
    a=p.parse_args()
    if a.output.exists(): raise ValueError('Publication output must be fresh')
    raw=a.manifest.read_bytes();m=json.loads(raw)
    token=os.environ.pop('HF_TOKEN_WRITE','')
    if not token.startswith('hf_'): raise ValueError('HF_TOKEN_WRITE required')
    logging.disable(logging.CRITICAL)
    from huggingface_hub import HfApi,CommitOperationCopy,CommitOperationAdd
    api=HfApi(token=token)
    if api.whoami().get('name')!='coolbho3k':raise ValueError('Unexpected publisher')
    head=api.model_info(REPO,revision='main',files_metadata=True)
    base=m['source_model']['revision']
    if head.sha!=base or head.private:
        raise ValueError('Public main advanced: review it before attempting promotion')
    if any(f.rfilename.startswith(PREFIX+'/') for f in head.siblings):
        raise ValueError('Never overwrite a published packed layout version')
    copies=[]
    for rank,receipt in enumerate((a.rank0,a.rank1)):
        r=json.loads(receipt.read_bytes())
        if (r['status']!='rank_staged_not_promoted' or r['rank']!=rank or r['repo']!=REPO
                or r['manifest_sha256']!=digest(raw)):
            raise ValueError('Mismatched staging receipt')
        info=api.model_info(REPO,revision=r['revision'],files_metadata=True)
        files={f.rfilename:f for f in info.siblings}
        expected={name for name,row in m['files'].items() if row['rank']==rank}
        if set(r['parts'])!=expected: raise ValueError('Missing staged tables')
        for name,parts in r['parts'].items():
            offset=0
            for i,part in enumerate(parts):
                if part['path']!=name+f'.part-{i:05d}' or part['offset']!=offset:
                    raise ValueError('Invalid ordered transport parts')
                offset+=part['bytes'];f=files[part['path']]
                if f.size!=part['bytes'] or not f.lfs or f.lfs.sha256!=part['sha256']:
                    raise ValueError('Staged Hub part differs from local bytes')
                copies.append(CommitOperationCopy(src_path_in_repo=part['path'],
                    path_in_repo=part['path'],src_revision=r['revision']))
            if offset!=m['files'][name]['bytes']:raise ValueError('Incomplete table parts')
            m['files'][name]['parts']=parts
    # Keep main's canonical manifest consistent with the additive model-card
    # notice. Old runners still fetch their old immutable snapshot and card.
    def public(name):
        with urllib.request.urlopen(f'https://huggingface.co/{REPO}/resolve/{base}/{name}',timeout=30) as response:
            data=response.read(2**20+1)
        if len(data)>2**20:raise ValueError('Oversized public metadata')
        return data
    canonical_raw=public('release-manifest.json')
    if digest(canonical_raw)!=m['source_model']['manifest_sha256']:
        raise ValueError('Canonical source inventory differs')
    canonical=json.loads(canonical_raw)
    card=public('README.md')
    notice=('\n\n## Optional lossless page15 Engram assets\n\n'
        'The [two-Spark recipe](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark) '
        'also provides a new packed SSD layout in [engram-page15-v1](engram-page15-v1/README.md). '
        'Original `engrams/*.safetensors`, main weights and draft weights are unchanged. '
        'Old recipe revisions keep using their pinned original files. New recipes pin '
        'the matching reader and packed manifest together; do not replace the original '
        'files manually. The layout is lossless, not another quantization.\n\n'
        'Cold bulk row retrieval was about 2x faster in component tests; measured whole-model '
        'decode changed from 30.35 to 30.78 tok/s and warmed 32K prefill was essentially flat. '
        'This is not a claim of a 2x serving speedup or broad quality qualification. '
        'Full credit to MiaAI Lab / Wesley Young for the native Engram/cache foundation; '
        'the derived reader and integration are AGPL-3.0-only in the recipe.\n').encode()
    card+=notice
    canonical['files']['README.md']=dict(bytes=len(card),sha256=digest(card))
    attributes=public('.gitattributes')
    attributes+=b'\nengram-page15-v1/*.bin.part-* filter=lfs diff=lfs merge=lfs -text\n'
    canonical['files']['.gitattributes']=dict(bytes=len(attributes),sha256=digest(attributes))
    manifest_raw=encoded(m)
    operations=copies+[
        CommitOperationAdd(path_in_repo=PREFIX+'/manifest.json',path_or_fileobj=manifest_raw),
        CommitOperationAdd(path_in_repo=PREFIX+'/README.md',path_or_fileobj=a.readme.read_bytes()),
        CommitOperationAdd(path_in_repo='README.md',path_or_fileobj=card),
        CommitOperationAdd(path_in_repo='.gitattributes',path_or_fileobj=attributes),
        CommitOperationAdd(path_in_repo='release-manifest.json',path_or_fileobj=encoded(canonical))]
    commit=api.create_commit(REPO,revision='main',parent_commit=base,operations=operations,
        num_threads=1,commit_message='Add lossless page15 Engrams; preserve canonical weights and legacy layout')
    # Journal the acknowledged commit before any later network check. A
    # verification timeout must never trigger a blind second publication.
    with a.output.open('x') as out:
        out.write(json.dumps(dict(status='published_verification_pending',repo=REPO,
            revision=commit.oid,manifest=m,manifest_sha256=digest(manifest_raw)),indent=2)+'\n')
    info=HfApi(token=False).model_info(REPO,revision=commit.oid,files_metadata=True)
    remote={f.rfilename:f for f in info.siblings}
    for name,row in canonical['files'].items():
        if name in ('README.md','.gitattributes'):continue
        previous=next(f for f in head.siblings if f.rfilename==name)
        current=remote[name]
        if (previous.blob_id,previous.lfs)!=(current.blob_id,current.lfs):
            raise ValueError('Canonical file identity changed unexpectedly')
    url=f'https://huggingface.co/{REPO}/resolve/{commit.oid}/{PREFIX}/manifest.json'
    with urllib.request.urlopen(url,timeout=30) as response:observed=response.read(65537)
    if observed!=manifest_raw:raise ValueError('Anonymous immutable manifest differs')
    for name,expected in (('README.md',card),('.gitattributes',attributes),
                          ('release-manifest.json',encoded(canonical))):
        url=f'https://huggingface.co/{REPO}/resolve/{commit.oid}/{name}'
        with urllib.request.urlopen(url,timeout=30) as response:observed=response.read(2**20+1)
        if observed!=expected:raise ValueError('Published metadata changed unexpectedly; inspect journal')
    result=dict(status='packed_engram_release_published',repo=REPO,revision=commit.oid,
        manifest_path=PREFIX+'/manifest.json',manifest_sha256=digest(manifest_raw),
        old_revision=base,canonical_weights_unchanged=True,manifest=m)
    with a.output.open('w') as out:out.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='manifest'}),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        print('Promotion failed: '+type(error).__name__+'; inspect public state before retrying.',flush=True)
        raise SystemExit(1)
