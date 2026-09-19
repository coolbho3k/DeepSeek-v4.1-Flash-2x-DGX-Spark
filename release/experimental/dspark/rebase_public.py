# SPDX-License-Identifier: AGPL-3.0-only
"""Stage a candidate on public portability code; never publish or deploy it.

Reapply the exact source-pinned transform to the verified PUBLIC parent, not
the campaign's private launcher. Require every serving byte to equal the
explicit candidate. Published image/cache/HF download integration is retained.
All qualification remains separate: this is only a reproducibility check.
"""
import argparse
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from prepare_candidate import transform
from contracts import Policy
from prepare import bounded_read,encoded,load_parent,safe_name,sha,MAX_FILE,MAX_TOTAL


def require_serving_parity(staged,candidate):
    expected={name:raw for name,raw in candidate.items() if name.startswith('serving/')}
    actual={name:raw for name,raw in staged.items() if name.startswith('serving/')}
    if not expected or expected.keys()!=actual.keys():
        raise ValueError('Changed serving inventory while rebasing')
    changed=[name for name in expected if expected[name]!=actual[name]]
    if changed:raise ValueError('Staged serving differs from explicit candidate: '+', '.join(changed))
    return len(expected)


def stage(args):
    parent=args.public_parent.absolute();candidate=args.candidate.absolute();out=args.output.absolute()
    if (out.exists() or out.resolve()!=out or not out.parent.is_dir()
            or out.is_relative_to(parent) or out.is_relative_to(candidate)):
        raise ValueError('Choose a fresh canonical sibling output')
    public_manifest,public=load_parent(parent,args.public_sha256)
    _,chosen=load_parent(candidate,args.candidate_sha256)
    selection=json.loads(chosen['runtime-requirements.json'])['dspark_candidate']
    policy=Policy(selection['draft_tokens'],selection['verification'],tuple(selection['prefix_lengths']))
    kernels=selection['draft_kernels']
    if kernels['top3']!=kernels['kv_only'] or kernels['full_markov_head']:
        raise ValueError('Unsupported selected kernel combination')
    receipt=json.loads(chosen['experiments/dspark/native/complete.json'])
    draft_binary=chosen['serving/dspark_draft_top3.so'] if kernels['top3'] else None
    payload=transform(public,policy,chosen['serving/cooperative_moe.so'],receipt,
        draft_binary=draft_binary,markov_add=kernels['markov_add'])
    count=require_serving_parity(payload,chosen)
    for name,raw in chosen.items():
        if name.startswith('experiments/dspark/'):
            if name in public and public[name]!=raw:raise ValueError('Preserve existing corresponding source: '+name)
            payload[name]=raw
    # Requirements describe this staging result honestly, not approval to ship.
    requirements=json.loads(payload['runtime-requirements.json'])
    requirements['dspark_candidate'].update(staged_serving_byte_parity=True,
        compared_candidate_manifest_sha256=args.candidate_sha256,
        public_fresh_clone_gpu_qualified=False)
    payload['runtime-requirements.json']=encoded(requirements)
    if len(payload)>1000 or sum(map(len,payload.values()))>MAX_TOTAL or any(len(v)>MAX_FILE for v in payload.values()):
        raise ValueError('Staged bundle exceeds bounded input scope')
    manifest=dict(format=public_manifest['format'],standalone_runtime=False,
        clean_rebuild_qualified=False,publication_approved=False,serving_qualified=False,
        variant='staged_public_dspark',parent_manifest_sha256=args.public_sha256,
        compared_candidate_manifest_sha256=args.candidate_sha256,
        files={name:dict(bytes=len(raw),sha256=sha(raw)) for name,raw in sorted(payload.items())})
    out.mkdir(mode=0o700)
    for name,raw in payload.items():
        path=out/safe_name(name);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(raw)
    raw=encoded(manifest)
    with (out/'bundle-manifest.json').open('xb') as stream:stream.write(raw)
    load_parent(out,sha(raw))
    return dict(status='public_candidate_staged_not_promoted',output=str(out),
        manifest_sha256=sha(raw),serving_files_byte_identical=count,
        public_inputs_unchanged=True,live_server_touched=False,published=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('public-parent','candidate','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--public-sha256',required=True)
    parser.add_argument('--candidate-sha256',required=True)
    print(json.dumps(stage(parser.parse_args()),indent=2))
