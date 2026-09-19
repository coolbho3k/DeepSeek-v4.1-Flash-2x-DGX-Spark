# SPDX-License-Identifier: AGPL-3.0-only
"""Prepare a bounded plain-CUDA top3 drafter specialization; no GPU work."""
import argparse
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'dcp_overlap'))
from prepare import bounded_read,encoded,load_parent,sha,safe_name

LAUNCH_SHA='bb0445284e2aba9a553c055fa290a9d7f9d860b5d8162c155edca367e4470865'


def kernels(raw):
    if sha(raw)!=LAUNCH_SHA:raise ValueError('Changed FP32 grouped parent')
    text=raw.decode()
    start='namespace ds41_grouped_staged {\n'
    stop='template<int WK,int WNT,int PF,bool UP>\nvoid resource('
    if text.count(start)!=1 or text.count(stop)!=1:raise ValueError('Changed extraction boundaries')
    body=start+text.split(start)[1].split(stop)[0]+'}\n'
    if body.count('TOP=6;')!=1:raise ValueError('Changed draft routing shape')
    return ('// SPDX-License-Identifier: AGPL-3.0-only\n'
            '// Exact pinned staged CUDA bodies, with top3 slot capacity.\n'+body.replace('TOP=6;','TOP=3;')).encode()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--parent-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.absolute()
    if out.exists() or out.resolve()!=out:raise ValueError('Fresh unredirected output required')
    _,parent=load_parent(a.parent.absolute(),a.parent_sha256)
    # Runtime is at release/runtime, a sibling of release/experimental.
    root=HERE.parent.parent/'runtime/vendor/miaai-cooperative-dependencies-agpl'
    manifest=json.loads(bounded_read(root/'UPSTREAM.json'))
    files={}
    for name,row in manifest['files'].items():
        raw=bounded_read(root/safe_name(name))
        if sha(raw)!=row['sha256']:raise ValueError('Changed native dependency')
        prefix='exllamav3/exllamav3_ext/'
        if name.startswith(prefix):files['source/include/'+name.removeprefix(prefix)]=raw
    for name in ('staged_register_gemv.cuh','staged_grouped_gemv.cuh','semantics.cuh'):
        files['source/include/'+name]=parent['native-source/'+name]
    files['source/include/draft_grouped_kernels.cuh']=kernels(parent['native-source/staged-grouped-launch.cu'])
    files['source/draft_top3.cu']=bounded_read(HERE/'draft_top3.cu')
    files['compile_native.py']=bounded_read(HERE/'compile_top3.py')
    files['prepare_top3.py']=bounded_read(Path(__file__).resolve())
    # Corresponding original host source and notices retained for attribution.
    files['original-staged-grouped-launch.cu']=parent['native-source/staged-grouped-launch.cu']
    mia=HERE.parent.parent/'runtime/vendor/miaai-cooperative-moe-agpl'
    for name in ('LICENSE','LICENSE.MIT','extensions/cooperative_moe/native/LICENSE.exllamav3'):
        files[Path(name).name]=bounded_read(mia/name)
    if sum(map(len,files.values()))>4*2**20:raise ValueError('Unexpected source size')
    out.mkdir(parents=True,mode=0o700)
    for name,raw in files.items():
        path=out/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(raw)
    record=dict(status='draft_top3_prepared',parent_manifest_sha256=a.parent_sha256,
        draft_rows_max=30,top_k=3,fp32_mma=True,fp32_route_weights=True,abi=1,
        license='AGPL-3.0-only',files={name:sha(raw) for name,raw in sorted(files.items())})
    with (out/'prepared.json').open('xb') as f:f.write(encoded(record))
    print(json.dumps(dict(output=str(out),prepared_sha256=sha(encoded(record)))))


if __name__=='__main__':main()
