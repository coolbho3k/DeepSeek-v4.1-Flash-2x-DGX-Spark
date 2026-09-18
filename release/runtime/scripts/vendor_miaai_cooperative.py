# SPDX-License-Identifier: AGPL-3.0-only
"""Pin and attribute the new cooperative MoE sources without touching old kits."""
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
REPO='MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks'
COMMIT='b9c49e90bdcc6f1e0192feb57214df11b67d36aa'
DEST=ROOT/'vendor/miaai-cooperative-moe-agpl'


def fetch(url):
    request=urllib.request.Request(url,headers={'User-Agent':'ds41-attributed-cooperative-port'})
    with urllib.request.urlopen(request,timeout=30) as response:raw=response.read(2*2**20+1)
    if len(raw)>2*2**20:raise ValueError('Unexpected source size')
    return raw


def main():
    if DEST.exists():raise ValueError('Preserve any existing import')
    tree=json.loads(fetch(f'https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1'))
    if tree['sha']!=COMMIT or tree.get('truncated'):raise ValueError('Incomplete source tree')
    records={e['path']:e for e in tree['tree'] if e['type']=='blob'}
    names=sorted(n for n in records if n.startswith('extensions/cooperative_moe/') or n in
        ('LICENSE','LICENSE.MIT','docs/cooperative-moe.md','docs/cooperative-moe-quickstart.md'))
    patch=['*** Begin Patch'];proof={}
    for name in names:
        entry=records[name];url=f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/{name}'
        raw=fetch(url);blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if entry['mode']!='100644' and entry['mode']!='100755':raise ValueError('Regular source files only')
        if blob!=entry['sha'] or len(raw)!=entry['size']:raise ValueError('Git source identity mismatch')
        prefix=b''
        if Path(name).suffix in ('.py','.cu','.cuh','.sh'):
            leader='//' if name.endswith(('.cu','.cuh')) else '#'
            prefix=('\n'.join(leader+' '+line for line in (
                'SPDX-License-Identifier: AGPL-3.0-only',
                "Attribution: MiaAI Lab, Wesley Young and upstream contributors; derived ExLlamaV3 code by Turboderp.",
                f'Upstream: {REPO} @ {COMMIT}',
                'Local change: provenance/license prefix only; original body follows unchanged.',
                'See repository LICENSE, LICENSE.MIT and native/LICENSE.exllamav3.'))+'\n\n').encode()
        # Preserve a shell shebang as the first line.
        local=prefix+raw
        if raw.startswith(b'#!'):
            first,rest=raw.split(b'\n',1);local=first+b'\n'+prefix+rest
        if not local.endswith(b'\n'):raise ValueError('Unexpected unterminated source')
        patch += [f'*** Add File: {DEST/name}',*('+'+line for line in local.decode().splitlines())]
        proof[name]=dict(url=url,git_blob_sha1=blob,upstream_sha256=hashlib.sha256(raw).hexdigest(),
            local_sha256=hashlib.sha256(local).hexdigest(),upstream_bytes=len(raw),license='AGPL-3.0-only with retained MIT notices')
    receipt=dict(repo=REPO,commit=COMMIT,license='AGPL-3.0-only',files=proof,
        native_dependency_commit='02aef45cd681b960a00afcd0749a4ab99e6c1bfe',activated=False,
        local_changes='Attribution/SPDX prefixes only; no upstream code executed by import.')
    patch += [f'*** Add File: {DEST/"UPSTREAM.json"}',*('+'+line for line in json.dumps(receipt,indent=2,sort_keys=True).splitlines()),'*** End Patch']
    subprocess.run(['apply_patch'],input='\n'.join(patch)+'\n',text=True,check=True,cwd=ROOT)
    print(json.dumps(dict(status='cooperative_sources_vendored_not_enabled',commit=COMMIT,files=len(names))))


if __name__=='__main__':main()
