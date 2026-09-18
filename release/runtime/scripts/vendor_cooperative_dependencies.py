# SPDX-License-Identifier: AGPL-3.0-only
"""Import the exact MIT-licensed ExLlama header closure into our AGPL port.

Does not touch the ExLlama checkout used by the running quantization campaign.
Original license notices are retained; this integration is AGPLv3-only.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess
from vendor_miaai_cooperative import fetch, ROOT

REPO = 'turboderp-org/exllamav3'
COMMIT = '02aef45cd681b960a00afcd0749a4ab99e6c1bfe'
DEST = ROOT/'vendor/miaai-cooperative-dependencies-agpl'
EXT = 'exllamav3/exllamav3_ext'


def main():
    if DEST.exists():
        raise ValueError('Preserve the existing dependency snapshot')
    tree = json.loads(fetch(f'https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1'))
    if tree['sha'] != COMMIT or tree.get('truncated'):
        raise ValueError('Incomplete pinned dependency tree')
    entries = {e['path']: e for e in tree['tree'] if e['type'] == 'blob'}
    pending = [f'{EXT}/{n}' for n in ('util.h', 'util.cuh', 'compat.cuh',
        'quant/exl3_gemv_kernel.cuh', 'quant/hadamard_inner.cuh')]
    pending.append('LICENSE')
    sources = {}; proof = {}
    while pending:
        name = pending.pop()
        if name in sources:
            continue
        entry = entries[name]
        url = f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/{name}'
        raw = fetch(url)
        blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if blob != entry['sha'] or len(raw) != entry['size']:
            raise ValueError('Git blob mismatch: '+name)
        local = raw if raw.endswith(b'\n') else raw+b'\n'
        sources[name] = local
        proof[name] = dict(url=url, git_blob_sha1=blob, upstream_sha256=hashlib.sha256(raw).hexdigest(),
            sha256=hashlib.sha256(local).hexdigest(), bytes=len(local), final_newline_added=local!=raw)
        for include in re.findall(r'^\s*#\s*include\s*"([^"]+)"', raw.decode(), re.M):
            path = posixpath.normpath(str(PurePosixPath(name).parent/include))
            if path not in entries:
                path = f'{EXT}/{include}'
            if path not in entries:
                raise ValueError('Unresolved quoted dependency: '+include)
            pending.append(path)
    sources['UPSTREAM.json'] = (json.dumps(dict(repo=REPO, commit=COMMIT, files=proof,
        upstream_license='MIT', integration_license='AGPL-3.0-only',
        local_changes='Only missing final newlines normalized; native overrides live in the separately attributed MiaAI snapshot.'),
        indent=2, sort_keys=True)+'\n').encode()
    sources['README.md'] = b'# AGPLv3 cooperative MoE integration dependencies\n\nThis snapshot is part of the AGPL-3.0-only MiaAI cooperative MoE port.\nThe unmodified ExLlamaV3 header subset is by Turboderp and retains its MIT license (LICENSE).\nSee UPSTREAM.json for exact commit and per-file provenance.\n'
    patch = ['*** Begin Patch']
    for name, raw in sorted(sources.items()):
        # Headers without final newlines require apply_patch's exact sentinel.
        lines = raw.decode().splitlines()
        patch += [f'*** Add File: {DEST/name}', *('+'+line for line in lines)]
        if not raw.endswith(b'\n'):
            raise ValueError('Unterminated pinned dependency: '+name)
    patch.append('*** End Patch')
    subprocess.run(['apply_patch'], input='\n'.join(patch)+'\n', text=True, check=True, cwd=ROOT)
    print(json.dumps(dict(status='pinned_dependencies_imported', files=len(proof), commit=COMMIT)))


if __name__ == '__main__':
    main()
