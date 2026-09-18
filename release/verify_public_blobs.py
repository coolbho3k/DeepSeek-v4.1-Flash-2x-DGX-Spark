# SPDX-License-Identifier: AGPL-3.0-only
"""Verify anonymous access to the pinned manifest and every image blob.

HEAD requests only: no layer downloads, Docker login, GPU work, or serving
changes. Anonymous pull tokens are held only in memory and never logged.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

import registry


def verify(runtime):
    manifest = registry.require_public(runtime)
    owner, name, digest = registry.IMAGE.fullmatch(runtime['image']).groups()
    query = urllib.parse.urlencode(dict(service='ghcr.io', scope=f'repository:{owner}/{name}:pull'))
    with urllib.request.urlopen('https://ghcr.io/token?' + query, timeout=30) as response:
        token = json.loads(response.read(2**20))['token']
    blobs = [manifest['config']] + manifest['layers']

    def check(row):
        url = f"https://ghcr.io/v2/{owner}/{name}/blobs/{row['digest']}"
        request = urllib.request.Request(url, method='HEAD', headers={'Authorization': 'Bearer ' + token})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 200 or int(response.headers['Content-Length']) != row['size']:
                raise ValueError('Anonymous blob metadata mismatch')
        return dict(digest=row['digest'], bytes=row['size'], anonymous_head_verified=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        observations = list(pool.map(check, blobs))
    return dict(status='anonymous_manifest_and_all_blob_headers_verified', image=runtime['image'],
        manifest_digest='sha256:' + digest, layer_count=len(manifest['layers']),
        blobs=observations, payload_bytes_downloaded=0, docker_credentials_used=False,
        server_touched=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, default=Path(__file__).resolve().parents[1] / 'recipe-lock.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous verification')
    report = verify(json.loads(args.lock.read_bytes())['runtime'])
    report['verifier_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'blobs'}), flush=True)
