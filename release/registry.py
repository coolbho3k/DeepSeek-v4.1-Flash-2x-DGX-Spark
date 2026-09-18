# SPDX-License-Identifier: AGPL-3.0-only
"""Pinned GHCR transport. No image build, container start, or GPU operations."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import uuid

IMAGE = re.compile(r'ghcr\.io/([a-z0-9][a-z0-9-]*)/([a-z0-9][a-z0-9._-]*)@sha256:([0-9a-f]{64})')
ASSETS = ('kernel-cache.tar', 'runtime-source.tar.gz')


def validate(runtime):
    if runtime.get('transport') != 'ghcr' or not IMAGE.fullmatch(runtime.get('image', '')):
        raise ValueError('GHCR requires an immutable owner/image@sha256 reference')
    if not re.fullmatch('[0-9a-f]{64}', runtime.get('cache_manifest_sha256', '')):
        raise ValueError('Missing kernel-cache manifest pin')
    if set(runtime.get('files', {})) != set(ASSETS):
        raise ValueError('Expected exactly the cache and corresponding-source assets')
    for name in ASSETS:
        row = runtime['files'][name]
        if type(row.get('bytes')) is not int or not 0 < row['bytes'] <= 2**31:
            raise ValueError('Invalid bounded runtime asset size')
        if not re.fullmatch('[0-9a-f]{64}', row.get('sha256', '')):
            raise ValueError('Missing runtime asset SHA256')


def public_manifest(runtime):
    """Verify anonymous GHCR access and the exact manifest digest; no layers."""
    validate(runtime)
    owner, name, digest = IMAGE.fullmatch(runtime['image']).groups()
    query = urllib.parse.urlencode({'service': 'ghcr.io', 'scope': f'repository:{owner}/{name}:pull'})
    with urllib.request.urlopen('https://ghcr.io/token?' + query, timeout=30) as response:
        token = json.loads(response.read(2**20)).get('token')
    if not token:
        raise ValueError('GHCR did not grant anonymous pull access')
    request = urllib.request.Request(f'https://ghcr.io/v2/{owner}/{name}/manifests/sha256:{digest}', headers={
        'Authorization': 'Bearer ' + token,
        'Accept': 'application/vnd.oci.image.manifest.v1+json, application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.v2+json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(4*2**20+1)
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('Public registry manifest digest mismatch')
    return json.loads(raw)


def require_public(runtime):
    try:
        return public_manifest(runtime)
    except urllib.error.HTTPError as error:
        if error.code in (401, 403, 404):
            raise ValueError('The pinned GHCR runtime is not anonymously accessible. The publisher must make the package Public; no local build or credential fallback was attempted.') from None
        raise


def prepare(runtime, root, expected, image_check, digest_file):
    validate(runtime)
    image = runtime['image']
    found = subprocess.run(['docker', 'image', 'inspect', image], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False)
    if found.returncode:
        subprocess.run(['docker', 'pull', '--platform=linux/arm64', image], check=True)
    node = image_check.inspect(image)
    image_check.verify(node, expected)
    downloads = Path(root)/'downloads'
    downloads.mkdir(parents=True, exist_ok=True)
    missing = []
    for name in ASSETS:
        path, row = downloads/name, runtime['files'][name]
        if path.is_symlink():
            raise ValueError('Redirected runtime asset')
        if path.exists():
            if not path.is_file() or path.stat().st_size != row['bytes'] or digest_file(path) != row['sha256']:
                raise ValueError('Preserve and inspect mismatched runtime asset: ' + name)
        else:
            missing.append(name)
    if not missing:
        return node['Id']
    # A stopped, pristine container exposes packaged files. It is NEVER started,
    # has no host mounts or GPU access, and is removed only by its returned ID.
    owner = uuid.uuid4().hex
    container = subprocess.check_output(['docker', 'create', '--network=none', '--runtime=runc',
        '--label=ds41.asset-extraction=' + owner, '--entrypoint=/bin/true', node['Id']], text=True).strip()
    if not re.fullmatch('[0-9a-f]{64}', container):
        raise ValueError('Unexpected asset-container ID; inspect Docker state')
    try:
        for name in missing:
            path, row = downloads/name, runtime['files'][name]
            partial = path.with_name(name + '.copying')
            if partial.exists() or partial.is_symlink():
                raise ValueError('Preserve interrupted asset copy before retrying: ' + str(partial))
            subprocess.run(['docker', 'cp', container + ':/opt/ds41-release/' + name, str(partial)], check=True)
            if partial.is_symlink() or partial.stat().st_size != row['bytes'] or digest_file(partial) != row['sha256']:
                raise ValueError('Packaged runtime asset hash/size mismatch')
            if path.exists() or path.is_symlink():
                raise ValueError('Runtime asset appeared during extraction')
            partial.rename(path)
    finally:
        # No --force: an unexpectedly running container must never be killed.
        subprocess.run(['docker', 'rm', container], check=True, stdout=subprocess.DEVNULL)
    return node['Id']
