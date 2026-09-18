# SPDX-License-Identifier: AGPL-3.0-only
"""Prepare an isolated GHCR build context from reviewed compiled binaries.

No Docker build/push/start occurs here. The emitted Dockerfile flattens the
donor's merged filesystem and adds cache/source assets; it compiles nothing.
It removes obsolete lower-layer content, not installed framework packages.
End users pull the resulting image by digest, never run this tool.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
CACHE_SHA = '9a5c9c563317cd9205b5e640a9f082e326a875497a2a742c2e9d34faa5cd1bad'
CACHE_MANIFEST_SHA = '939a5317880b3003e76657db9a349ab509225c463686604ca536c39e34b1e1ee'


def image_name(owner):
    if not re.fullmatch('[a-z0-9][a-z0-9-]{0,38}', owner):
        raise ValueError('Use your lowercase GitHub account name')
    return f'ghcr.io/{owner}/deepseek-v4.1-flash-exl3-3bpw-2x-dgx-spark'


def quoted(value):
    if any(c in value for c in ('\n', '\r', '\0', '$')):
        raise ValueError('Unsupported Dockerfile metadata; inspect before packaging')
    return json.dumps(value)


def dockerfile(node):
    if node.get('Architecture') != 'arm64' or node.get('Os') != 'linux':
        raise ValueError('Expected the compiled ARM64 Linux runtime')
    config = node['Config']
    allowed = {'Env', 'Entrypoint', 'Cmd', 'WorkingDir', 'User', 'Labels', 'ExposedPorts', 'StopSignal'}
    if any(value for key, value in config.items() if key not in allowed):
        raise ValueError('Unhandled image configuration; do not silently discard it')
    lines = ['# SPDX-License-Identifier: AGPL-3.0-only',
        '# Packaging only: preserve compiled binaries, paths and native sources.',
        'ARG RUNTIME_SOURCE', 'FROM ${RUNTIME_SOURCE} AS compiled', 'FROM scratch',
        'COPY --from=compiled / /',
        'COPY kernel-cache.tar runtime-source.tar.gz runtime-assets.json /opt/ds41-release/']
    for entry in config.get('Env', []):
        key, value = entry.split('=', 1)
        if not re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', key):
            raise ValueError('Invalid environment key')
        if any(word in key.upper() for word in ('TOKEN', 'SECRET', 'PASSWORD', 'CREDENTIAL')):
            raise ValueError('Credential-like image environment must be reviewed and removed')
        lines.append('ENV ' + key + '=' + quoted(value))
    for field, instruction in (('WorkingDir', 'WORKDIR'), ('User', 'USER'), ('StopSignal', 'STOPSIGNAL')):
        if config.get(field):
            lines.append(instruction + ' ' + quoted(config[field]))
    for port in sorted(config.get('ExposedPorts', {})):
        if not re.fullmatch('[0-9]+/(tcp|udp)', port):
            raise ValueError('Unexpected exposed port')
        lines.append('EXPOSE ' + port)
    for field in ('Entrypoint', 'Cmd'):
        if config.get(field) is not None:
            lines.append(field.upper() + ' ' + json.dumps(config[field]))
    # Do not carry campaign-specific build labels into the public package.
    lines += ['LABEL org.opencontainers.image.title="DeepSeek V4.1 Flash EXL3 3bpw — two DGX Sparks"',
        'LABEL org.opencontainers.image.licenses="AGPL-3.0-only"',
        'LABEL org.opencontainers.image.description="MiaAI-based EXL3 serving adaptations, native vision, DSpark and FP4 KV. Third-party components retain their own licenses. Fresh-clone GPU qualification pending."',
        'LABEL io.ds41.upstream="https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks"',
        'LABEL io.ds41.corresponding-source="/opt/ds41-release/runtime-source.tar.gz"']
    return '\n'.join(lines) + '\n'


def copy_hashed(source, destination):
    digest = hashlib.sha256()
    with source.open('rb') as src, destination.open('xb') as dst:
        while block := src.read(2**20):
            digest.update(block)
            dst.write(block)
            os.posix_fadvise(src.fileno(), src.tell()-len(block), len(block), os.POSIX_FADV_DONTNEED)
        dst.flush()
        os.posix_fadvise(dst.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return {'bytes': destination.stat().st_size, 'sha256': digest.hexdigest()}


def prepare(output, source_image, owner=None):
    if not re.fullmatch('sha256:[0-9a-f]{64}', source_image):
        raise ValueError('Use an exact locally installed donor image ID')
    output = output.absolute()
    if output.exists() or output.is_symlink() or output.resolve() != output:
        raise ValueError('Use a new unredirected build-context directory')
    node = json.loads(subprocess.check_output(['docker', 'image', 'inspect', source_image]))[0]
    text = dockerfile(node)
    output.mkdir(parents=True)
    (output/'Dockerfile').write_text(text)
    (output/'.dockerignore').write_text('*\n!Dockerfile\n!kernel-cache.tar\n!runtime-source.tar.gz\n!runtime-assets.json\n')
    row = copy_hashed(ROOT/'artifacts/ds41-runtime-cache-v1.tar', output/'kernel-cache.tar')
    if row['sha256'] != CACHE_SHA:
        raise ValueError('Cache archive differs from the reviewed serving artifact')
    source = output/'runtime-source.tar.gz'
    with tarfile.open(source, 'w:gz', compresslevel=1) as archive:
        for path in sorted((ROOT/'release/runtime').rglob('*')):
            if path.is_symlink():
                raise ValueError('Redirected corresponding source')
            if path.is_file() and '__pycache__' not in path.parts:
                archive.add(path, arcname='runtime/' + path.relative_to(ROOT/'release/runtime').as_posix(), recursive=False)
        for name in ('LICENSE', 'CREDITS.md', 'THIRD_PARTY_NOTICES.md', 'release/package_ghcr.py', 'release/registry.py'):
            archive.add(ROOT/name, arcname=name, recursive=False)
    source_row = {'bytes': source.stat().st_size, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    assets = {'files': {'kernel-cache.tar': row, 'runtime-source.tar.gz': source_row},
        'cache_manifest_sha256': CACHE_MANIFEST_SHA,
        'donor_image_id': source_image, 'fresh_clone_gpu_tested': False,
        'packaging': 'flattened_existing_binaries_not_framework_rebase'}
    (output/'runtime-assets.json').write_text(json.dumps(assets, indent=2) + '\n')
    target = image_name(owner) if owner else None
    plan = {'status': 'ghcr_context_prepared_not_built_or_uploaded', 'source_image': source_image,
        'target_image': target, 'context': str(output), 'assets': assets,
        'publication_requires': ['authenticated GitHub owner', 'build and content audit',
            'push and immutable digest', 'public anonymous-pull verification', 'update frozen image identity and recipe lock'],
        'server_restarted': False}
    (output/'publication-plan.local.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({key: value for key, value in plan.items() if key != 'assets'}, indent=2))
    return plan


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-image', required=True)
    p.add_argument('--owner', help='Lowercase GitHub username, when known')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.owner:
        image_name(a.owner)
    prepare(a.output, a.source_image, a.owner)
