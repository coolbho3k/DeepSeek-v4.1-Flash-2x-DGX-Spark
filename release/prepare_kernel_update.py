# SPDX-License-Identifier: AGPL-3.0-only
"""Promote the tested kernel delta and append release assets to the existing OCI image.

No Docker lifecycle, compilation, registry writes, or live-server mutations.
The public kit's portable controller is preserved. Existing image layers are
hard-linked into a new layout, never overwritten or re-exported from serving.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

from repack_oci import Layers, encoded

ROOT = Path(__file__).resolve().parents[1]
BASE_DIGEST = 'sha256:8cc05f677a94367d56a058c0bd93742b51b7e1596953e61bcd9fabcc0ffc9fde'
CANDIDATE_SHA = 'b81ec8d744c579d57b530ab0d22fa446b2812b113ad6ca1d2511405dd14abfe3'
EXTENSIONS = {'.ttir', '.json', '.source', '.cubin', '.ptx', '.ttgir', '.llir'}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(2**20):
            digest.update(block)
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return digest.hexdigest()


def prepare(output, tag):
    if output.exists() or output.resolve() != output:
        raise ValueError('Use a fresh canonical output directory')
    if not tag.startswith('ghcr.io/coolbho3k/deepseek-v4.1-flash-exl3-3bpw-2x-dgx-spark:'):
        raise ValueError('Unexpected repository')
    parent = ROOT / 'artifacts/ds41-runtime-recipe-v21'
    candidate = ROOT / 'artifacts/ds41-runtime-kernel-batch-v1'
    public = ROOT / 'release/runtime'
    if sha(candidate / 'bundle-manifest.json') != CANDIDATE_SHA:
        raise ValueError('Changed tested runtime')
    old = json.loads((parent / 'bundle-manifest.json').read_bytes())['files']
    new = json.loads((candidate / 'bundle-manifest.json').read_bytes())['files']
    changed = {name: row for name, row in new.items() if row != old.get(name)}
    for name, row in changed.items():
        if sha(candidate / name) != row['sha256']:
            raise ValueError('Changed candidate payload: ' + name)
        if name in old and sha(public / name) != old[name]['sha256']:
            raise ValueError('Public portability change overlaps candidate: ' + name)
        if name not in old and (public / name).exists():
            raise ValueError('Preserve preexisting public file: ' + name)
    for name in ('kernel-batch-serving-v106.json', 'kernel-batch-upstream-c1-v106.json', 'kernel-batch-smoke-v106.json'):
        report = json.loads((ROOT / 'reports' / name).read_bytes())
        if report['status'] not in ('complete', 'complete_server_left_running'):
            raise ValueError('Incomplete serving qualification')
    output.mkdir(parents=True)
    backup = output / 'previous-public-kit'
    shutil.copytree(public, backup)
    for name in changed:
        path = public / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate / name, path)

    base_cache = ROOT / 'artifacts/ds41-runtime-cache-v1'
    live = ROOT / 'artifacts/ds41-portable-runs-v1/ds41-release-v106/node0/cache'
    manifest = json.loads((base_cache / 'cache-manifest.json').read_bytes())
    cache = output / 'cache'
    cache.mkdir()
    for name, row in manifest['files'].items():
        source = base_cache / name
        if source.is_symlink() or sha(source) != row['sha256']:
            raise ValueError('Changed base cache: ' + name)
        target = cache / name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)
    added = []
    for source in sorted((live / 'triton').rglob('*')):
        if source.is_symlink():
            raise ValueError('Redirected live compiler cache')
        if not source.is_file():
            continue
        name = source.relative_to(live).as_posix()
        if name in manifest['files']:
            continue
        if source.suffix not in EXTENSIONS or len(source.relative_to(live).parts) != 3:
            raise ValueError('Unrecognized new compiler-cache entry: ' + name)
        before = source.stat()
        target = cache / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or sha(source) != sha(target):
            raise ValueError('Compiler cache changed during snapshot: ' + name)
        manifest['files'][name] = dict(bytes=target.stat().st_size, mtime_ns=target.stat().st_mtime_ns, sha256=sha(target))
        added.append(name)
    raw = encoded(manifest)
    (cache / 'cache-manifest.json').write_bytes(raw)
    manifest_sha = hashlib.sha256(raw).hexdigest()
    context = output / 'context'
    context.mkdir()
    archive_path = context / 'kernel-cache.tar'
    with tarfile.open(archive_path, 'w', copybufsize=2**20) as archive:
        for path in sorted(cache.rglob('*')):
            if not path.is_file():
                continue
            member = archive.gettarinfo(str(path), arcname=path.relative_to(cache).as_posix())
            member.uid = member.gid = 0
            member.uname = member.gname = ''
            with path.open('rb') as stream:
                archive.addfile(member, stream)
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            archive.members.clear()
    cache_row = dict(bytes=archive_path.stat().st_size, sha256=sha(archive_path))
    requirements = json.loads((public / 'runtime-requirements.json').read_bytes())
    requirements['kernel_batch_candidate'].update(full_model_qualified=True, public_fresh_clone_gpu_qualified=False,
        serving_evidence='deployment_v106_matched_decode_prefill_image_tools_c6')
    requirements['auxiliary_cache_archive'].update(file='kernel-cache.tar', files=len(manifest['files']),
        bytes=cache_row['bytes'], sha256=cache_row['sha256'], manifest_sha256=manifest_sha,
        payload_bytes=sum(row['bytes'] for row in manifest['files'].values()))
    (public / 'runtime-requirements.json').write_bytes(encoded(requirements))

    base = ROOT / 'artifacts/ds41-ghcr-oci-v1'
    original_index = json.loads((base / 'index.json').read_bytes())
    descriptor = original_index['manifests'][0]
    if descriptor['digest'] != BASE_DIGEST:
        raise ValueError('Unexpected previously published base image')
    layout = output / 'oci'
    blobs = layout / 'blobs/sha256'
    blobs.mkdir(parents=True)
    old_blobs = base / 'blobs/sha256'
    image_manifest = json.loads((old_blobs / BASE_DIGEST.split(':')[1]).read_bytes())
    for row in [descriptor, image_manifest['config']] + image_manifest['layers']:
        source = old_blobs / row['digest'].split(':')[1]
        if source.is_symlink() or source.stat().st_size != row['size']:
            raise ValueError('Invalid base blob')
        os.link(source, blobs / source.name)
    config = json.loads((blobs / image_manifest['config']['digest'].split(':')[1]).read_bytes())
    layers = Layers(output / 'cache-layer')
    member = tarfile.TarInfo('opt/ds41-release/kernel-cache.tar')
    member.size = cache_row['bytes']
    member.mode = 0o644
    with archive_path.open('rb') as stream:
        layers.add(member, stream)
    layers.finish()
    for path in layers.blobs.iterdir():
        os.link(path, blobs / path.name)
    image_manifest['layers'] += layers.manifests
    config['rootfs']['diff_ids'] += layers.diffids
    config['history'] += [{'created_by': 'ds41 tested kernel cache update; no live-container commit'}]

    def blob(value, media):
        data = encoded(value)
        digest = hashlib.sha256(data).hexdigest()
        (blobs / digest).write_bytes(data)
        return dict(mediaType=media, digest='sha256:' + digest, size=len(data))

    image_manifest['config'] = blob(config, 'application/vnd.oci.image.config.v1+json')
    new_descriptor = blob(image_manifest, 'application/vnd.oci.image.manifest.v1+json')
    (layout / 'index.json').write_bytes(encoded(dict(schemaVersion=2, manifests=[dict(new_descriptor,
        annotations={'org.opencontainers.image.ref.name': tag, 'io.containerd.image.name': tag})])))
    (layout / 'oci-layout').write_bytes(encoded(dict(imageLayoutVersion='1.0.0')))
    assets = json.loads((base / 'publication/runtime-assets.json').read_bytes())
    assets['files']['kernel-cache.tar'] = cache_row
    assets['cache_manifest_sha256'] = manifest_sha
    assets['kernel_batch'] = dict(online_decode_attention=True, length_aware_radix_topk=True)
    (context / 'runtime-assets.json').write_bytes(encoded(assets))
    report = dict(status='prepared_source_finalization_and_audit_required', changed_kit_files=sorted(changed),
        candidate_manifest_sha256=CANDIDATE_SHA, base_image_digest=BASE_DIGEST,
        new_compiler_cache_files=len(added), total_cache_files=len(manifest['files']),
        cache_manifest_sha256=manifest_sha, cache=cache_row, server_touched=False)
    (output / 'preparation.json').write_bytes(encoded(report))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    prepare(args.output, args.tag)
