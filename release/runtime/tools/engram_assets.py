# SPDX-License-Identifier: AGPL-3.0-only
"""Portable immutable page15 downloads and cheap, fail-closed startup checks.

Only preparation downloads/hashes payloads. Serving mounts its two rank-owned
files read-only and checks the pinned inventory, headers and local verification
receipt. The original HF snapshot and original Engram paths are never changed.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import tempfile
import urllib.request

MANIFEST_SHA = '43b3b6305d3edd8ec24b9014fccd842c623f7cfce41014611cbd85244dc62854'
PREFIX = 'engram-page15-v1'
MAGIC = 0x3247504531345344


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True)+'\n').encode()


def regular(path, limit=None):
    if (path.resolve() != path or not stat.S_ISREG(path.lstat().st_mode)
            or (limit is not None and path.stat().st_size > limit)):
        raise ValueError('Expected unredirected regular Engram asset')
    return path


def fingerprint(path):
    s = regular(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def load_manifest(root):
    raw = regular(root/'manifest.json', 65536).read_bytes()
    return parse_manifest(raw)


def parse_manifest(raw):
    """Use the same pinned schema for publication and runtime preparation."""
    if len(raw) > 65536:
        raise ValueError('Oversized packed Engram inventory')
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA:
        raise ValueError('Packed Engram inventory differs from the reader pin')
    m = json.loads(raw)
    if (m['format'] != 'ds41_engram_page15_release_v1' or m['layout'] != 'page15'
            or m['reader_abi'] != 2 or m['tensor_parallel_size'] != 2
            or m['row_bytes'] != 264 or m['page_bytes'] != 4096 or m['rows_per_page'] != 15
            or m['lossless'] is not True):
        raise ValueError('Unsupported packed Engram format')
    expected = {f'{PREFIX}/engram-layer-{layer:02}-rank{rank}-page15.bin'
                for rank in (0, 1) for layer in (1, 14)}
    if set(m['files']) != expected:
        raise ValueError('Packed inventory must contain exactly four tables')
    for name, row in m['files'].items():
        if (name != f'{PREFIX}/engram-layer-{row["layer"]:02}-rank{row["rank"]}-page15.bin'
                or not 0 <= row['lo'] < row['hi'] <= row['total_rows']
                or row['bytes'] != 4096 + ((row['hi']-row['lo']+14)//15)*4096
                or not re.fullmatch('[0-9a-f]{64}', row['sha256'])):
            raise ValueError('Malformed packed table descriptor')
        offset = 0
        for number, part in enumerate(row['parts']):
            if (part['path'] != name+f'.part-{number:05d}' or part['offset'] != offset
                    or type(part['bytes']) is not int or not 0 < part['bytes'] <= 8*2**30
                    or not re.fullmatch('[0-9a-f]{64}', part['sha256'])):
                raise ValueError('Malformed packed download part')
            offset += part['bytes']
        if offset != row['bytes']:
            raise ValueError('Incomplete packed download inventory')
    return m


def atomic_json(path, value):
    """Publish a small pointer only after all immutable inputs have verified."""
    if path.is_symlink():
        raise ValueError('Redirected preparation receipt')
    fd, temp = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(encoded(value)); out.flush(); os.fsync(out.fileno())
        os.replace(temp, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def reference(root, rank):
    return dict(root=str(root), rank=rank, manifest_sha256=MANIFEST_SHA)


def check_reference(ref, rank):
    if (type(rank) is not int or rank not in (0, 1) or not isinstance(ref, dict)
            or set(ref) != {'root', 'rank', 'manifest_sha256'} or ref['rank'] != rank
            or type(ref['rank']) is not int
            or ref['manifest_sha256'] != MANIFEST_SHA):
        raise ValueError('Packed Engram rank or manifest does not match this reader')
    root = Path(ref['root'])
    if (not root.is_absolute() or str(root) != ref['root'] or '..' in root.parts
            or len(root.parts) < 3 or any(c in str(root) for c in (',', '\0', '\n', '\r'))):
        raise ValueError('Use an explicit normalized packed asset directory')
    return root


def mounts(ref, rank):
    root = check_reference(ref, rank)
    return [(str(root/f'engram-layer-{layer:02}-rank{rank}-page15.bin'),
             f'/opt/ds41-engram-packed/engram-layer-{layer:02}-rank{rank}-page15.bin', True)
            for layer in (1, 14)]


def check_files(root, manifest, rank):
    result = {}
    for name, row in manifest['files'].items():
        if row['rank'] != rank: continue
        path = root/Path(name).name
        info = fingerprint(path)
        if info[2] != row['bytes']:
            raise ValueError('Packed Engram size changed')
        with path.open('rb') as source:
            header = source.read(64)
        expected = (MAGIC, row['layer'], row['lo'], row['hi'], row['total_rows'], 264, 15, 4096)
        if header != struct.pack('<8Q', *expected):
            raise ValueError('Packed Engram format or rank header differs')
        result[path.name] = info
    return result


def validate(ref, rank):
    root = check_reference(ref, rank)
    manifest = load_manifest(root)
    receipt = json.loads(regular(root/'verified.json', 65536).read_bytes())
    if (receipt.get('format') != 'ds41_verified_engram_page15_v1' or receipt.get('rank') != rank
            or receipt.get('manifest_sha256') != MANIFEST_SHA
            or receipt.get('files') != check_files(root, manifest, rank)):
        raise ValueError('Packed Engrams changed after full SHA256 verification; inspect before retrying')
    return dict(layout='page15', rank=rank, files=2, manifest_sha256=MANIFEST_SHA)


def download_table(base, row, path, opener=urllib.request.urlopen):
    """Resume directly into one assembly file; no second copy of huge parts.

Every part and the full assembled file are SHA256 checked. An interruption
leaves only a .download-part, never a usable published table. Resumption
rehashes the existing prefix before continuing the current HTTP byte range.
    """
    if path.exists():
        regular(path)
        source_path = path
    else:
        source_path = path.with_name(path.name+'.download-part')
    if source_path.is_symlink():
        raise ValueError('Redirected packed download')
    if source_path.exists():
        regular(source_path)
    whole = hashlib.sha256()
    with source_path.open('rb' if source_path == path else 'r+b' if source_path.exists() else 'x+b') as out:
        existing = os.fstat(out.fileno()).st_size
        if existing > row['bytes'] or (source_path == path and existing != row['bytes']):
            raise ValueError('Unexpected assembled Engram size')
        for part in row['parts']:
            digest = hashlib.sha256()
            present = max(0, min(part['bytes'], existing-part['offset']))
            out.seek(part['offset'])
            remaining = present
            while remaining:
                block = out.read(min(2**20, remaining))
                if not block: raise ValueError('Truncated packed download prefix')
                digest.update(block); whole.update(block); remaining -= len(block)
                os.posix_fadvise(out.fileno(), out.tell()-len(block), len(block), os.POSIX_FADV_DONTNEED)
            if present < part['bytes']:
                if source_path == path:
                    raise ValueError('Published table is incomplete')
                headers = {'Range': f'bytes={present}-'} if present else {}
                request = urllib.request.Request(base+part['path'], headers=headers)
                with opener(request, timeout=120) as response:
                    if present and (response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {present}-')):
                        raise ValueError('Server did not honor the packed resume range')
                    if not present and response.status != 200:
                        raise ValueError('Unexpected packed download response')
                    remaining = part['bytes']-present
                    while remaining:
                        block = response.read(min(2**20, remaining))
                        if not block: raise ValueError('Interrupted packed download')
                        out.write(block); out.flush()
                        digest.update(block); whole.update(block); remaining -= len(block)
                        os.posix_fadvise(out.fileno(), out.tell()-len(block), len(block), os.POSIX_FADV_DONTNEED)
                    if response.read(1): raise ValueError('Oversized packed download part')
                out.flush(); os.fsync(out.fileno())
            if digest.hexdigest() != part['sha256']:
                raise ValueError('Packed part checksum mismatch; preserve partial for inspection')
        if whole.hexdigest() != row['sha256']:
            raise ValueError('Assembled packed checksum mismatch')
    if source_path != path:
        os.link(source_path, path, follow_symlinks=False)  # Exclusive publication.
        source_path.unlink()  # Only our exact, now-published assembly inode.
    print(json.dumps(dict(stage='packed_table_verified', file=path.name, bytes=row['bytes'])), flush=True)


def prepare(spec, cache, rank, download):
    if (type(rank) is not int or rank not in (0, 1) or spec['manifest_sha256'] != MANIFEST_SHA
            or spec['manifest_path'] != PREFIX+'/manifest.json'
            or not re.fullmatch('[0-9a-f]{40}', spec['revision'])
            or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', spec['repo'])):
        raise ValueError('Immutable packed Engram publication required')
    root = cache/'engram-assets'/MANIFEST_SHA/f'rank{rank}'
    root.mkdir(parents=True, exist_ok=True)
    if root.resolve() != root:
        raise ValueError('Redirected packed asset cache')
    ref = reference(root, rank)
    if (root/'verified.json').exists():
        validate(ref, rank)
        return ref
    base = f'https://huggingface.co/{spec["repo"]}/resolve/{spec["revision"]}/'
    download(base+spec['manifest_path'], root/'manifest.json', MANIFEST_SHA)
    manifest = load_manifest(root)
    if manifest['repo_id'] != spec['repo']:
        raise ValueError('Unexpected packed Engram repository')
    for name, row in manifest['files'].items():
        if row['rank'] == rank:
            download_table(base, row, root/Path(name).name)
    atomic_json(root/'verified.json', dict(format='ds41_verified_engram_page15_v1',
        rank=rank, manifest_sha256=MANIFEST_SHA, files=check_files(root, manifest, rank)))
    validate(ref, rank)
    return ref
