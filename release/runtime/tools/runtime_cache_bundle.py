"""Offline, bounded warm-cache transport with exact hashes and nanosecond mtimes.

Packs only compiler/autotune cache roots, never HF cache, temp files or logs.
Does not unpickle, import kernels, start containers, or change source caches.
Pack/extract require an idle GPU,48GiB RAM and32GiB free-disk reserve.
An integrity-checked archive is not publication/privacy/licensing approval.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile

from export_runtime_image import resources, validate_resources, fresh_path, exclusive

GIB = 2**30
ROOTS = {'flashinfer','tilelang','triton','torch-extensions','cuda','vllm',
         'exllamav3','numba','torch'}
MANIFEST_LIMIT = 16*2**20
FORMAT = 'ds41_auxiliary_runtime_cache_v1'


def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique(pairs):
    value = {}
    for key,item in pairs:
        if key in value:
            raise ValueError('Duplicate manifest key')
        value[key] = item
    return value


def relative(name):
    if not isinstance(name,str) or not name or len(name) > 1024:
        raise ValueError('Invalid cache entry name')
    p = PurePosixPath(name)
    if (str(p) != name or p.is_absolute() or '..' in p.parts or p.parts[0] not in ROOTS
            or any(c in name for c in ('\\','\0','\n','\r')) or len(p.parts) < 2
            or name.endswith(('.lock','.log')) or p.name == '.ninja_lock'):
        raise ValueError('Cache path is outside the public compiler allowlist')
    return name


def fingerprint(path):
    info = path.lstat()
    if path.resolve() != path or not stat.S_ISREG(info.st_mode):
        raise ValueError('Only unredirected regular cache files are supported')
    return [info.st_dev,info.st_ino,info.st_mode,info.st_size,info.st_mtime_ns,
            info.st_ctime_ns,info.st_nlink]


def inventory(source):
    if source.resolve() != source or not source.is_dir():
        raise ValueError('Use an existing unredirected cache directory')
    files = {}
    for root in sorted(ROOTS):
        base = source/root
        if not base.exists():
            continue
        if base.resolve() != base or not base.is_dir():
            raise ValueError('Redirected compiler cache root')
        for directory,dirs,names in os.walk(base,followlinks=False):
            for name in dirs:
                path = Path(directory)/name
                if path.resolve() != path:
                    raise ValueError('Redirected compiler cache subdirectory')
            for name in names:
                if name.endswith(('.lock','.log')) or name == '.ninja_lock':
                    continue
                path = Path(directory)/name
                key = relative(path.relative_to(source).as_posix())
                info = fingerprint(path)
                if info[3] > 512*2**20 or len(files) >= 100000:
                    raise ValueError('Compiler cache inventory exceeds bounded limits')
                files[key] = info
    if not files or sum(row[3] for row in files.values()) > 8*GIB:
        raise ValueError('Expected a nonempty compiler cache of at most8GiB')
    return dict(sorted(files.items()))


def hash_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def manifest_from_source(source,files):
    rows = {}
    for name,stamp in files.items():
        path = source/name
        if fingerprint(path) != stamp:
            raise ValueError('Source cache changed before hashing')
        digest = hash_file(path)
        if fingerprint(path) != stamp:
            raise ValueError('Source cache changed during hashing')
        rows[name] = dict(bytes=stamp[3],sha256=digest,mtime_ns=stamp[4])
    raw = encoded(dict(format=FORMAT,files=rows))
    validate_manifest(raw,sha(raw))
    return raw


def validate_manifest(raw,expected_sha):
    if (len(raw) > MANIFEST_LIMIT or not re.fullmatch('[0-9a-f]{64}',expected_sha)
            or sha(raw) != expected_sha):
        raise ValueError('Cache manifest differs from its independent bounded identity')
    def reject(value):
        raise ValueError('Nonfinite manifest number')
    value = json.loads(raw,object_pairs_hook=unique,parse_constant=reject)
    if set(value) != {'format','files'} or value['format'] != FORMAT:
        raise ValueError('Unsupported cache manifest')
    files = value['files']
    if not isinstance(files,dict) or not 1 <= len(files) <= 100000:
        raise ValueError('Invalid cache inventory count')
    for name,row in files.items():
        relative(name)
        if (not isinstance(row,dict) or set(row) != {'bytes','sha256','mtime_ns'}
                or type(row['bytes']) is not int or not 0 <= row['bytes'] <= 512*2**20
                or type(row['mtime_ns']) is not int or not 0 <= row['mtime_ns'] < 2**63
                or not isinstance(row['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',row['sha256'])):
            raise ValueError('Invalid cache size/hash/mtime')
    if sum(row['bytes'] for row in files.values()) > 8*GIB:
        raise ValueError('Cache exceeds8GiB staging budget')
    return value


class CheckedReader:
    def __init__(self,stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self,size=-1):
        raw = self.stream.read(size)
        self.digest.update(raw)
        return raw


def write_archive(source,files,manifest_raw,output):
    manifest = validate_manifest(manifest_raw,sha(manifest_raw))
    with output.open('xb') as stream, tarfile.open(fileobj=stream,mode='w',format=tarfile.PAX_FORMAT) as archive:
        info = tarfile.TarInfo('cache-manifest.json')
        info.size = len(manifest_raw)
        info.mode = 0o644
        archive.addfile(info,io.BytesIO(manifest_raw))
        for name,row in manifest['files'].items():
            path = source/name
            if fingerprint(path) != files[name]:
                raise ValueError('Source changed before archiving; preserve incomplete archive')
            info = tarfile.TarInfo(name)
            info.mode = 0o644
            info.size = row['bytes']
            seconds,nanos = divmod(row['mtime_ns'],10**9)
            info.mtime = seconds
            info.pax_headers = {'mtime':f'{seconds}.{nanos:09d}'}
            with path.open('rb') as source_stream:
                checked = CheckedReader(source_stream)
                archive.addfile(info,checked)
                if checked.digest.hexdigest() != row['sha256']:
                    raise ValueError('Archived cache payload changed')
            if fingerprint(path) != files[name]:
                raise ValueError('Source changed while archiving')
    if inventory(source) != files:
        raise ValueError('Source cache inventory changed during packing')


def pack(source,output,receipt):
    fresh_path(output)
    fresh_path(receipt)
    if output == receipt or output.is_relative_to(source) or receipt.is_relative_to(source):
        raise ValueError('Archive and private receipt must be separate from the preserved source')
    validate_resources(resources(output.parent))
    files = inventory(source)
    size = sum(row[3] for row in files.values()) + len(files)*4096 + MANIFEST_LIMIT
    if shutil.disk_usage(output.parent).free < 32*GIB+size:
        raise ValueError('Cache archive must preserve32GiB free disk')
    raw = manifest_from_source(source,files)
    write_archive(source,files,raw,output)
    result = dict(status='compiler_cache_archive_prepared',archive=str(output),
        files=len(files),payload_bytes=sum(row[3] for row in files.values()),
        archive_bytes=output.stat().st_size,archive_sha256=hash_file(output),
        manifest_sha256=sha(raw),manifest_bytes=len(raw),mtime_preserved=True,
        source_cache_changed=False,publication_approved=False,
        payload_privacy_review_complete=False,physical_deployment_verified=False,
        source_sha256=hash_file(Path(__file__).resolve()))
    exclusive(receipt,result)
    return result


def extract_archive(archive,archive_sha,manifest_sha,destination):
    fresh_path(destination)
    if archive.resolve() != archive or not archive.is_file() or archive.stat().st_size > 9*GIB:
        raise ValueError('Use a bounded unredirected cache archive')
    if not re.fullmatch('[0-9a-f]{64}',archive_sha) or hash_file(archive) != archive_sha:
        raise ValueError('Cache archive SHA256 mismatch')
    with tarfile.open(archive,mode='r|') as stream:
        first = stream.next()
        if (first is None or first.name != 'cache-manifest.json' or not first.isfile()
                or first.size > MANIFEST_LIMIT):
            raise ValueError('Cache archive must begin with a bounded regular manifest')
        raw = stream.extractfile(first).read(MANIFEST_LIMIT+1)
        manifest = validate_manifest(raw,manifest_sha)
        size = sum(row['bytes'] for row in manifest['files'].values())
        if shutil.disk_usage(destination.parent).free < 32*GIB+size+64*2**20:
            raise ValueError('Extracted cache must preserve32GiB free disk')
        destination.mkdir()
        with (destination/'cache-manifest.json').open('xb') as output:
            output.write(raw)
        seen = set()
        while True:
            item = stream.next()
            if item is None:
                break
            name = relative(item.name)
            if name in seen or name not in manifest['files'] or not item.isfile():
                raise ValueError('Unexpected, duplicate or nonregular cache archive member')
            row = manifest['files'][name]
            if item.size != row['bytes']:
                raise ValueError('Cache archive member size differs from pinned manifest')
            target = destination/name
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.parent.resolve() != target.parent:
                raise ValueError('Redirected extraction parent')
            digest = hashlib.sha256()
            with stream.extractfile(item) as source, target.open('xb') as output:
                while raw := source.read(2**20):
                    output.write(raw)
                    digest.update(raw)
            if digest.hexdigest() != row['sha256']:
                raise ValueError('Extracted cache payload SHA256 mismatch')
            os.chmod(target,0o644)
            os.utime(target,ns=(row['mtime_ns'],row['mtime_ns']))
            if target.stat().st_mtime_ns != row['mtime_ns']:
                raise ValueError('Filesystem did not preserve required cache mtime')
            seen.add(name)
        if seen != set(manifest['files']):
            raise ValueError('Cache archive lacks required payloads')
    return dict(status='compiler_cache_archive_extracted_verified',files=len(seen),
        payload_bytes=size,archive_sha256=archive_sha,manifest_sha256=manifest_sha,
        mtime_preserved=True,physical_deployment_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action',required=True)
    p = actions.add_parser('pack')
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--receipt',required=True,type=Path)
    p = actions.add_parser('extract')
    p.add_argument('--archive',required=True,type=Path)
    p.add_argument('--archive-sha256',required=True)
    p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--directory',required=True,type=Path)
    p.add_argument('--receipt',required=True,type=Path)
    args = parser.parse_args()
    if args.action == 'pack':
        result = pack(args.source.absolute(),args.output.absolute(),args.receipt.absolute())
    else:
        fresh_path(args.receipt.absolute())
        validate_resources(resources(args.directory.absolute().parent))
        result = extract_archive(args.archive.absolute(),args.archive_sha256,
            args.manifest_sha256,args.directory.absolute())
        exclusive(args.receipt.absolute(),result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
