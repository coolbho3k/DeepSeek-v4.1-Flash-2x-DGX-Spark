"""Verify an extracted runtime-input bundle using only Python's standard library.

This proves file integrity, NOT a runtime rebuild, GPU qualification or release
approval. Run from any directory; no original workspace or receipt is needed.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import resource
import stat

MAX_FILE = 8 * 2**20
MAX_TOTAL = 64 * 2**20


def relative(name):
    if (not isinstance(name, str) or not name or '\\' in name or '\0' in name
            or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts
            or str(PurePosixPath(name)) != name or name == '.'):
        raise ValueError('Unsafe or noncanonical bundle path')
    return name


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def read_regular(path):
    before = path.lstat()
    if (path.resolve() != path or not stat.S_ISREG(before.st_mode)
            or before.st_size > MAX_FILE):
        raise ValueError('Expected a bounded, unredirected regular file')
    data = path.read_bytes()
    after = path.lstat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if any(getattr(before, k) != getattr(after, k) for k in fields):
        raise ValueError('Input changed while being read')
    return data


def verify(directory, expected_manifest=None):
    directory = Path(directory).absolute()
    if directory.resolve() != directory or not directory.is_dir():
        raise ValueError('Use an unredirected extracted bundle directory')
    raw = read_regular(directory/'bundle-manifest.json')
    manifest_sha = hashlib.sha256(raw).hexdigest()
    if expected_manifest is not None and manifest_sha != expected_manifest:
        raise ValueError('Manifest differs from the separately supplied digest')
    manifest = json.loads(raw, object_pairs_hook=unique)
    if (manifest.get('format') not in ('ds41_runtime_inputs_v1', 'ds41_runtime_inputs_v2', 'ds41_runtime_inputs_v3', 'ds41_runtime_inputs_v4', 'ds41_runtime_inputs_v5')
            or manifest.get('standalone_runtime') is not False
            or manifest.get('clean_rebuild_qualified') is not False
            or manifest.get('publication_approved') is not False):
        raise ValueError('Unexpected bundle format or unsupported qualification claim')
    files = manifest['files']
    if not isinstance(files, dict) or not 1 <= len(files) <= 1000:
        raise ValueError('Invalid file inventory')
    actual = set()
    for path in directory.rglob('*'):
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or path.resolve() != path:
            raise ValueError('Symlinks and special files are not allowed')
        actual.add(relative(path.relative_to(directory).as_posix()))
    if actual != set(files) | {'bundle-manifest.json'}:
        raise ValueError('Missing or unexpected bundle files')
    total = 0
    for name, row in sorted(files.items()):
        relative(name)
        if (name == 'bundle-manifest.json' or type(row.get('bytes')) is not int
                or not 0 <= row['bytes'] <= MAX_FILE
                or not isinstance(row.get('sha256'), str)
                or not re.fullmatch('[0-9a-f]{64}', row['sha256'])):
            raise ValueError('Invalid file descriptor')
        total += row['bytes']
        if total > MAX_TOTAL:
            raise ValueError('Bundle exceeds its small-input scope')
        data = read_regular(directory/name)
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError(f'Bundle file changed: {name}')
    return dict(status='runtime_input_bundle_hash_verified', files=len(files),
                payload_bytes=total, manifest_sha256=manifest_sha,
                standalone_runtime=False, clean_rebuild_qualified=False,
                publication_approved=False, gpu_work_performed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--manifest-sha256')
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (256*2**20, 256*2**20))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    print(json.dumps(verify(args.directory, args.manifest_sha256), indent=2))


if __name__ == '__main__':
    main()
