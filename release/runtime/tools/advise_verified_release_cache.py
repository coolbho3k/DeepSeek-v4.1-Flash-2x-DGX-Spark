"""Release only clean pages of same-host fully verified public model inputs.

Default is a read-only plan. --execute needs idle Docker/GPU and a fresh
private receipt. No payload reads, file edits/deletions or global cache flush.
This helper covers the optional bound model view, not an upload directory.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

sys.dont_write_bytecode = True
MAPPED_SHA = '7788d874a68294ef2df0b0a3ba63d39a70953dad4596c6a7536b561934d568ca'
source = Path(__file__).resolve().with_name('verify_mapped_release.py')
if hashlib.sha256(source.read_bytes()).hexdigest() != MAPPED_SHA:
    raise ValueError('Unreviewed bound-release verifier')
spec = importlib.util.spec_from_file_location('ds41_verified_cache_inputs', source)
mapped = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mapped)
flat = mapped.flat


def memory():
    return {parts[0][:-1]: int(parts[1])*1024
            for line in Path('/proc/meminfo').read_text().splitlines()
            if (parts := line.split())[0] in ('MemTotal:', 'MemFree:', 'MemAvailable:')}


def idle():
    for command in (['docker', 'ps', '-q'],
                    ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']):
        if subprocess.check_output(command, text=True, timeout=15).strip():
            raise ValueError('Clean-page advice requires idle local Docker and GPU')
    if memory()['MemAvailable'] < 48*2**30:
        raise ValueError('Preserve48GiB available RAM before input-cache advice')


def targets(directory, manifest, summary, bindings, receipt):
    mapped.check_receipt(directory, manifest, summary, bindings, receipt)
    fingerprints = receipt['fingerprints']
    return [(name, mapped.absolute(bindings[name]) if name in bindings else directory/name,
             fingerprints['bound_sources' if name in bindings else 'view_files'][name])
            for name in sorted(fingerprints['view_files'])]


def advise(rows):
    # Validate every target before advising any; recheck opened descriptors to
    # catch replacements between inventory and open. Advisory cache release
    # leaves dirty/mapped pages alone and does not promise a reclaimed amount.
    for name, path, expected in rows:
        if flat.regular(path) != expected:
            raise ValueError('Verified input changed before advice: '+name)
    advised = 0
    for name, path, expected in rows:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            if flat.fingerprint(os.fstat(fd)) != expected:
                raise ValueError('Verified input changed at open: '+name)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            if flat.fingerprint(os.fstat(fd)) != expected or flat.regular(path) != expected:
                raise ValueError('Verified input changed during advice: '+name)
            advised += 1
        finally:
            os.close(fd)
    return advised


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--bindings', required=True, type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    # Docker's Go CLI reserves a large virtual address range. Run the idle
    # observations BEFORE limiting this Python verifier's address space;
    # no subprocess is launched after the bound is applied.
    if args.execute:
        idle()
    resource.setrlimit(resource.RLIMIT_AS, (256*2**20, 256*2**20))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    directory = mapped.absolute(str(args.directory))
    manifest, summary = flat.load_manifest(args.manifest.absolute(), args.manifest_sha256)
    bindings, _ = flat.small_json(args.bindings.absolute())
    receipt, receipt_raw = flat.small_json(args.receipt.absolute())
    rows = targets(directory, manifest, summary, bindings, receipt)
    if args.execute and args.output is None:
        parser.error('--execute requires a fresh private --output')
    if args.output is not None:
        mapped.fresh(args.output.absolute(), directory)
        if any(args.output.absolute().is_relative_to(path.parent) for _, path, _ in rows):
            raise ValueError('Private advice receipt must stay outside preserved model inputs')
    result = dict(status='verified_release_cache_advice_planned', files=len(rows),
        input_bytes=sum(stamp[3] for _, _, stamp in rows),
        source_receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(),
        manifest_sha256=summary['manifest_sha256'], bindings_sha256=mapped.digest(bindings),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        memory_before=memory(), files_advised=0, disk_files_modified=False,
        payload_hashes_reread=False, global_cache_flush=False)
    if args.execute:
        result['files_advised'] = advise(rows)
        mapped.check_receipt(directory, manifest, summary, bindings, receipt)
        result.update(status='verified_release_clean_pages_advised',
                      fingerprints_unchanged=True, memory_after=memory())
    if args.output is not None:
        with args.output.absolute().open('xb') as stream:
            stream.write(mapped.encoded(result))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
