"""Verify a materialized DS41 release against an independently supplied manifest hash.

Standalone standard-library tool: no workspace receipts, ML imports or network.
Default mode only inspects the manifest. --full streams ALL published payloads
with a 1MiB buffer; run it before loading models. --check-receipt only rechecks
local fingerprints from a previous full pass, and is not another payload hash.
Neither mode establishes model quality, publisher authenticity or approval.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import resource
import shutil
import stat
import subprocess

FORMAT = 'ds41_hf_weights_release_v1'
SOURCE = 'df42c109f1defefcbfcedbe7d905718a12266e40'
SELECTION = '4639e1cf64c5845bb89ae4f9b3648293014bb8c0c32daee2d19f7df0a3f5e86e'
PACKAGE = '7f7cfee4a7dc618196b0699a57034dce3316b83257a7320687d079d72d13ee73'
MAX_MANIFEST = 4 * 2**20
MAX_FILES = 1000
MAX_TOTAL = 512 * 2**30
BUFFER = 2**20
SHARDS = {f'model-{i:05d}-of-00051.safetensors' for i in range(1, 52)}
SHARDS |= {f'engrams/engram-layer-{i:02d}.safetensors' for i in (1, 14)}
SHARDS |= {f'draft/model-{i:05d}-of-00002.safetensors' for i in (1, 2)}


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def relative(name):
    if (not isinstance(name, str) or not name or '\\' in name or '\0' in name
            or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts
            or str(PurePosixPath(name)) != name or name == '.'):
        raise ValueError('Unsafe or noncanonical release path')
    if name == '.cache' or name.startswith('.cache/') or name == '.git' or name.startswith('.git/'):
        raise ValueError('Local cache and git files cannot be public payloads')
    return name


def fingerprint(info):
    return [info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns, info.st_nlink]


def regular(path):
    info = path.lstat()
    if path.resolve() != path or not stat.S_ISREG(info.st_mode):
        raise ValueError('Expected a materialized, unredirected regular file: '+str(path))
    return fingerprint(info)


def small_json(path):
    before = regular(path)
    if before[3] > MAX_MANIFEST:
        raise ValueError('Oversized JSON input')
    raw = path.read_bytes()
    if regular(path) != before:
        raise ValueError('JSON input changed while reading')
    def reject(value):
        raise ValueError('Nonfinite JSON value')
    return json.loads(raw, object_pairs_hook=unique, parse_constant=reject), raw


def load_manifest(path, expected_sha256):
    if not isinstance(expected_sha256, str) or not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('Supply an independent SHA256 for the public manifest')
    manifest, raw = small_json(path)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Public manifest differs from the supplied digest')
    if (manifest.get('format') != FORMAT or manifest.get('manifest_excludes_itself') is not True
            or manifest.get('source_revision') != SOURCE
            or manifest.get('selected_manifest_sha256') != SELECTION
            or manifest.get('candidate_manifest_sha256') != PACKAGE
            or manifest.get('publication_authorized') is not True
            or manifest.get('repo_id') != 'coolbho3k/DeepSeek-V4.1-Flash-EXL3-3bpw'):
        raise ValueError('Unsupported release identity or approval claim')
    files = manifest.get('files')
    if not isinstance(files, dict) or not 1 <= len(files) <= MAX_FILES:
        raise ValueError('Invalid public inventory')
    total = 0
    for name, row in files.items():
        relative(name)
        if (name == 'release-manifest.json' or not isinstance(row, dict)
                or set(row) != {'bytes', 'sha256'} or type(row['bytes']) is not int
                or not 0 <= row['bytes'] <= MAX_TOTAL
                or not isinstance(row['sha256'], str)
                or not re.fullmatch('[0-9a-f]{64}', row['sha256'])):
            raise ValueError('Invalid public file descriptor')
        if any(parent.as_posix() in files for parent in PurePosixPath(name).parents if str(parent) != '.'):
            raise ValueError('Public file/directory collision')
        total += row['bytes']
    if total > MAX_TOTAL or {name for name in files if name.endswith('.safetensors')} != SHARDS:
        raise ValueError('Incomplete/unsupported weight inventory or oversized release')
    required = {'config.json', 'model.safetensors.index.json', 'engrams/model.safetensors.index.json',
                'draft/model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json',
                'README.md', 'LICENSE', 'evaluation-summary.json'}
    if not required <= set(files):
        raise ValueError('Missing required public metadata')
    return manifest, dict(status='public_release_manifest_inspected', manifest_sha256=expected_sha256,
        files=len(files)+1, weight_shards=len(SHARDS), payload_bytes=total,
        weight_bytes=sum(files[name]['bytes'] for name in SHARDS),
        full_payload_hash_verified=False, publication_approved=False)


def inventory(directory, files):
    if directory.resolve() != directory or not directory.is_dir():
        raise ValueError('Use a materialized canonical model directory, not HF snapshot symlinks')
    actual = {}
    # Do not follow symlink directories, even the local HF download metadata.
    for base, directories, names in os.walk(directory, followlinks=False):
        for name in directories:
            path = Path(base)/name
            if path.is_symlink() or path.resolve() != path:
                raise ValueError('Redirected release directory')
        for name in names:
            path = Path(base)/name
            logical = path.relative_to(directory).as_posix()
            stamp = regular(path)
            # hf download --local-dir adds these bookkeeping files. They are
            # neither public payloads nor accepted sources for model files.
            if logical.startswith('.cache/huggingface/'):
                continue
            actual[relative(logical)] = stamp
            if len(actual) > MAX_FILES+1:
                raise ValueError('Too many release files')
    if set(actual) != set(files) | {'release-manifest.json'}:
        raise ValueError('Missing or unexpected release files')
    if any(actual[name][3] != row['bytes'] for name, row in files.items()):
        raise ValueError('Payload size differs from public manifest')
    return actual


def idle_before_full_hash():
    """Avoid a large page-cache workload beside a resident CUDA model."""
    if shutil.which('nvidia-smi'):
        jobs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
            '--format=csv,noheader'], text=True, timeout=15).strip()
        if jobs:
            raise ValueError('Full payload hashing requires idle GPUs; leave serving workers unchanged')


def hash_payload(path, expected_stamp):
    digest = hashlib.sha256()
    # The canonical-path check rejects symlink parents; O_NOFOLLOW also closes
    # the leaf symlink race. This is integrity checking, not a hostile-FS sandbox.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb', buffering=0) as stream:
        if fingerprint(os.fstat(stream.fileno())) != expected_stamp:
            raise ValueError('Payload changed before hashing')
        while chunk := stream.read(BUFFER):
            digest.update(chunk)
            os.posix_fadvise(stream.fileno(), stream.tell()-len(chunk), len(chunk), os.POSIX_FADV_DONTNEED)
        if fingerprint(os.fstat(stream.fileno())) != expected_stamp:
            raise ValueError('Payload changed during hashing')
    if regular(path) != expected_stamp:
        raise ValueError('Payload path changed during hashing')
    return digest.hexdigest()


def verify_full(directory, manifest, summary, progress=lambda row: None):
    before = inventory(directory, manifest['files'])
    expected = dict(manifest['files'])
    expected['release-manifest.json'] = dict(sha256=summary['manifest_sha256'])
    done = 0
    for name in sorted(expected):
        if hash_payload(directory/name, before[name]) != expected[name]['sha256']:
            raise ValueError('Payload SHA256 mismatch: '+name)
        done += before[name][3]
        progress(dict(file=name, bytes_hashed=done))
    if inventory(directory, manifest['files']) != before:
        raise ValueError('Release changed during full verification')
    return dict(format='ds41_downloaded_release_hash_receipt_v1',
        status='all_public_release_payloads_sha256_verified', directory=str(directory),
        manifest_sha256=summary['manifest_sha256'], files=before, bytes_hashed=done,
        time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        full_payload_hash_verified=True, publication_approved=False,
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def check_receipt(directory, manifest, summary, receipt):
    if (receipt.get('format') != 'ds41_downloaded_release_hash_receipt_v1'
            or receipt.get('status') != 'all_public_release_payloads_sha256_verified'
            or receipt.get('directory') != str(directory)
            or receipt.get('manifest_sha256') != summary['manifest_sha256']
            or receipt.get('full_payload_hash_verified') is not True
            or receipt.get('verifier_sha256') != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise ValueError('Full verification receipt does not cover these inputs/tool')
    actual = inventory(directory, manifest['files'])
    if actual != receipt.get('files'):
        raise ValueError('Locally verified release changed; full verification is required again')
    if hash_payload(directory/'release-manifest.json', actual['release-manifest.json']) != summary['manifest_sha256']:
        raise ValueError('Downloaded manifest changed')
    return dict(status='previous_full_hash_local_fingerprints_unchanged',
        manifest_sha256=summary['manifest_sha256'], files=len(actual),
        current_payload_hash_pass=False, previous_full_payload_hash_verified=True,
        publication_approved=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--directory', type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--full', action='store_true')
    modes.add_argument('--check-receipt', type=Path)
    parser.add_argument('--output', type=Path, help='Fresh PRIVATE receipt, outside the release directory')
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (256*2**20, 256*2**20))
    if bool(args.directory) != bool(args.full or args.check_receipt) or bool(args.output) != args.full:
        parser.error('--full requires --directory and --output; --check-receipt requires --directory')
    manifest, result = load_manifest(args.manifest.absolute(), args.manifest_sha256)
    if args.directory:
        directory = args.directory.absolute()
        if args.full:
            output = args.output.absolute()
            if (output.resolve() != output or output.exists() or output.is_symlink()
                    or output.is_relative_to(directory) or not output.parent.is_dir()):
                raise ValueError('Use a fresh private receipt outside the downloaded release')
            # Fail before reading ANY weight payload or writing an output.
            idle_before_full_hash()
            result = verify_full(directory, manifest, result,
                progress=lambda row: print(json.dumps(dict(stage='payload_hashed', **row)), flush=True))
            with output.open('x') as stream:
                json.dump(result, stream, indent=2, sort_keys=True)
                stream.write('\n')
            result = {key: value for key, value in result.items() if key != 'files'} | {'files': len(result['files'])}
        else:
            receipt, _ = small_json(args.check_receipt.absolute())
            result = check_receipt(directory, manifest, result, receipt)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
