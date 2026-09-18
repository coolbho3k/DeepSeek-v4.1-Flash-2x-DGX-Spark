"""Verify ALL public release files through explicit read-only shard bindings.

The host view contains real metadata plus empty shard mount targets. It is
NOT an upload directory. Docker must bind every verified original shard onto
its exact /model path. No source links, copies, renames or mutations occur.
The strict flat-directory verifier remains unchanged and is the default path.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import shutil
import sys

sys.dont_write_bytecode = True
FLAT_SHA = 'c2da0b3e0d7f9c27f626c00aa5316686fa19c773942e4be3e99f2956cbb3a458'
path = Path(__file__).resolve().with_name('verify_downloaded_release.py')
if hashlib.sha256(path.read_bytes()).hexdigest() != FLAT_SHA:
    raise ValueError('Unreviewed materialized-release verifier')
spec = importlib.util.spec_from_file_location('ds41_flat_release_check',path)
flat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flat)


def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def absolute(value):
    if (not isinstance(value,str) or not value.startswith('/') or len(Path(value).parts) < 4
            or str(Path(value)) != value or '..' in Path(value).parts
            or any(c in value for c in ('\0','\n','\r',','))):
        raise ValueError('Use an explicit normalized file path suitable for a read-only Docker bind')
    return Path(value)


def bindings_checked(directory,bindings):
    if not isinstance(bindings,dict) or set(bindings) != flat.SHARDS:
        raise ValueError('Every one of the55 published shards must have exactly one binding')
    for value in bindings.values():
        source = absolute(value)
        if source.is_relative_to(directory):
            raise ValueError('Mapped shards must be outside the metadata/mount-target view')
    if len(set(bindings.values())) != len(bindings):
        raise ValueError('Distinct public shards must have distinct explicit source files')
    return bindings


def inventory(directory,manifest,bindings):
    bindings_checked(directory,bindings)
    # Reuse strict directory/path/type checking, but require the mount targets
    # themselves to be empty. Their data is verified at the bound source below.
    targets = {name:row | ({'bytes':0} if name in bindings else {})
               for name,row in manifest['files'].items()}
    files = flat.inventory(directory,targets)
    sources = {name:flat.regular(absolute(source)) for name,source in bindings.items()}
    if any(stamp[3] != manifest['files'][name]['bytes'] for name,stamp in sources.items()):
        raise ValueError('Bound source size differs from the public shard manifest')
    return dict(view_files=files,bound_sources=sources)


def verify_full(directory,manifest,summary,bindings,progress=lambda row:None):
    before = inventory(directory,manifest,bindings)
    expected = dict(manifest['files'])
    expected['release-manifest.json'] = dict(sha256=summary['manifest_sha256'])
    done = 0
    for name,row in sorted(expected.items()):
        bound = name in bindings
        path = absolute(bindings[name]) if bound else directory/name
        stamp = before['bound_sources' if bound else 'view_files'][name]
        if flat.hash_payload(path,stamp) != row['sha256']:
            raise ValueError('Public payload SHA256 mismatch: '+name)
        done += stamp[3]
        progress(dict(file=name,bytes_hashed=done,bound_shard=bound))
    if inventory(directory,manifest,bindings) != before:
        raise ValueError('Public view or shard sources changed during full verification')
    return dict(format='ds41_bound_release_hash_receipt_v1',
        status='all_public_release_payloads_sha256_verified_via_bindings',
        directory=str(directory),manifest_sha256=summary['manifest_sha256'],
        bindings=bindings,bindings_sha256=digest(bindings),fingerprints=before,
        bytes_hashed=done,files=len(expected),full_payload_hash_verified=True,
        source_shards_copied=False,source_shards_hardlinked=False,source_shards_modified=False,
        host_view_is_upload_directory=False,physical_container_view_verified=False,
        publication_approved=False,time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        materialized_verifier_sha256=FLAT_SHA)


def check_receipt(directory,manifest,summary,bindings,receipt):
    if (receipt.get('format') != 'ds41_bound_release_hash_receipt_v1'
            or receipt.get('status') != 'all_public_release_payloads_sha256_verified_via_bindings'
            or receipt.get('directory') != str(directory)
            or receipt.get('manifest_sha256') != summary['manifest_sha256']
            or receipt.get('bindings') != bindings or receipt.get('bindings_sha256') != digest(bindings)
            or receipt.get('full_payload_hash_verified') is not True
            or receipt.get('host_view_is_upload_directory') is not False
            or receipt.get('materialized_verifier_sha256') != FLAT_SHA
            or receipt.get('verifier_sha256') != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise ValueError('Full bound-release receipt does not cover this exact view/bindings/tool')
    actual = inventory(directory,manifest,bindings)
    if actual != receipt.get('fingerprints'):
        raise ValueError('Verified metadata, mount targets or original shard sources changed')
    if flat.hash_payload(directory/'release-manifest.json',actual['view_files']['release-manifest.json']) != summary['manifest_sha256']:
        raise ValueError('Public manifest changed')
    return dict(status='previous_full_bound_release_fingerprints_unchanged',
        manifest_sha256=summary['manifest_sha256'],bindings_sha256=digest(bindings),
        files=len(actual['view_files']),current_payload_hash_pass=False,
        previous_full_payload_hash_verified=True,host_view_is_upload_directory=False,
        publication_approved=False)


def fresh(path,directory=None):
    if (path.resolve() != path or path.exists() or path.is_symlink() or not path.parent.is_dir()
            or directory is not None and path.is_relative_to(directory)):
        raise ValueError('Use fresh unredirected outputs; private receipts/bindings stay outside the view')


def prepare_view(directory,manifest,summary,sources):
    fresh(directory)
    expected = dict(manifest['files'])
    expected['release-manifest.json'] = dict(sha256=summary['manifest_sha256'])
    if not isinstance(sources,dict) or set(sources) != set(expected):
        raise ValueError('Preparation requires an explicit source for EVERY public file')
    originals = {}
    for name,value in sources.items():
        source = absolute(value)
        if directory.is_relative_to(source.parent) or source.is_relative_to(directory):
            raise ValueError('New view must be separate from all preserved input trees')
        originals[name] = flat.regular(source)
        if 'bytes' in expected[name] and originals[name][3] != expected[name]['bytes']:
            raise ValueError('Source size differs from published file descriptor')
    metadata_bytes = sum(stamp[3] for name,stamp in originals.items() if name not in flat.SHARDS)
    if metadata_bytes > 256*2**20 or shutil.disk_usage(directory.parent).free < 32*2**30+metadata_bytes:
        raise ValueError('Metadata view must stay bounded and preserve32GiB disk reserve')
    directory.mkdir()
    bindings = {}
    for name,row in sorted(expected.items()):
        target = directory/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if name in flat.SHARDS:
            with target.open('xb'):
                pass
            bindings[name] = sources[name]
            continue
        source = absolute(sources[name])
        stamp = originals[name]
        fd = os.open(source,os.O_RDONLY | os.O_NOFOLLOW)
        digest = hashlib.sha256()
        with os.fdopen(fd,'rb',buffering=0) as input_stream,target.open('xb') as output:
            if flat.fingerprint(os.fstat(input_stream.fileno())) != stamp:
                raise ValueError('Metadata source changed before copying')
            while raw := input_stream.read(flat.BUFFER):
                output.write(raw)
                digest.update(raw)
            if flat.fingerprint(os.fstat(input_stream.fileno())) != stamp:
                raise ValueError('Metadata source changed during copying')
        if digest.hexdigest() != row['sha256']:
            raise ValueError('Metadata copied bytes differ from public manifest')
    if any(flat.regular(absolute(sources[name])) != stamp for name,stamp in originals.items()):
        raise ValueError('An original source changed during view preparation')
    inventory(directory,manifest,bindings)
    return bindings,dict(status='public_metadata_and_empty_shard_targets_prepared',
        files=len(expected),bound_shards=len(bindings),metadata_bytes=metadata_bytes,
        host_view_is_upload_directory=False,full_payload_hash_verified=False,
        source_shards_copied=False,source_shards_hardlinked=False,source_shards_modified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--directory',required=True,type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare-sources',type=Path)
    modes.add_argument('--full',action='store_true')
    modes.add_argument('--check-receipt',type=Path)
    parser.add_argument('--bindings',required=True,type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20))
    directory = args.directory.absolute()
    manifest,summary = flat.load_manifest(args.manifest.absolute(),args.manifest_sha256)
    if bool(args.output) != bool(args.prepare_sources or args.full):
        parser.error('Prepare/full requires a fresh private --output; receipt checking is read-only')
    if args.output:
        fresh(args.output.absolute(),directory)
    if args.prepare_sources:
        fresh(args.bindings.absolute(),directory)
        sources,_ = flat.small_json(args.prepare_sources.absolute())
        flat.idle_before_full_hash()
        bindings,result = prepare_view(directory,manifest,summary,sources)
        with args.bindings.absolute().open('xb') as stream:
            stream.write(encoded(bindings))
    else:
        bindings,_ = flat.small_json(args.bindings.absolute())
        if args.full:
            flat.idle_before_full_hash()
            result = verify_full(directory,manifest,summary,bindings,
                progress=lambda row:print(json.dumps(dict(stage='public_payload_hashed',**row)),flush=True))
        else:
            receipt,_ = flat.small_json(args.check_receipt.absolute())
            result = check_receipt(directory,manifest,summary,bindings,receipt)
    if args.output:
        with args.output.absolute().open('xb') as stream:
            stream.write(encoded(result))
    print(json.dumps({k:v for k,v in result.items() if k not in ('bindings','fingerprints')},indent=2))


if __name__ == '__main__':
    main()
