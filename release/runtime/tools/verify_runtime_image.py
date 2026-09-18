"""Check installed image contents/configuration without pulls or containers.

Image IDs can change during Docker transport even when rootfs and execution
configuration match. This portable identity uses the ordered rootfs diff IDs
and a canonical Config hash. It does not export environment values, claim a
public registry exists, or inspect packages by importing them.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

FORMAT = 'ds41_runtime_image_identity_v1'


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate identity key')
        result[key] = value
    return result


def identity(node):
    rootfs = node['RootFS']
    layers = rootfs.get('Layers')
    if (node.get('Os') != 'linux' or node.get('Architecture') != 'arm64'
            or rootfs.get('Type') != 'layers' or not isinstance(layers, list)
            or not 1 <= len(layers) <= 256
            or any(not isinstance(x, str) or not re.fullmatch('sha256:[0-9a-f]{64}', x) for x in layers)
            or not isinstance(node.get('Config'), dict)):
        raise ValueError('Expected an ARM64 Linux layered runtime image')
    return dict(format=FORMAT, os='linux', architecture='arm64',
        variant=node.get('Variant', ''), rootfs_diff_ids=layers,
        execution_config_sha256=digest(node['Config']))


def inspect_command(image, remote=None):
    if not re.fullmatch(r'(?:sha256:|[a-z0-9][a-z0-9._:/-]*@sha256:)[0-9a-f]{64}', image):
        raise ValueError('Use an immutable locally installed image ID or registry digest, not a tag')
    command = ['docker', 'image', 'inspect', image]
    if remote is not None:
        if not re.fullmatch(r'(?:[a-zA-Z0-9_][a-zA-Z0-9_.-]*@)?[a-zA-Z0-9][a-zA-Z0-9.-]*', remote):
            raise ValueError('Use a plain SSH host or user@host')
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', remote, shlex.join(command)]
    return command


def inspect(image, remote=None):
    # Do not impose RLIMIT_AS on Docker's Go client (its virtual reservations
    # exceed its small resident footprint). No image layers are read here.
    raw = subprocess.check_output(inspect_command(image, remote), text=True, timeout=30)
    if len(raw) > 4*2**20:
        raise ValueError('Unexpectedly large image metadata')
    nodes = json.loads(raw)
    if not isinstance(nodes, list) or len(nodes) != 1:
        raise ValueError('Expected one installed image')
    return nodes[0]


def read_identity(path, expected_sha256):
    if (not re.fullmatch('[0-9a-f]{64}', expected_sha256)
            or path.resolve() != path or not path.is_file() or path.stat().st_size > 1024**2):
        raise ValueError('Use a bounded public image identity and independent SHA256')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Public image identity changed')
    value = json.loads(raw, object_pairs_hook=unique)
    if (set(value) != {'format','os','architecture','variant','rootfs_diff_ids','execution_config_sha256'}
            or value['format'] != FORMAT):
        raise ValueError('Unsupported image identity format')
    return value


def verify(node, expected):
    actual = identity(node)
    if actual != expected:
        raise ValueError('Installed runtime layers/configuration differ from the public identity')
    return dict(status='installed_runtime_rootfs_and_config_match', image_id=node['Id'],
        identity_sha256=digest(actual), layers=len(actual['rootfs_diff_ids']),
        execution_config_sha256=actual['execution_config_sha256'],
        image_pulled=False, containers_started=False, gpu_queried=False,
        filesystem_payloads_rehashed=False, clean_rebuild_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--ssh', help='Optional read-only inspection on a remote host')
    parser.add_argument('--identity', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--capture', action='store_true', help='Write a fresh identity; not qualification')
    mode.add_argument('--identity-sha256')
    args = parser.parse_args()
    path = args.identity.absolute()
    if args.capture:
        if path.resolve() != path or path.exists() or path.is_symlink() or not path.parent.is_dir():
            raise ValueError('Use a fresh unredirected identity file')
        node = inspect(args.image, args.ssh)
        actual = identity(node)
        with path.open('xb') as stream:
            stream.write(encoded(actual))
        result = dict(status='runtime_image_identity_captured', image_id=node['Id'],
            identity_sha256=digest(actual), layers=len(actual['rootfs_diff_ids']),
            private_environment_exported=False, clean_rebuild_qualified=False)
    else:
        expected = read_identity(path, args.identity_sha256)
        result = verify(inspect(args.image, args.ssh), expected)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
