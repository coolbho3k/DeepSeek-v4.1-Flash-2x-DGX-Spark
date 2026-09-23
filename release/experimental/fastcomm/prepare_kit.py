# SPDX-License-Identifier: AGPL-3.0-only
"""Clone a verified runtime and add the fastcomm module, library and hooks.

Adds serving/ds41/fastcomm.py (library hash pinned inside) and
serving/libfastcomm.so, applies exact edits from edits.json, registers both new
files in the startup PINS and backend attestation, refreshes source pins and
writes a new immutable manifest plus receipt.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'release/runtime'))
from verify import verify


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def replace(path, before, after):
    text = path.read_text()
    if text.count(before) != 1:
        raise ValueError('Changed source anchor in ' + str(path))
    path.write_text(text.replace(before, after))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-kit', type=Path, required=True)
    p.add_argument('--parent-sha256', required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    parent = a.parent_kit.absolute()
    verify(parent, a.parent_sha256)
    kit = a.output.absolute()
    if kit.exists() or a.receipt.exists():
        raise ValueError('Fresh output required')
    library = a.library.read_bytes()
    shutil.copytree(parent, kit)
    (kit / 'serving/libfastcomm.so').write_bytes(library)
    module = (HERE / 'fastcomm.py').read_text().replace(
        'LIB_SHA256 = None  # Set by the kit preparer.', f"LIB_SHA256 = '{sha(library)}'", 1)
    assert sha(library) in module
    (kit / 'serving/ds41/fastcomm.py').write_text(module)
    (kit / 'native-source').mkdir(exist_ok=True)
    shutil.copy2(HERE / 'fastcomm.cu', kit / 'native-source/fastcomm.cu')
    for name, pairs in json.loads((HERE / 'edits.json').read_bytes()).items():
        for before, after in pairs:
            replace(kit / name, before, after)
    module_sha = sha((kit / 'serving/ds41/fastcomm.py').read_bytes())
    for name, anchor, entries in (
            ('spark_combined_miaai.py', 'PINS = {', {'ds41.fastcomm': module_sha}),
            ('spark_backend_attestation.py', 'PRIVATE_SOURCES = {', {
                'ds41/fastcomm.py': module_sha, 'libfastcomm.so': sha(library)})):
        replace(kit / 'serving' / name, anchor, anchor + '\n' +
                ''.join(f'    {key!r}: {value!r},\n' for key, value in entries.items()))
    manifest = json.loads((parent / 'bundle-manifest.json').read_bytes())
    history = {name: {row['sha256'], sha((kit / name).read_bytes())}
               for name, row in manifest['files'].items()
               if name.endswith('.py') and name.startswith(('serving/', 'tools/'))}
    for _ in range(32):
        updates = {}
        for name, old in history.items():
            new = sha((kit / name).read_bytes())
            for digest in old:
                if digest != new:
                    if digest in updates and updates[digest] != new:
                        raise RuntimeError('Ambiguous pin')
                    updates[digest] = new
            old.add(new)
        changed = False
        for name in history:
            path = kit / name
            text = before = path.read_text()
            for old, new in updates.items():
                text = text.replace(old, new)
            if text != before:
                path.write_text(text)
                changed = True
        if not changed:
            break
    else:
        raise RuntimeError('Hash cycle')
    for path in (kit / 'serving').rglob('*.py'):
        ast.parse(path.read_text())
    requirements = json.loads((kit / 'runtime-requirements.json').read_bytes())
    requirements['loaded_backend_verification']['sha256'] = sha(
        (kit / 'serving/spark_backend_attestation.py').read_bytes())
    requirements['fastcomm'] = dict(library_sha256=sha(library), limit_bytes=262144,
        channels=['main', 'dcp_side'], bit_exact_vs_nccl=True, full_model_qualified=False)
    (kit / 'runtime-requirements.json').write_bytes(encoded(requirements))
    overlay = {q.relative_to(kit / 'serving').as_posix(): sha(q.read_bytes())
               for q in sorted((kit / 'serving').rglob('*')) if q.is_file() and q.name != 'overlay-manifest.json'}
    (kit / 'serving/overlay-manifest.json').write_bytes(encoded(overlay))
    manifest['parent_manifest_sha256'] = a.parent_sha256
    manifest['files'] = {q.relative_to(kit).as_posix(): dict(bytes=q.stat().st_size, sha256=sha(q.read_bytes()))
                         for q in sorted(kit.rglob('*')) if q.is_file() and q.name != 'bundle-manifest.json'}
    (kit / 'bundle-manifest.json').write_bytes(encoded(manifest))
    digest = sha((kit / 'bundle-manifest.json').read_bytes())
    proof = verify(kit, digest)
    a.receipt.write_bytes(encoded(dict(parent_kit=str(parent), parent_sha256=a.parent_sha256,
        candidate_kit=str(kit), candidate_kit_sha256=digest, verification=proof)))
    print(json.dumps(dict(kit=str(kit), sha256=digest)))


if __name__ == '__main__':
    main()
