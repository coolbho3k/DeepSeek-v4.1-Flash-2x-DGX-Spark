# SPDX-License-Identifier: AGPL-3.0-only
"""Clone a verified runtime and apply exact, unique source edits from a JSON file.

The edits file is {"relative/path.py": [[before, after], ...], ...}. Every
anchor must occur exactly once. Source pins are refreshed recursively and a
new immutable manifest plus receipt are written, as for the other preparers.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'release/runtime'))
from verify import verify

def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def edit(path, edits, append=''):
    text = path.read_text()
    for before, after in edits:
        if text.count(before) != 1:
            raise ValueError(f'Changed source anchor in {path}: {before[:60]!r}')
        text = text.replace(before, after)
    path.write_text(text + append)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-kit', type=Path, required=True)
    p.add_argument('--parent-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--edits', type=Path, required=True)
    p.add_argument('--label', required=True)
    a = p.parse_args()
    edits = json.loads(a.edits.read_bytes())
    parent = a.parent_kit.absolute()
    verify(parent, a.parent_sha256)
    kit = a.output.absolute()
    if kit.exists() or a.receipt.exists():
        raise ValueError('Fresh output required')
    shutil.copytree(parent, kit)
    for name, pairs in edits.items():
        edit(kit / name, [tuple(x) for x in pairs])
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
    requirements.setdefault('source_edits', {})[a.label] = dict(
        edits_sha256=sha(a.edits.read_bytes()), files=sorted(edits))
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
    changes = {n: dict(before=sha((parent / n).read_bytes()) if (parent / n).is_file() else None,
                       after=sha((kit / n).read_bytes())) for n in manifest['files']
               if not (parent / n).is_file() or (parent / n).read_bytes() != (kit / n).read_bytes()}
    a.receipt.write_bytes(encoded(dict(parent_kit=str(parent), parent_sha256=a.parent_sha256,
        candidate_kit=str(kit), candidate_kit_sha256=digest, changed_files=changes, verification=proof)))
    print(json.dumps(dict(kit=str(kit), sha256=digest, changed_files=sorted(changes))))


if __name__ == '__main__':
    main()
