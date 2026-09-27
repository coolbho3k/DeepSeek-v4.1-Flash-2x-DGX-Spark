# SPDX-License-Identifier: AGPL-3.0-only
"""Clone a verified runtime and add confidence-capped DSpark verification.

Adds serving/ds41/verify_cap.py (threshold and row bound baked in), applies the
exact edits in edits.json, registers the module in the startup PINS and backend
attestation, refreshes source pins and writes a new immutable manifest plus
receipt. The parent kit is never modified.
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
        raise ValueError('Changed source anchor in ' + str(path) + ': ' + before[:80])
    path.write_text(text.replace(before, after))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-kit', type=Path, required=True)
    p.add_argument('--parent-sha256', required=True)
    p.add_argument('--threshold', type=float, required=True)
    p.add_argument('--max-rows', type=int, choices=(24, 36), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    parent = a.parent_kit.absolute()
    verify(parent, a.parent_sha256)
    kit = a.output.absolute()
    if kit.exists() or a.receipt.exists():
        raise ValueError('Fresh output required')
    shutil.copytree(parent, kit)
    module = kit / 'serving/ds41/verify_cap.py'
    shutil.copy2(HERE / 'verify_cap.py', module)
    replace(module, 'THRESHOLD = 0.1\n', f'THRESHOLD = {a.threshold!r}\n')
    replace(module, 'MAX_ROWS = 36\n', f'MAX_ROWS = {a.max_rows}\n')
    for name, pairs in json.loads((HERE / 'edits.json').read_bytes()).items():
        for before, after in pairs:
            replace(kit / name, before, after)
    module_sha = sha(module.read_bytes())
    for name, anchor, entries in (
            ('spark_combined_miaai.py', 'PINS = {', {'ds41.verify_cap': module_sha}),
            ('spark_backend_attestation.py', 'PRIVATE_SOURCES = {', {'ds41/verify_cap.py': module_sha})):
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
    requirements['verify_cap'] = dict(threshold=a.threshold, min_drafts=1, max_rows=a.max_rows,
        dead_rows_route_to_anchor=True, exact_truncation=True, full_model_qualified=False)
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
        candidate_kit=str(kit), candidate_kit_sha256=digest, threshold=a.threshold,
        max_rows=a.max_rows, verification=proof)))
    print(json.dumps(dict(kit=str(kit), sha256=digest)))


if __name__ == '__main__':
    main()
