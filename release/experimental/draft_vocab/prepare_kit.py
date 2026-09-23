# SPDX-License-Identifier: AGPL-3.0-only
"""Clone a verified runtime and add only the frequency-ranked draft vocabulary.

The subset file and module are pinned by hash; the native DSpark loading
wrapper installs the subset hooks on the loaded drafter instance after its
existing shared-table validation. Target weights, logits and verification,
KV, memory limits and launch profile are unchanged.
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


def replace(path, before, after):
    text = path.read_text()
    if text.count(before) != 1:
        raise ValueError('Changed source anchor in ' + str(path))
    path.write_text(text.replace(before, after))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-kit', type=Path, required=True)
    p.add_argument('--parent-sha256', required=True)
    p.add_argument('--subset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    parent = a.parent_kit.absolute()
    verify(parent, a.parent_sha256)
    kit = a.output.absolute()
    if kit.exists() or a.receipt.exists():
        raise ValueError('Fresh output required')
    subset_raw = a.subset.read_bytes()
    subset = json.loads(subset_raw)
    shutil.copytree(parent, kit)
    source = Path(__file__).resolve().parent / 'draft_vocab.py'
    module = source.read_text().replace('VOCAB = 129280\n',
        f"VOCAB = 129280\nSUBSET_SHA256 = '{sha(subset_raw)}'\n", 1)
    assert 'SUBSET_SHA256' in module
    (kit / 'serving/ds41/draft_vocab.py').write_text(module)
    (kit / 'serving/ds41/draft_vocab_subset.json').write_bytes(subset_raw)
    replace(kit / 'serving/ds41/combined_dspark.py',
            "                raise RuntimeError('Native DSpark failed to share the exact target vocabulary tables')\n"
            "            return result\n",
            "                raise RuntimeError('Native DSpark failed to share the exact target vocabulary tables')\n"
            "            from .draft_vocab import install, load_subset, SUBSET_SHA256\n"
            "            return install(result, load_subset(\n"
            "                Path(__file__).with_name('draft_vocab_subset.json'), SUBSET_SHA256))\n")
    module_sha = sha((kit / 'serving/ds41/draft_vocab.py').read_bytes())
    for name, anchor, entries in (
            ('spark_combined_miaai.py', 'PINS = {', {'ds41.draft_vocab': module_sha}),
            ('spark_backend_attestation.py', 'PRIVATE_SOURCES = {', {
                'ds41/draft_vocab.py': module_sha,
                'ds41/draft_vocab_subset.json': sha(subset_raw)})):
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
    requirements['draft_vocab'] = dict(per_rank=subset['per_rank'], total=2 * subset['per_rank'],
        subset_sha256=sha(subset_raw), in_sample_coverage=subset['in_sample_coverage'],
        target_distribution_unchanged=True, full_model_qualified=False)
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
