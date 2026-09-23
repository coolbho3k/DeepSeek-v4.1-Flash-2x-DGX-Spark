# SPDX-License-Identifier: AGPL-3.0-only
"""Clone a verified runtime and replace only the native vocabulary row store.

The Python stage, callback protocol, ABI, weights, KV and launch profile are
unchanged. Source pins that name the old binary or changed Python files are
refreshed recursively, and a new immutable manifest plus receipt are written.
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment', type=Path)
    p.add_argument('--deployment-sha256')
    p.add_argument('--parent-kit', type=Path, help='Alternative to --deployment, e.g. release/runtime')
    p.add_argument('--parent-sha256')
    p.add_argument('--binary-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    if a.parent_kit:
        raw = b''
        config = dict(kit_manifest_sha256=a.parent_sha256)
        parent = a.parent_kit.absolute()
    else:
        raw = a.deployment.read_bytes()
        if sha(raw) != a.deployment_sha256:
            raise ValueError('Changed baseline deployment')
        config = json.loads(raw)
        parent = Path(config['nodes'][0]['kit'])
    verify(parent, config['kit_manifest_sha256'])
    kit = a.output.absolute()
    if kit.exists() or a.receipt.exists():
        raise ValueError('Fresh output required')
    build = json.loads((a.binary_dir / 'complete.json').read_bytes())
    binary = (a.binary_dir / 'libds41_vocab_rows.so').read_bytes()
    source = (a.binary_dir / 'ds41_vocab_row_store.cpp').read_bytes()
    if (build['status'] != 'vocab_cache_build_and_cpu_parity_pass' or sha(binary) != build['binary_sha256']
            or sha(source) != build['source_sha256']):
        raise ValueError('Unqualified vocabulary build')
    old_binary = sha((parent / 'serving/libds41_vocab_rows.so').read_bytes())
    shutil.copytree(parent, kit)
    (kit / 'serving/libds41_vocab_rows.so').write_bytes(binary)
    (kit / 'native-source/ds41_vocab_row_store.cpp').write_bytes(source)
    stage = kit / 'serving/ds41/native_vocab_stage.py'
    text = stage.read_text()
    if text.count(old_binary) != 1:
        raise ValueError('Changed binary pin anchor')
    stage.write_text(text.replace(old_binary, build['binary_sha256']))
    manifest = json.loads((parent / 'bundle-manifest.json').read_bytes())
    history = {name: {row['sha256'], sha((kit / name).read_bytes())}
               for name, row in manifest['files'].items()
               if name.endswith('.py') and name.startswith(('serving/', 'tools/'))}
    for _ in range(32):
        updates = {old_binary: build['binary_sha256']}
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
    requirements['host_roundtrips'] = dict(vocab_row_cache_slots=4096, vocab_row_cache_bytes=4096 * 10240,
        parallel_miss_reads=True, abi_unchanged=True, cpu_parity=True, full_model_qualified=False)
    (kit / 'runtime-requirements.json').write_bytes(encoded(requirements))
    overlay = {q.relative_to(kit / 'serving').as_posix(): sha(q.read_bytes())
               for q in sorted((kit / 'serving').rglob('*')) if q.is_file() and q.name != 'overlay-manifest.json'}
    (kit / 'serving/overlay-manifest.json').write_bytes(encoded(overlay))
    manifest['parent_manifest_sha256'] = config['kit_manifest_sha256']
    manifest['files'] = {q.relative_to(kit).as_posix(): dict(bytes=q.stat().st_size, sha256=sha(q.read_bytes()))
                         for q in sorted(kit.rglob('*')) if q.is_file() and q.name != 'bundle-manifest.json'}
    (kit / 'bundle-manifest.json').write_bytes(encoded(manifest))
    digest = sha((kit / 'bundle-manifest.json').read_bytes())
    proof = verify(kit, digest)
    changes = {n: dict(before=sha((parent / n).read_bytes()) if (parent / n).is_file() else None,
                       after=sha((kit / n).read_bytes())) for n in manifest['files']
               if not (parent / n).is_file() or (parent / n).read_bytes() != (kit / n).read_bytes()}
    result = dict(parent_deployment=str(a.deployment), parent_deployment_sha256=sha(raw),
                  parent_kit_manifest_sha256=config['kit_manifest_sha256'], candidate_kit=str(kit),
                  candidate_kit_manifest_sha256=digest, changed_files=changes, verification=proof, build=build)
    a.receipt.write_bytes(encoded(result))
    print(json.dumps(dict(kit=str(kit), sha256=digest, changed_files=sorted(changes))))


if __name__ == '__main__':
    main()
