# SPDX-License-Identifier: AGPL-3.0-only
"""Clone a verified runtime and change only its native profiler settings.

Diagnostic only: Python stacks and tensor shapes locate host syncs and GEMM
shapes. Stack collection adds CPU overhead, so such traces are never used for
timing; the unchanged profile settings remain the measurement configuration.
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


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-kit', type=Path, required=True)
    p.add_argument('--parent-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=3, choices=(1, 2, 3))
    a = p.parse_args()
    parent = a.parent_kit.absolute()
    verify(parent, a.parent_sha256)
    kit = a.output.absolute()
    assert not kit.exists() and not a.receipt.exists()
    shutil.copytree(parent, kit)
    path = kit / 'tools/portable_node.py'
    source = path.read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'PROFILE' for t in n.targets))
    old = ast.literal_eval(node.value)['profiler-config']
    settings = json.loads(old)
    assert settings['profiler'] == 'torch' and settings['max_iterations'] == 1
    settings.update(torch_profiler_with_stack=True, torch_profiler_record_shapes=True,
                    max_iterations=a.iterations)
    new = json.dumps(settings)
    assert source.count(repr(old)) == 1
    path.write_text(source.replace(repr(old), repr(new)))
    ast.parse(path.read_text())
    path = kit / 'serving/conservative.yaml'
    text = path.read_text()
    assert text.count('profiler-config: ' + old) == 1
    path.write_text(text.replace('profiler-config: ' + old, 'profiler-config: ' + new))
    overlay = {q.relative_to(kit / 'serving').as_posix(): sha(q.read_bytes())
               for q in sorted((kit / 'serving').rglob('*')) if q.is_file() and q.name != 'overlay-manifest.json'}
    (kit / 'serving/overlay-manifest.json').write_bytes(encoded(overlay))
    manifest = json.loads((parent / 'bundle-manifest.json').read_bytes())
    manifest['parent_manifest_sha256'] = a.parent_sha256
    changes = [n for n in manifest['files'] if (kit / n).read_bytes() != (parent / n).read_bytes()]
    assert set(changes) == {'tools/portable_node.py', 'serving/conservative.yaml', 'serving/overlay-manifest.json'}
    for name in changes:
        raw = (kit / name).read_bytes()
        manifest['files'][name] = dict(bytes=len(raw), sha256=sha(raw))
    (kit / 'bundle-manifest.json').write_bytes(encoded(manifest))
    digest = sha((kit / 'bundle-manifest.json').read_bytes())
    proof = verify(kit, digest)
    a.receipt.write_bytes(encoded(dict(parent_kit=str(parent), parent_sha256=a.parent_sha256,
        candidate_kit=str(kit), candidate_kit_sha256=digest, changed_files=changes,
        profiler_config=settings, diagnostic_only=True, verification=proof)))
    print(json.dumps(dict(kit=str(kit), sha256=digest)), flush=True)


if __name__ == '__main__':
    main()
