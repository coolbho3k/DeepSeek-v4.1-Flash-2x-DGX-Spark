# SPDX-License-Identifier: AGPL-3.0-only
"""Clone a verified runtime and change only native draft/verification selection."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'release/runtime'))
from verify import verify


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment', type=Path, required=True)
    p.add_argument('--deployment-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--rejection-sample-method', choices=('block',),
        help='Keep probabilistic drafting and switch standard verification to block verification')
    a = p.parse_args()
    raw = a.deployment.read_bytes()
    assert sha(raw) == a.deployment_sha256
    config = json.loads(raw)
    parent = Path(config['nodes'][0]['kit'])
    verify(parent, config['kit_manifest_sha256'])
    kit = a.output.absolute()
    assert kit.resolve() == kit and not kit.exists() and not a.receipt.exists()
    shutil.copytree(parent, kit)
    path = kit / 'tools/portable_node.py'
    source = path.read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'PROFILE' for t in n.targets))
    profile = ast.literal_eval(node.value)
    old = profile['speculative-config']
    spec = json.loads(old)
    if a.rejection_sample_method:
        assert spec.get('draft_sample_method') == 'probabilistic'
        assert spec.get('rejection_sample_method', 'standard') == 'standard'
        spec['rejection_sample_method'] = a.rejection_sample_method
    else:
        assert spec.get('draft_sample_method', 'greedy') == 'greedy'
        spec['draft_sample_method'] = 'probabilistic'
    new = json.dumps(spec)
    assert source.count(repr(old)) == 1
    path.write_text(source.replace(repr(old), repr(new)))
    ast.parse(path.read_text())
    path = kit / 'serving/conservative.yaml'
    source = path.read_text()
    assert source.count('speculative-config: ' + old) == 1
    path.write_text(source.replace('speculative-config: ' + old, 'speculative-config: ' + new))
    overlay = {p.relative_to(kit / 'serving').as_posix(): sha(p.read_bytes())
        for p in sorted((kit / 'serving').rglob('*')) if p.is_file() and p.name != 'overlay-manifest.json'}
    (kit / 'serving/overlay-manifest.json').write_bytes(encoded(overlay))
    manifest = json.loads((parent / 'bundle-manifest.json').read_bytes())
    manifest['parent_manifest_sha256'] = config['kit_manifest_sha256']
    changes = [n for n in manifest['files'] if (kit / n).read_bytes() != (parent / n).read_bytes()]
    assert set(changes) == {'tools/portable_node.py', 'serving/conservative.yaml', 'serving/overlay-manifest.json'}
    for name in changes:
        raw_file = (kit / name).read_bytes()
        manifest['files'][name] = dict(bytes=len(raw_file), sha256=sha(raw_file))
    (kit / 'bundle-manifest.json').write_bytes(encoded(manifest))
    digest = sha((kit / 'bundle-manifest.json').read_bytes())
    proof = verify(kit, digest)
    result = dict(baseline_deployment=str(a.deployment), baseline_deployment_sha256=sha(raw),
        candidate_kit=str(kit), candidate_kit_sha256=digest, changed_files=changes,
        speculative_config=spec, verification=proof)
    a.receipt.write_bytes(encoded(result))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
