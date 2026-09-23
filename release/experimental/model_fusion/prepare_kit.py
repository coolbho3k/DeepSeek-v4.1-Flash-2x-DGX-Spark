# SPDX-License-Identifier: AGPL-3.0-only
"""Clone a pinned runtime and overlay only the two exact fusion candidates."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', type=Path, required=True)
    parser.add_argument('--deployment-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary-dir', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    raw = args.deployment.read_bytes()
    if sha(raw) != args.deployment_sha256:
        raise ValueError('Changed baseline deployment')
    config = json.loads(raw)
    parent = Path(config['nodes'][0]['kit'])
    verify(parent, config['kit_manifest_sha256'])
    kit = args.output.absolute()
    if kit.resolve() != kit or kit.exists() or args.receipt.exists():
        raise ValueError('Fresh, unredirected output required')
    receipt = json.loads((args.binary_dir / 'complete.json').read_bytes())
    binary = (args.binary_dir / 'dual_gather.so').read_bytes()
    if sha(binary) != receipt['binary_sha256'] or receipt['binary_sha256'] != 'e8194b01e87e068d4349b1ee7821d6bad1b295ba39281cadaeb42bdbe7b5c67f':
        raise ValueError('Changed qualified gather build')
    for name, digest in receipt['source_files'].items():
        if sha((args.source_dir / name).read_bytes()) != digest:
            raise ValueError('Changed source: ' + name)
    shutil.copytree(parent, kit)
    source = Path(__file__).resolve().parent
    for name in ('packed_wo_a_rows.py', 'dual_gather.py'):
        shutil.copy2(source / name, kit / 'serving/ds41' / name)
    (kit / 'serving/libds41_dual_gather.so').write_bytes(binary)
    shutil.copytree(args.source_dir, kit / 'vendor/model-fusion-dual-gather')
    shutil.copy2(args.binary_dir / 'complete.json', kit / 'vendor/model-fusion-dual-gather/build.json')
    # Dispatch only verification rows. Preserve original single-row and larger
    # GEMM/reconstruct paths and the existing graph-ownership check.
    replace(kit / 'serving/spark_packed_wo_a.py', '    m, g, k = x.shape',
            "    if algorithm == 'auto' and 2 <= len(x) <= 4:\n"
            '        from ds41.packed_wo_a_rows import forward\n'
            '        return forward(x, weight, scale, tile_n=16, warps=4)\n'
            '    m, g, k = x.shape')
    replace(kit / 'serving/spark_grouped_prefill.py',
            '        self.fat_module=fat_module',
            '        self.fat_module=fat_module\n'
            '        from ds41.dual_gather import load\n'
            '        load()')
    replace(kit / 'serving/spark_grouped_prefill.py',
            '                m.gather(inputs,fat.tokens,fat.experts,p[1],fat.h13g,fat_rows)\n'
            '                m.gather(inputs,fat.tokens,fat.experts,p[4],fat.h13u,fat_rows)',
            '                from ds41.dual_gather import forward as dual_gather\n'
            '                dual_gather(inputs,fat.tokens,fat.experts,p[1],p[4],\n'
            '                    fat.h13g,fat.h13u,fat_rows,ids.numel(),stream)')
    # Include new sources in the same startup and post-load integrity checks.
    for name, anchor, entries in (
        ('spark_combined_miaai.py', 'PINS = {', {
            'ds41.packed_wo_a_rows': sha((kit / 'serving/ds41/packed_wo_a_rows.py').read_bytes()),
            'ds41.dual_gather': sha((kit / 'serving/ds41/dual_gather.py').read_bytes())}),
        ('spark_backend_attestation.py', 'PRIVATE_SOURCES = {', {
            'ds41/packed_wo_a_rows.py': sha((kit / 'serving/ds41/packed_wo_a_rows.py').read_bytes()),
            'ds41/dual_gather.py': sha((kit / 'serving/ds41/dual_gather.py').read_bytes()),
            'libds41_dual_gather.so': sha(binary)})):
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
            text = path.read_text(); before = text
            for old, new in updates.items():
                text = text.replace(old, new)
            if text != before:
                path.write_text(text); changed = True
        if not changed:
            break
    else:
        raise RuntimeError('Hash cycle')
    for path in (kit / 'serving').rglob('*.py'):
        ast.parse(path.read_text())
    requirements = json.loads((kit / 'runtime-requirements.json').read_bytes())
    requirements['loaded_backend_verification']['sha256'] = sha((kit / 'serving/spark_backend_attestation.py').read_bytes())
    requirements['model_fusion'] = dict(shared_wo_a_rows=[2,3,4], tile_n=16, warps=4,
        dual_prefill_gather=True, additional_persistent_gpu_bytes=0,
        component_exact=True, full_model_qualified=False)
    (kit / 'runtime-requirements.json').write_bytes(encoded(requirements))
    overlay = {p.relative_to(kit / 'serving').as_posix(): sha(p.read_bytes())
               for p in sorted((kit / 'serving').rglob('*')) if p.is_file() and p.name != 'overlay-manifest.json'}
    (kit / 'serving/overlay-manifest.json').write_bytes(encoded(overlay))
    manifest['parent_manifest_sha256'] = config['kit_manifest_sha256']
    manifest['files'] = {p.relative_to(kit).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p.read_bytes()))
                         for p in sorted(kit.rglob('*')) if p.is_file() and p.name != 'bundle-manifest.json'}
    (kit / 'bundle-manifest.json').write_bytes(encoded(manifest))
    digest = sha((kit / 'bundle-manifest.json').read_bytes())
    proof = verify(kit, digest)
    changes = {name: dict(before=sha((parent / name).read_bytes()) if (parent / name).is_file() else None,
                         after=sha((kit / name).read_bytes())) for name in manifest['files']
               if not (parent / name).is_file() or (parent / name).read_bytes() != (kit / name).read_bytes()}
    result = dict(parent_deployment=str(args.deployment), parent_deployment_sha256=sha(raw),
                  parent_kit_manifest_sha256=config['kit_manifest_sha256'], candidate_kit=str(kit),
                  candidate_kit_manifest_sha256=digest, changed_files=changes, verification=proof)
    args.receipt.write_bytes(encoded(result))
    print(json.dumps(dict(kit=str(kit), sha256=digest, changed_files=len(changes))))


if __name__ == '__main__':
    main()
