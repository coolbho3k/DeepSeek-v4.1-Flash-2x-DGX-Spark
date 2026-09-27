# SPDX-License-Identifier: AGPL-3.0-only
"""Clone a verified runtime kit and install searched NVFP4 scales and opt-in NVFP4 index keys.

Files the parent still carries exactly as git HEAD had them are replaced by this repo's
versions. Files whose parent copy differs from HEAD (source pins, profile defaults) receive
only the reviewed textual edits. New: the NVFP4 index-key module, its native kernel and
receipt, the kernel source and GPU probes. The main-KV writer gains DS41_FP4_KV_MODE=nvfp4_search
(the profile default for descriptors that do not record a mode; nvfp4_4over6 and legacy stay
byte-identical to the parent's). Source pins, the overlay and bundle manifests and the runtime
requirements are recomputed; the parent kit is never modified. A descriptor that records
fp4_kv_mode (as every deployment so far does) and no indexer fields runs exactly the parent's
arithmetic (DS41_INDEXER_K_FORMAT=mxfp4, DS41_INDEXER_DECODE_QUERY=fp8).
  prepare_kit.py --parent-kit KIT --parent-sha256 SHA --output NEW_KIT --receipt RECEIPT.json
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / 'release/runtime'
sys.path.insert(0, str(RUNTIME))
from verify import verify  # noqa: E402

# Parent copies must equal git HEAD (these hashes) before this repo's version replaces them.
REPLACE = {
    'serving/ds41/vllm_dcp.py': '943f807efa71018dd1a23dc159a4c5ac91da98b1a881f843db3dcdb91877fcdd',
    'ds41/vllm_dcp.py': '943f807efa71018dd1a23dc159a4c5ac91da98b1a881f843db3dcdb91877fcdd',
    'serving/spark_indexer_k_math.py': '5ee4f01443af118a6bc50393a967860a30980f7314e38207800a3fbf9840ac57',
    # Hook-count checks that must allow the two NVFP4 format hooks.
    'serving/ds41/vllm_dcp_runtime.py': '73608f392b3b12c9ea7a461d804b13a994178add038852a5d8df9ffac4df91f5',
    'ds41/vllm_dcp_runtime.py': '73608f392b3b12c9ea7a461d804b13a994178add038852a5d8df9ffac4df91f5',
    'serving/ds41/vllm_fp4_main.py': '4d2783a7182755b7577d9a2f8905ecae9dac8e652835ef6184fbc06039954547',
    'ds41/vllm_fp4_main.py': '4d2783a7182755b7577d9a2f8905ecae9dac8e652835ef6184fbc06039954547',
    # Main-KV writer with the E4M3 scale search, and its accuracy and speed probes.
    'serving/ds41/fp4_main_kv.py': 'fc5b8ae2e00e0c2ee854b5b08a7feb747b0ab4dc24ca78a7e5e6690d2f1062bf',
    'ds41/fp4_main_kv.py': 'fc5b8ae2e00e0c2ee854b5b08a7feb747b0ab4dc24ca78a7e5e6690d2f1062bf',
    'serving/ds41/fp4_rope_store.py': '2c2f9303d1adfbdcbc098c3a6bb9bb2d21a9d329fc8957078b34d66cba9697c4',
    'ds41/fp4_rope_store.py': '2c2f9303d1adfbdcbc098c3a6bb9bb2d21a9d329fc8957078b34d66cba9697c4',
    'probes/check_nvfp4_four_over_six.py': 'f47ddd50f1928cb7fdbc80f7976a5ee63fa9fc7505cfddac7960810defcf2b53',
    'probes/bench_nvfp4_kv.py': '0d250ac2f1059d5bba174098a7f7163425cb0a1810e4591321ecea872e17fd77',
}
ADD = ('serving/ds41/nvfp4_indexer.py', 'ds41/nvfp4_indexer.py',
       'serving/nvfp4-indexer-native/nvfp4_indexer.so', 'serving/nvfp4-indexer-native/complete.json',
       'kernels/nvfp4_indexer.cu', 'probes/check_nvfp4_indexer_gpu.py',
       'probes/check_nvfp4_indexer_integration_gpu.py', 'probes/check_nvfp4_prefill_gpu.py',
       'probes/check_nvfp4_decode_gpu.py')
PROFILE_EDITS = [
    ("fp4_kv_mode='nvfp4_4over6', swa_kv_group_size=32)",
     "fp4_kv_mode='nvfp4_search', swa_kv_group_size=32,\n    indexer_k_format='mxfp4', indexer_decode_query='fp8')"),
    ("""    if values['fp4_kv_mode'] not in ('nvfp4_4over6', 'legacy'):
        raise ValueError('FP4 KV mode must be nvfp4_4over6 or legacy')""",
     """    if values['fp4_kv_mode'] not in ('nvfp4_search', 'nvfp4_4over6', 'legacy'):
        raise ValueError('FP4 KV mode must be nvfp4_search, nvfp4_4over6 or legacy')"""),
    ("        for key in ('fp4_kv_mode', 'swa_kv_group_size'):",
     "        for key in ('fp4_kv_mode', 'swa_kv_group_size', 'indexer_k_format', 'indexer_decode_query'):"),
    ("        raise ValueError('SWA KV group size must be 32 or 64; RoPE stays BF16')\n    return values\n",
     "        raise ValueError('SWA KV group size must be 32 or 64; RoPE stays BF16')\n"
     "    if values['indexer_k_format'] not in ('mxfp4', 'nvfp4'):\n"
     "        raise ValueError('Indexer key format must be mxfp4 or nvfp4 (experimental)')\n"
     "    if values['indexer_decode_query'] not in ('fp8', 'nvfp4'):\n"
     "        raise ValueError('Indexer decode query must be fp8 or nvfp4')\n"
     "    if values['indexer_decode_query'] == 'nvfp4' and values['indexer_k_format'] != 'nvfp4':\n"
     "        raise ValueError('NVFP4 decode queries require NVFP4 index keys')\n"
     "    # Defaults stay unrecorded so descriptors predating these fields are unchanged.\n"
     "    for key, default in (('indexer_k_format', 'mxfp4'), ('indexer_decode_query', 'fp8')):\n"
     "        if values[key] == default:\n"
     "            del values[key]\n"
     "    return values\n"),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def edit(path, changes):
    text = path.read_text()
    for before, after in changes:
        if text.count(before) != 1:
            raise ValueError(f'Reviewed anchor not found exactly once in {path.name}: {before[:60]!r}')
        text = text.replace(before, after)
    path.write_text(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-kit', type=Path, required=True)
    p.add_argument('--parent-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    parent = a.parent_kit.absolute()
    verify(parent, a.parent_sha256)
    kit = a.output.absolute()
    if kit.exists() or a.receipt.exists():
        raise ValueError('Fresh output required')
    for name, expected in REPLACE.items():
        if sha((parent / name).read_bytes()) != expected:
            raise ValueError(f'Parent {name} is not the reviewed pre-change source')
    if any((parent / name).exists() for name in ADD):
        raise ValueError('Parent already carries NVFP4 index-key files')
    manifest = json.loads((parent / 'bundle-manifest.json').read_bytes())
    shutil.copytree(parent, kit)
    for name in (*REPLACE, *ADD):
        (kit / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(RUNTIME / name, kit / name)
    new_sha = lambda name: sha((kit / name).read_bytes())
    for name in ('serving/ds41/launch_profile.py', 'tools/launch_profile.py'):
        edit(kit / name, PROFILE_EDITS)
    edit(kit / 'tools/portable_node.py', [(
        "if key not in ('kv_cap_mib', 'fp4_kv_mode', 'swa_kv_group_size'):",
        "if key not in ('kv_cap_mib', 'fp4_kv_mode', 'swa_kv_group_size', 'indexer_k_format', 'indexer_decode_query'):")])
    edit(kit / 'serving/spark_combined_miaai.py', [
        ("        put(arithmetic, 'paged_logits', indexer.paged_logits)\n",
         "        from ds41 import nvfp4_indexer\n"
         "        # Opt-in NVFP4 index keys: capture-safe FP8-query scorer.\n"
         "        put(arithmetic, 'paged_logits',\n"
         "            nvfp4_indexer.graph_paged_logits if nvfp4_indexer.ENABLED else indexer.paged_logits)\n"),
        ("    'ds41.fastcomm': ",
         f"    'ds41.nvfp4_indexer': '{new_sha('serving/ds41/nvfp4_indexer.py')}',\n\n    'ds41.fastcomm': "),
    ])
    edit(kit / 'serving/spark_backend_attestation.py', [(
        "    'ds41/fastcomm.py': ",
        ''.join(f"    '{name[len('serving/'):]}': '{new_sha(name)}',\n" for name in (
            'serving/nvfp4-indexer-native/nvfp4_indexer.so', 'serving/nvfp4-indexer-native/complete.json',
            'serving/ds41/nvfp4_indexer.py')) + "    'ds41/fastcomm.py': ")])
    # Propagate changed source hashes into every Python pin table (as the SWA fix kit does).
    history = {name: {row['sha256'], new_sha(name)} for name, row in manifest['files'].items()
               if name.endswith('.py') and name.startswith(('serving/', 'tools/'))}
    for _ in range(32):
        updates = {}
        for name, old in history.items():
            current = new_sha(name)
            for digest in old:
                if digest != current:
                    if digest in updates and updates[digest] != current:
                        raise RuntimeError('Ambiguous pin')
                    updates[digest] = current
            old.add(current)
        changed = False
        for name in history:
            path = kit / name
            text = before = path.read_text()
            for old_digest, new_digest in updates.items():
                text = text.replace(old_digest, new_digest)
            if text != before:
                path.write_text(text)
                changed = True
        if not changed:
            break
    else:
        raise RuntimeError('Hash cycle')
    for path in (*(kit / 'serving').rglob('*.py'), *(kit / 'tools').rglob('*.py')):
        ast.parse(path.read_text())
    requirements = json.loads((kit / 'runtime-requirements.json').read_bytes())
    requirements['loaded_backend_verification']['sha256'] = new_sha('serving/spark_backend_attestation.py')
    requirements['nvfp4_indexer_keys'] = dict(
        environment=['DS41_INDEXER_K_FORMAT', 'DS41_INDEXER_DECODE_QUERY'], default=['mxfp4', 'fp8'],
        experimental=['nvfp4', 'nvfp4'], record_bytes=72, scale_group=16,
        selection='every E4M3 scale in [amax/6.5, amax/2.5] from E4M3(amax/6); strictly lower group SSE replaces',
        module_sha256=new_sha('serving/ds41/nvfp4_indexer.py'),
        native_sha256=new_sha('serving/nvfp4-indexer-native/nvfp4_indexer.so'),
        gpu_unit_qualified=True, boot_qualified=False, full_model_quality_qualified=False)
    kv = requirements['nvfp4_kv_four_over_six']
    kv.update(codec_sha256=new_sha('serving/ds41/fp4_main_kv.py'), default='nvfp4_search',
              rollback=['nvfp4_4over6', 'legacy'],
              selection='every E4M3 scale in [amax/6.5, amax/2.5] from E4M3(amax/6), strictly lower reconstructed '
                        'group SSE replaces; ties retain legacy /6; never worse than /6 or four-over-six')
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
        candidate_kit=str(kit), candidate_kit_sha256=digest, change='nvfp4-scale-search-and-indexer-keys-opt-in',
        replaced=sorted(REPLACE), added=list(ADD), verification=proof)))
    print(json.dumps(dict(kit=str(kit), sha256=digest)))


if __name__ == '__main__':
    main()
