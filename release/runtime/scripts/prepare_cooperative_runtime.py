# SPDX-License-Identifier: AGPL-3.0-only
"""Freeze an opt-in cooperative candidate derived from runtime53, not a release.

Does not change the baseline runtime, deployment, active jobs or model weights.
DS41_ENABLE_COOPERATIVE_MOE=1 selects the new path at process startup; zero
retains the parent staged dispatch for matched A/B tests. GPU proof pending.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'probes'))
from verify_runtime_bundle import verify

PARENT = ROOT/'artifacts/ds41-runtime-optimize-v53'
PARENT_SHA = 'b2731dd60dbd1ca6ff0b56d20768f6b3adf97b0413bf8a5323ed249572036c81'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Parent rewrite anchor changed: '+old[:120])
    return text.replace(old, new)


def combined_source(parent, additions):
    source = (parent/'serving/spark_combined_miaai.py').read_text()
    pins = ''.join(f"    '{name}': '{sha(raw)}',\n" for name, raw in additions.items())
    source = once(source, 'PINS = {\n', 'PINS = {\n'+pins)
    source = once(source, "vocabulary_enabled = vocabulary_selection == '1'", """vocabulary_enabled = vocabulary_selection == '1'
    cooperative_selection = os.environ.get('DS41_ENABLE_COOPERATIVE_MOE', '0')
    if cooperative_selection not in ('0', '1'):
        raise ValueError('Cooperative MoE selection must be0 or1')
    cooperative_enabled = cooperative_selection == '1'""")
    source = once(source, "if DESCRIPTOR['ssd_input_vocabulary'] is not vocabulary_enabled:",
        "if DESCRIPTOR.get('cooperative_moe', False) is not cooperative_enabled:\n                raise RuntimeError('Cooperative MoE selection changed after startup')\n            if DESCRIPTOR['ssd_input_vocabulary'] is not vocabulary_enabled:")
    old = """compile_shared(async_moe.AsyncSmallDispatcher,'__call__',route_prepare.forward_replacements(),
            {'_ds41_prepare_routes':route_prepare.prepare})"""
    new = """route_rules = route_prepare.forward_replacements()
        route_globals = {'_ds41_prepare_routes':route_prepare.prepare}
        if cooperative_enabled:
            from ds41 import cooperative_moe, cooperative_contract
            route_rules += cooperative_contract.forward_replacements()
            route_globals.update(_ds41_coop_eligible=cooperative_contract.selected_shape,
                _ds41_coop_call=cooperative_moe.configure(Path(__file__).parent))
        compile_shared(async_moe.AsyncSmallDispatcher,'__call__',route_rules,route_globals)"""
    source = once(source, old, new)
    source = once(source, 'dspark_enabled=dspark_enabled, dspark_full_model_qualified=False,',
        """dspark_enabled=dspark_enabled, dspark_full_model_qualified=False,
            cooperative_moe=cooperative_enabled,
            cooperative_upstream_commit='b9c49e90bdcc6f1e0192feb57214df11b67d36aa',
            cooperative_rows=list(range(5,9)) if cooperative_enabled else [],
            cooperative_small_rows_keep_staged=True,
            cooperative_additional_persistent_scratch_bytes=0,
            cooperative_quality_qualified=False, cooperative_performance_measured=False,""")
    return source.encode()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',type=int,choices=range(1,20),default=1)
    args=parser.parse_args()
    target=ROOT/f'artifacts/ds41-runtime-cooperative-v{args.version}'
    verify(PARENT, PARENT_SHA)
    if target.exists():
        raise ValueError('Preserve the existing cooperative candidate')
    parent = json.loads((PARENT/'bundle-manifest.json').read_bytes())
    payload = {name: (PARENT/name).read_bytes() for name in parent['files']}
    additions = {f'ds41.{name}': (ROOT/f'ds41/{name}.py').read_bytes()
        for name in ('cooperative_contract', 'cooperative_routes', 'cooperative_moe')}
    payload['serving/spark_combined_miaai.py'] = combined_source(PARENT, additions)
    for name, raw in additions.items():
        payload['serving/'+name.replace('.', '/')+'.py'] = raw
    build = ROOT/'artifacts/cooperative-moe-build-v1'
    native = json.loads((build/'complete.json').read_bytes())
    binary = (build/'cooperative_moe.so').read_bytes()
    if (native['status'] != 'cooperative_moe_built_cpu_only' or native['gpu_qualified']
            or sha(binary) != native['binary_sha256']):
        raise ValueError('Unexpected native build identity')
    payload['serving/cooperative_moe.so'] = binary
    payload['serving/cooperative-native.json'] = encoded(native)
    # Update, do not disable, runtime source attestation. The attestation's
    # dense/vision inventory is unchanged; cooperative quality remains pending.
    attestation=payload['serving/spark_backend_attestation.py'].decode()
    tree=ast.parse(attestation)
    node=next(n for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='PRIVATE_SOURCES' for t in n.targets))
    pins=ast.literal_eval(node.value)
    changed=['spark_combined_miaai.py','cooperative_moe.so','cooperative-native.json',
        'ds41/cooperative_contract.py','ds41/cooperative_moe.py','ds41/cooperative_routes.py']
    for name in changed:pins[name]=sha(payload['serving/'+name])
    attestation=once(attestation,ast.get_source_segment(attestation,node),
        'PRIVATE_SOURCES = '+repr(pins))
    attestation=once(attestation,
        "limit = 3266944 if path == Path(__file__).parent / 'ds41_moe_mul1_v1.so' else 2 * 2**20",
        "limit = {'ds41_moe_mul1_v1.so': 3266944, 'cooperative_moe.so': "+str(len(binary))+"}.get(path.name, 2 * 2**20)")
    payload['serving/spark_backend_attestation.py']=attestation.encode()
    payload['serving/overlay-manifest.json']=encoded({name.removeprefix('serving/'):sha(raw)
        for name,raw in payload.items() if name.startswith('serving/') and name!='serving/overlay-manifest.json'})
    for directory in ('miaai-cooperative-moe-agpl', 'miaai-cooperative-dependencies-agpl'):
        for path in (ROOT/'vendor'/directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                payload[str(path.relative_to(ROOT))] = path.read_bytes()
    for name in ('scripts/prepare_cooperative_runtime.py', 'scripts/build_cooperative_moe.py',
            'scripts/vendor_miaai_cooperative.py', 'scripts/vendor_cooperative_dependencies.py',
            'probes/check_cooperative_moe_cpu.py', 'probes/check_cooperative_moe_gpu.py',
            'probes/check_combined_miaai_gpu.py', 'probes/check_exl3_prefill_bench.py',
            'docs/cooperative-moe-port.md'):
        payload[name] = (ROOT/name).read_bytes()
    # Do not inherit historical qualification claims for changed/new files.
    manifest = dict(format=parent['format'], standalone_runtime=False,
        clean_rebuild_qualified=False, publication_approved=False,
        variant=f'miaai_cooperative_moe_candidate_v{args.version}', parent_manifest_sha256=PARENT_SHA,
        full_model_launch_admitted=False, gpu_qualified=False,
        selection_env={'DS41_ENABLE_COOPERATIVE_MOE':'1'},
        missing=['Both-rank actual-weight GPU numerical and graph-replay tests',
            'Matched serving quality/acceptance and latency measurements'],
        files={n:dict(bytes=len(raw), sha256=sha(raw)) for n, raw in sorted(payload.items())})
    for name, raw in payload.items():
        if name.endswith('.py'):
            compile(raw, name, 'exec')
    target.mkdir()
    for name, raw in payload.items():
        path = target/name; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
    manifest_raw = encoded(manifest)
    (target/'bundle-manifest.json').write_bytes(manifest_raw)
    integrity = verify(target, sha(manifest_raw))
    result = dict(status='cooperative_candidate_prepared_not_deployed', kit=str(target),
        manifest_sha256=sha(manifest_raw), parent_manifest_sha256=PARENT_SHA,
        changed_parent_files=[n for n in parent['files'] if sha(payload[n]) != parent['files'][n]['sha256']],
        new_files=sorted(set(payload)-set(parent['files'])), gpu_qualified=False,
        full_model_launch_admitted=False, integrity=integrity)
    with (ROOT/f'reports/ds41-runtime-cooperative-v{args.version}.json').open('xb') as stream:
        stream.write(encoded(result))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
