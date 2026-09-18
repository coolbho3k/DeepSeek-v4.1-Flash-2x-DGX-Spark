# SPDX-License-Identifier: AGPL-3.0-only
"""Additive draft-EXL3 + latest MiaAI candidate; preserve all frozen rollback kits."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'probes'))
from verify_runtime_bundle import verify

PARENT = ROOT/'artifacts/ds41-runtime-cooperative-v4'
PARENT_SHA = '5fdce9271eb7e8fb39c01011376a806fc5f39ebebe61fa2830a39cfe1cc1f483'
LATEST = '8404ac7d389c418300d0bee960d52313247930e1'


def sha(raw): return hashlib.sha256(raw).hexdigest()
def encoded(value): return (json.dumps(value, indent=2, sort_keys=True)+'\n').encode()
def once(source, old, new):
    if source.count(old) != 1: raise ValueError('Changed rewrite anchor: '+old[:100])
    return source.replace(old, new)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version', type=int, required=True)
    p.add_argument('--release', type=int, required=True)
    p.add_argument('--kv-mib', type=int, default=1536)
    p.add_argument('--utilization',type=float,choices=(.92,.922,.925),default=.925)
    a = p.parse_args()
    if not 704 <= a.kv_mib <= 1702: raise ValueError('Stay within the tensor-only saving')
    target = ROOT/f'artifacts/ds41-runtime-draft-combined-v{a.version}'
    deployment = ROOT/f'reports/ds41-release-v{a.release}-deployment.json'
    if target.exists() or deployment.exists(): raise ValueError('Fresh candidate and deployment required')
    verify(PARENT, PARENT_SHA)
    old = json.loads((PARENT/'bundle-manifest.json').read_bytes())
    payload = {n:(PARENT/n).read_bytes() for n in old['files']}
    def edit(name, old_text, new_text):
        payload[name] = once(payload[name].decode(), old_text, new_text).encode()
    for name in ('draft_exl3_contract', 'draft_exl3_serving'):
        payload[f'serving/ds41/{name}.py'] = (ROOT/f'ds41/{name}.py').read_bytes()
    path = 'serving/ds41/combined_dspark.py'
    source = payload[path].decode()
    start = source.index('def make_weight_loader(original, stream):')
    end = source.index('\ndef make_image_scheduler', start)
    source = source[:start]+'''def make_weight_loader(original, stream):
    from .draft_exl3_serving import make_weight_loader as packed_loader
    return packed_loader(original, stream, _draft_scope)

'''+source[end:]
    source = once(source, '    return [\n', '    from .draft_exl3_serving import make_quantizer_patches\n    return [\n        *make_quantizer_patches(_draft_scope),\n')
    payload[path] = source.encode()
    # Three additional immutable banks, not a weight-copy or workspace expansion.
    for path in ('serving/spark_fused_moe.py','serving/spark_fused_moe_async.py'):
        edit(path, 'len(self.banks) >= 40', 'len(self.banks) >= 43')
        edit(path, 'More than 40 immutable DS41 expert banks', 'More than 43 immutable target-plus-draft expert banks')
    edit('serving/ds41/combined_config.py', 'KV_CAP_BYTES = 738197504', f'KV_CAP_BYTES = {a.kv_mib*2**20}')
    edit('serving/spark_kv_cap.py', 'CAP_BYTES = 1_009_612_800', f'CAP_BYTES = {a.kv_mib*2**20}')
    if a.utilization != .925:
        edit('serving/ds41/combined_config.py','INITIAL_UTILIZATION = 0.925',f'INITIAL_UTILIZATION = {a.utilization}')
        edit('serving/spark_combined_ready.py','INITIAL_UTILIZATION = 0.925',f'INITIAL_UTILIZATION = {a.utilization}')
        edit('serving/conservative.yaml','gpu-memory-utilization: 0.925',f'gpu-memory-utilization: {a.utilization}')
        edit('tools/portable_node.py',"'gpu-memory-utilization': '0.925'",f"'gpu-memory-utilization': '{a.utilization}'")
    # Startup-only UMA accounting: retain the SAME768MiB continuous stop
    # reserve and .925 allocator ceiling; admit6GiB of reclaimable file cache.
    source=payload['serving/combined_worker.py'].decode()
    if source.count("'required_available=required+GIB'")!=1 or source.count("'required_initial_host_available_bytes=required+GIB'")!=1:
        raise ValueError('Startup reserve anchors changed')
    source=source.replace("'required_available=required+GIB'", "'required_available=required+768*2**20'")
    source=source.replace("'required_initial_host_available_bytes=required+GIB'", "'required_initial_host_available_bytes=required+768*2**20'")
    source=source.replace("'STARTUP_CACHE_ALLOWANCE': 5*2**30", "'STARTUP_CACHE_ALLOWANCE': 6*2**30")
    source=source.replace('explicit high-utilization 1GiB startup reserve', 'explicit high-utilization768MiB startup reserve')
    source=once(source, '    def stamp(s):', "    names += [f'/draft-exl3/draft-{i:05d}-of-00003.safetensors' for i in (1,2,3)]\n    def stamp(s):")
    payload['serving/combined_worker.py']=source.encode()
    edit('tools/portable_pair.py', 'kv_cap_bytes_per_rank=738197504', f'kv_cap_bytes_per_rank={a.kv_mib*2**20}')
    edit('tools/portable_node.py', "env = dict(CUDA_VISIBLE_DEVICES='0',", "env = dict(DS41_ENABLE_COOPERATIVE_MOE='1', DS41_DRAFT_EXL3_PATH='/draft-exl3', CUDA_VISIBLE_DEVICES='0',")
    edit('tools/portable_node.py', "mounts = [(node['model'],'/model',True),", "mounts = [('/home/emi/code/ds41/artifacts/ds41-draft-exl3-3bpw-sparse-v3','/draft-exl3',True), (node['model'],'/model',True),")
    edit('serving/spark_combined_miaai.py', "    license='AGPL-3.0-only', upstream_commit='979e68a62c90b24d928f5638596e0ceed90e9f34',", f"    license='AGPL-3.0-only', upstream_commit='{LATEST}', draft_experts='exl3_3bit_mul1',")
    # Require selected cooperative mode consistently in parent/spawn/test processes.
    edit('serving/serve.py', 'from spark_combined_miaai import register as register_combined', "if os.environ.get('DS41_ENABLE_COOPERATIVE_MOE', '1') != '1':\n    raise ValueError('Combined draft candidate requires cooperative MoE')\nos.environ['DS41_ENABLE_COOPERATIVE_MOE']='1'\n\nfrom spark_combined_miaai import register as register_combined")
    path = 'serving/spark_combined_ready.py'
    edit(path, '    validate_groups(runner.kv_cache_config, dspark=enabled)', '    from ds41.draft_exl3_serving import inventory as draft_inventory\n    packed_draft = draft_inventory(draft) if enabled else None\n    validate_groups(runner.kv_cache_config, dspark=enabled)')
    edit(path, "draft_expert_dtype=getattr(draft.config,'expert_dtype','fp4') if enabled else None,", "draft_expert_dtype='exl3_3bit_mul1' if enabled else None, draft_exl3=packed_draft,")
    edit(path, "result.get('draft_expert_dtype')!='fp4'", "result.get('draft_expert_dtype')!='exl3_3bit_mul1'\n                or result.get('draft_exl3',{}).get('actual_parameter_bytes')!=2562494976")
    # Add new sources to the existing attestation, retaining all native pins.
    path = 'serving/spark_backend_attestation.py'
    source = payload[path].decode(); tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id=='PRIVATE_SOURCES' for t in n.targets))
    pins = ast.literal_eval(node.value)
    for name in ('draft_exl3_contract','draft_exl3_serving'):
        pins[f'ds41/{name}.py'] = sha(payload[f'serving/ds41/{name}.py'])
    payload[path] = once(source, ast.get_source_segment(source,node), 'PRIVATE_SOURCES = '+repr(pins)).encode()
    # Propagate changed source identities through the existing acyclic pin graph.
    histories={n:{info['sha256']} for n,info in old['files'].items() if n.endswith('.py')}
    for _ in range(20):
        for n,history in histories.items():history.add(sha(payload[n]))
        replacements = {previous:sha(payload[n]) for n,history in histories.items()
            for previous in history if previous != sha(payload[n])}
        changed = False
        for name, raw in list(payload.items()):
            if not name.endswith('.py') or not name.startswith(('serving/','tools/')): continue
            for before, after in replacements.items(): raw = raw.replace(before.encode(), after.encode())
            if raw != payload[name]: payload[name] = raw; changed = True
        if not changed: break
    else: raise RuntimeError('Source pin graph failed to settle')
    # Rebuild all PRIVATE_SOURCES directly from final payload identities.
    source = payload['serving/spark_backend_attestation.py'].decode(); tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PRIVATE_SOURCES' for t in n.targets))
    pins=ast.literal_eval(node.value)
    for name in pins:
        if 'serving/'+name in payload: pins[name]=sha(payload['serving/'+name])
    payload['serving/spark_backend_attestation.py']=once(source,ast.get_source_segment(source,node),'PRIVATE_SOURCES = '+repr(pins)).encode()
    # The controller verifies this policy before executing the inspector. Keep
    # its JSON pin synchronized too, not just references embedded in Python.
    requirements=json.loads(payload['runtime-requirements.json'])
    requirements['loaded_backend_verification']=dict(
        format='ds41_loaded_combined_miaai_v9',
        inspector='serving/spark_backend_attestation.py',
        sha256=sha(payload['serving/spark_backend_attestation.py']))
    payload['runtime-requirements.json']=encoded(requirements)
    payload['serving/overlay-manifest.json']=encoded({n.removeprefix('serving/'):sha(raw)
        for n,raw in payload.items() if n.startswith('serving/') and n!='serving/overlay-manifest.json'})
    for name in ('scripts/prepare_draft_combined_runtime.py','probes/check_draft_exl3_serving_gpu.py'):
        payload[name]=(ROOT/name).read_bytes()
    for path in (ROOT/'vendor/miaai-latest-20260916-agpl').rglob('*'):
        if path.is_file(): payload[str(path.relative_to(ROOT))]=path.read_bytes()
    manifest=dict(format=old['format'],standalone_runtime=False,clean_rebuild_qualified=False,publication_approved=False,
        variant='draft_exl3_plus_latest_miaai',parent_manifest_sha256=PARENT_SHA,
        upstream_commit=LATEST,serving_qualified=False,kv_cap_bytes_per_rank=a.kv_mib*2**20,
        files={n:dict(bytes=len(raw),sha256=sha(raw)) for n,raw in sorted(payload.items())})
    for name,raw in payload.items():
        if name.endswith('.py'): compile(raw,name,'exec')
    target.mkdir()
    for name,raw in payload.items():
        path=target/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(raw)
    raw=encoded(manifest);(target/'bundle-manifest.json').write_bytes(raw)
    verify(target,sha(raw))
    config=json.loads((ROOT/'reports/ds41-release-v77-deployment.json').read_bytes())
    config.update(run_id=f'ds41-release-v{a.release}',kit_manifest_sha256=sha(raw))
    for node in config['nodes']:node['kit']=str(target)
    deployment.write_bytes(encoded(config))
    receipt=dict(kit=str(target),manifest_sha256=sha(raw),deployment=str(deployment),
        deployment_sha256=sha(encoded(config)),kv_cap_bytes_per_rank=a.kv_mib*2**20,
        gpu_utilization=a.utilization,upstream_commit=LATEST,serving_qualified=False)
    (ROOT/f'reports/draft-combined-runtime-v{a.version}.json').write_bytes(encoded(receipt))
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
