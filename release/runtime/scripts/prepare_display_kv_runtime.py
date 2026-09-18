# SPDX-License-Identifier: AGPL-3.0-only
"""Freeze additive display KV on top of the unchanged recipe-v9 rollback."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'probes'))
from verify_runtime_bundle import verify
PARENT=ROOT/'artifacts/ds41-runtime-recipe-v9'
PARENT_SHA='97c9365cecb44425578c7aa7ebf6c367e1b0853398b1a99e6c5fee6351d8503b'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(v):return (json.dumps(v,indent=2,sort_keys=True)+'\n').encode()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version',type=int,default=12)
    p.add_argument('--release',type=int,default=94)
    p.add_argument('--library',type=Path,required=True)
    a=p.parse_args()
    target=ROOT/f'artifacts/ds41-runtime-recipe-v{a.version}'
    deployment=ROOT/f'reports/ds41-release-v{a.release}-deployment.json'
    if target.exists() or deployment.exists():raise ValueError('Fresh immutable candidate required')
    verify(PARENT,PARENT_SHA)
    old=json.loads((PARENT/'bundle-manifest.json').read_bytes())
    payload={n:(PARENT/n).read_bytes() for n in old['files']}
    histories={n:{sha(raw)} for n,raw in payload.items() if n.endswith('.py')}
    def edit(n,before,after):
        source=payload[n].decode()
        if source.count(before)!=1:raise ValueError('Changed anchor: '+n+': '+before[:80])
        payload[n]=source.replace(before,after).encode()
    payload['serving/ds41/display_kv.py']=(ROOT/'ds41/display_kv.py').read_bytes()
    payload['serving/libds41_display_kv.so']=a.library.read_bytes()
    payload['sources/display_kv.c']=(ROOT/'kernels/display_kv.c').read_bytes()
    edit('serving/ds41/combined_config.py',
         '    return [min(value, KV_CAP_BYTES) for value in available_memory]',
         '    from .display_kv import credited_budgets\n    return credited_budgets(available_memory, KV_CAP_BYTES)')
    edit('serving/serve.py',
         'from spark_kv_cap import register\n',
         'from ds41.display_kv import register as register_display_kv\nregister_display_kv()\n\nfrom spark_kv_cap import register\n')
    edit('serving/spark_kv_cap.py',"stage='ds41_downward_kv_pool_cap'","stage='ds41_additive_kv_pool_cap'")
    edit('serving/spark_kv_cap.py','cap_bytes=CAP_BYTES, native_admission_preserved=True',
         'ordinary_cap_bytes=1073741824, external_display_bytes=1879048192, native_ordinary_admission_preserved=True')
    edit('serving/spark_kv_cap.py',"stage='ds41_downward_kv_pool_planned'","stage='ds41_additive_kv_pool_planned'")
    name='tools/portable_node.py'
    edit(name,"    env.update(profile_environment(config['serving']))",
         "    env.update(profile_environment(config['serving']))\n    env['NVIDIA_DRIVER_CAPABILITIES']='compute,utility,graphics,display'")
    edit(name,'    if owner is not None:\n',
         "    # Expose only the DRM card and its device group, never elevate the worker.\n    args += ['--device=/dev/dri/card0','--group-add',str(os.stat('/dev/dri/card0').st_gid)]\n    if owner is not None:\n")
    edit('tools/portable_pair.py',
         "kv_cap_bytes_per_rank=config['serving']['kv_cap_mib']*2**20)",
         "ordinary_kv_cap_bytes_per_rank=config['serving']['kv_cap_mib']*2**20, external_display_bytes_per_rank=1879048192, kv_cap_bytes_per_rank=config['serving']['kv_cap_mib']*2**20+1879048192)")
    # Independent code/library attestations propagated to both worker and inspector.
    name='serving/spark_backend_attestation.py'
    source=payload[name].decode()
    assign=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='PRIVATE_SOURCES' for t in n.targets))
    pins=ast.literal_eval(assign.value)
    for n in ('ds41/display_kv.py','libds41_display_kv.so'):pins[n]=sha(payload['serving/'+n])
    edit(name,ast.get_source_segment(source,assign),'PRIVATE_SOURCES = '+repr(pins))
    for _ in range(32):
        for n,values in histories.items():values.add(sha(payload[n]))
        updates={before:sha(payload[n]) for n,values in histories.items()
                 for before in values if before!=sha(payload[n])}
        changed=False
        for n,raw in list(payload.items()):
            if not n.endswith('.py') or not n.startswith(('serving/','tools/')):continue
            for before,after in updates.items():raw=raw.replace(before.encode(),after.encode())
            if raw!=payload[n]:payload[n]=raw;changed=True
        if not changed:break
    else:raise RuntimeError('Source pin propagation failed')
    requirements=json.loads(payload['runtime-requirements.json'])
    requirements['loaded_backend_verification']['sha256']=sha(payload['serving/spark_backend_attestation.py'])
    payload['runtime-requirements.json']=encoded(requirements)
    payload['serving/overlay-manifest.json']=encoded({n.removeprefix('serving/'):sha(raw)
        for n,raw in payload.items() if n.startswith('serving/') and n!='serving/overlay-manifest.json'})
    payload['scripts/prepare_display_kv_runtime.py']=Path(__file__).read_bytes()
    for n,raw in payload.items():
        if n.endswith('.py'):compile(raw,n,'exec')
    manifest=dict(format=old['format'],standalone_runtime=False,clean_rebuild_qualified=False,
        publication_approved=False,variant='additive_display_kv_1gib_plus_1p75gib',
        parent_manifest_sha256=PARENT_SHA,serving_qualified=False,
        files={n:dict(bytes=len(raw),sha256=sha(raw)) for n,raw in sorted(payload.items())})
    for n,raw in payload.items():
        path=target/n;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    raw=encoded(manifest);(target/'bundle-manifest.json').write_bytes(raw);verify(target,sha(raw))
    config=json.loads((ROOT/'.state/ds41-release-v1789611312298858252.json').read_bytes())
    assert config['serving']['kv_cap_mib']==1024 and config['serving']['gpu_memory_utilization']==.92
    config.update(run_id=f'ds41-release-v{a.release}',kit_manifest_sha256=sha(raw))
    for node in config['nodes']:node['kit']=str(target)
    deployment.write_bytes(encoded(config))
    print(json.dumps(dict(kit=str(target),deployment=str(deployment),manifest_sha256=sha(raw))))

if __name__=='__main__':main()
