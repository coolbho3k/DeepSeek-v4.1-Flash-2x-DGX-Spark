# SPDX-License-Identifier: AGPL-3.0-only
"""Freeze independently selectable, GPU-tested kernels into one serving candidate."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'probes'))
from verify_runtime_bundle import verify
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,indent=2,sort_keys=True)+'\n').encode()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version',type=int,required=True)
    p.add_argument('--test-version',type=int,required=True)
    p.add_argument('--without-attention',action='store_true')
    p.add_argument('--without-topk',action='store_true')
    a=p.parse_args()
    selected=dict(online_decode_attention=not a.without_attention,length_aware_radix_topk=not a.without_topk)
    if not any(selected.values()):raise ValueError('Use original profile for a no-change launch')
    config=json.loads((ROOT/'reports/ds41-release-v105-deployment.json').read_bytes())
    parent=Path(config['nodes'][0]['kit']);parent_sha=config['kit_manifest_sha256']
    target=ROOT/f'artifacts/ds41-runtime-kernel-batch-v{a.version}'
    profile=ROOT/f'reports/kernel-batch-profile-v{a.version}.json'
    if target.exists() or profile.exists():raise ValueError('Fresh immutable candidate required')
    verify(parent,parent_sha)
    native=json.loads((ROOT/'artifacts/topk-build-v1/complete.json').read_bytes())
    components=['online_decode_attention','length_aware_topk','length_aware_topk_native']
    current={name:sha((ROOT/f'ds41/{name}.py').read_bytes()) for name in components}
    for host in (0,1):
        result=json.loads((ROOT/f'reports/kernel-batch-gpu-v{a.test_version}/host{host}/complete.json').read_bytes())
        if result['status']!='complete' or not all(result['guards'].values()):
            raise ValueError('Complete component and graph-error proof required on both GPUs')
        for label,name,enabled in (('attention','online_decode_attention',selected['online_decode_attention']),
                                   ('topk','length_aware_topk_native',selected['length_aware_radix_topk'])):
            if enabled and (result['candidates'][label]['status']!='pass' or result['source_sha256'][name]!=current[name]):
                raise ValueError('Changed or unqualified selected kernel')
        earlier=json.loads((ROOT/f'reports/kernel-batch-gpu-v3/host{host}/complete.json').read_bytes())
        if earlier['source_sha256']['length_aware_topk']!=current['length_aware_topk']:
            raise ValueError('Changed exact ordering/validation dependency')
    binary=(ROOT/'artifacts/topk-build-v1/topk.so').read_bytes()
    if sha(binary)!=native['binary_sha256'] or sha((ROOT/'kernels/length_aware_topk.cu').read_bytes())!=native['source_sha256']:
        raise ValueError('Native source/binary changed')
    previous=json.loads((parent/'bundle-manifest.json').read_bytes())
    payload={name:(parent/name).read_bytes() for name in previous['files']}
    history={name:{sha(raw)} for name,raw in payload.items() if name.endswith('.py')}
    def edit(name,old,new):
        source=payload[name].decode()
        if source.count(old)!=1:raise ValueError(('Changed anchor',name,old[:100]))
        payload[name]=source.replace(old,new).encode()
    for name in components:payload[f'serving/ds41/{name}.py']=(ROOT/f'ds41/{name}.py').read_bytes()
    payload['serving/topk-native/topk.so']=binary
    payload['serving/topk-native/complete.json']=encoded(native)
    name='serving/spark_combined_miaai.py'
    edit(name,"DESCRIPTOR = dict(implementation=",'KERNEL_BATCH = '+repr(selected)+'\nDESCRIPTOR = dict(kernel_batch=KERNEL_BATCH, implementation=')
    edit(name,'        from ds41 import combined_dspark as dspark\n',
        '        from ds41 import combined_dspark as dspark\n'
        '        from ds41 import online_decode_attention as batch_attention\n'
        '        from ds41 import length_aware_topk_native as batch_topk\n')
    edit(name,"        put(modules['spark_topk'], 'decode_topk', topk_graph.decode_topk)",
        "        put(modules['spark_topk'], 'decode_topk',\n"
        "            batch_topk.wrap(topk_graph.decode_topk, Path(__file__).parent/'topk-native')\n"
        "            if KERNEL_BATCH['length_aware_radix_topk'] else topk_graph.decode_topk)")
    # Clone attention only after the preceding staged globals are installed.
    # Replacing the existing entry retains uniqueness, rollback and idempotence.
    edit(name,'            for owner, name, old, new in changes:\n',
        '            for index, (owner, name, old, new) in enumerate(changes):\n'
        "                if owner is attention and name == 'packed_sparse_attention_with_lse' and KERNEL_BATCH['online_decode_attention']:\n"
        '                    new = batch_attention.wrap(new)\n'
        '                    changes[index] = (owner, name, old, new)\n')
    name='serving/spark_backend_attestation.py';source=payload[name].decode()
    assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='PRIVATE_SOURCES' for t in n.targets))
    pins=ast.literal_eval(assignment.value)
    for path in [f'ds41/{n}.py' for n in components]+['topk-native/topk.so','topk-native/complete.json']:
        pins[path]=sha(payload['serving/'+path])
    edit(name,ast.get_source_segment(source,assignment),'PRIVATE_SOURCES = '+repr(pins))
    for _ in range(32):
        for name,values in history.items():values.add(sha(payload[name]))
        updates={old:sha(payload[name]) for name,values in history.items() for old in values if old!=sha(payload[name])}
        changed=False
        for name,raw in list(payload.items()):
            if not name.endswith('.py') or not name.startswith(('serving/','tools/')):continue
            for old,new in updates.items():raw=raw.replace(old.encode(),new.encode())
            if raw!=payload[name]:payload[name]=raw;changed=True
        if not changed:break
    else:raise ValueError('Source pins did not converge')
    requirements=json.loads(payload['runtime-requirements.json'])
    requirements['kernel_batch_candidate']=dict(selected=selected,test_version=a.test_version,
        both_gpu_component_pass=True,full_model_qualified=False,serving_settings_unchanged=True)
    requirements['loaded_backend_verification']['sha256']=sha(payload['serving/spark_backend_attestation.py'])
    payload['runtime-requirements.json']=encoded(requirements)
    payload['serving/overlay-manifest.json']=encoded({n.removeprefix('serving/'):sha(raw)
        for n,raw in payload.items() if n.startswith('serving/') and n!='serving/overlay-manifest.json'})
    for name in ('kernels/length_aware_topk.cu','scripts/build_topk_candidate.py',
                 'scripts/prepare_kernel_batch_runtime.py','probes/check_kernel_batch_gpu.py'):
        payload[name]=(ROOT/name).read_bytes()
    for name,raw in payload.items():
        if name.endswith('.py'):compile(raw,name,'exec')
    manifest=dict(format=previous['format'],standalone_runtime=False,clean_rebuild_qualified=False,
        publication_approved=False,variant='kernel_batch',parent_manifest_sha256=parent_sha,
        serving_qualified=False,files={n:dict(bytes=len(raw),sha256=sha(raw)) for n,raw in sorted(payload.items())})
    for name,raw in payload.items():
        path=target/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    raw=encoded(manifest);(target/'bundle-manifest.json').write_bytes(raw);verify(target,sha(raw))
    config['kit_manifest_sha256']=sha(raw)
    for node in config['nodes']:node['kit']=str(target)
    profile.write_bytes(encoded(config))
    print(json.dumps(dict(kit=str(target),profile=str(profile),manifest_sha256=sha(raw),selected=selected)),flush=True)

if __name__=='__main__':main()
