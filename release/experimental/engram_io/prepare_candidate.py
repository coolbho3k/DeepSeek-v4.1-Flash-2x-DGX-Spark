# SPDX-License-Identifier: AGPL-3.0-only
"""Freeze an explicitly pinned Engram candidate; do not launch or publish it.

Uses the existing frozen-kit verifier and preserves serving/KV/safety settings.
All source/binary pins and live backend descriptors are updated deliberately;
no qualification checks are disabled. Local packed inputs are independently
pinned by their complete rank manifests and verified on their owning hosts.
"""
import argparse
import ast
import copy
import json
from pathlib import Path
import sys

from stage_transform import transform as transform_stage

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'dcp_overlap'))
from prepare import bounded_read, encoded, load_parent, safe_name, sha, MAX_FILE, MAX_TOTAL


def prepare(a):
    parent=Path(a.parent).absolute()
    out=Path(a.output).absolute()
    if out.resolve()!=out or out.exists() or not out.parent.is_dir() or out.is_relative_to(parent):
        raise ValueError('Fresh unredirected candidate sibling required')
    previous,payload=load_parent(parent,a.parent_sha256)
    old=copy.copy(payload)
    def edit(name,before,after):
        raw=payload[name].decode()
        if raw.count(before)!=1:
            raise ValueError('Changed source anchor in '+name+': '+before[:120])
        payload[name]=raw.replace(before,after).encode()
    def assignment(name,key):
        text=payload[name].decode()
        found=[n for n in ast.parse(text).body if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id==key for t in n.targets)]
        if len(found)!=1:
            raise ValueError('Changed assignment '+key)
        return ast.get_source_segment(text,found[0]),ast.literal_eval(found[0].value)
    library=bounded_read(Path(a.library).absolute())
    payload['serving/miaai-row-store-v1.so']=library
    payload['serving/miaai_engram.py']=transform_stage(payload['serving/miaai_engram.py'],sha(library),component=False)
    payload['serving/miaai_row_store.cpp']=bounded_read(HERE/'row_store.cpp')
    payload['serving/row_store_core.cpp']=bounded_read(HERE/'row_store_core.cpp')
    policy=dict(layout=a.layout,gpu_readable_host=a.mapped,deferred_retrieval=a.overlap,
                row_bytes_unchanged=True,reader_abi=2)
    payload['serving/ds41/engram_io/__init__.py']=b'# SPDX-License-Identifier: AGPL-3.0-only\n'
    payload['serving/ds41/engram_io/policy.py']=(
        '# SPDX-License-Identifier: AGPL-3.0-only\n'
        f'LAYOUT = {a.layout!r}\nMAPPED = {a.mapped!r}\nOVERLAP = {a.overlap!r}\n'
        f'UPSTREAM_SHA = {a.upstream_sha256!r}\n').encode()
    for name in ('integration.py','overlap.py'):
        payload['serving/ds41/engram_io/'+name]=bounded_read(HERE/name)
    edit('serving/spark_native_engram.py','    import miaai_engram as core\n',
        '    import miaai_engram as core\n'
        '    from ds41.engram_io.integration import create_stage, close_stage, install\n'
        '    install()\n')
    edit('serving/spark_native_engram.py','self._ds41_native_stage = core.NativeStage(self, library)',
        'self._ds41_native_stage = create_stage(self, library)')
    edit('serving/spark_native_engram.py','        self._ds41_native_stage.close()\n',
        '        close_stage(self._ds41_native_stage)\n')
    edit('serving/spark_native_engram.py',"DESCRIPTOR = dict(implementation=",'DESCRIPTOR = dict('+', '.join(f'{k}={v!r}' for k,v in policy.items())+', implementation=')
    backend='serving/spark_backend_attestation.py'
    before,descriptor=assignment(backend,'NATIVE_ENGRAM_DESCRIPTOR')
    descriptor.update(policy)
    edit(backend,before,'NATIVE_ENGRAM_DESCRIPTOR = '+repr(descriptor))
    # The portable host check carries the same exact descriptor, not a relaxed
    # subset. Replace its old literal before propagating all changed hashes.
    old_descriptor=ast.literal_eval(before.split('=',1)[1].strip())
    edit('tools/portable_node.py',repr(old_descriptor),repr(descriptor))
    edit(backend,'or stage.staging_bytes!=cap*536 or stage.ids.numel()!=cap',
         'or stage.staging_bytes!=cap*(272 if stage.mapped else 536) or stage.ids.numel()!=cap')
    edit(backend,"    print(json.dumps(dict(stage='ds41_native_engram_loaded',tables=2,",
        '    from ds41.engram_io.integration import audit as audit_engram_io\n'
        '    audit_engram_io(stages)\n'
        "    print(json.dumps(dict(stage='ds41_native_engram_loaded',tables=2,")

    inputs=[]
    for rank,path in enumerate((a.rank0_manifest,a.rank1_manifest)):
        raw=bounded_read(Path(path).absolute())
        manifest=json.loads(raw)
        if manifest['status']!='complete' or manifest['rank']!=rank or not manifest['independent_partition_match']:
            raise ValueError('Complete independent packed evidence is required')
        root=str(Path(manifest['shards'][0]['path']).parent)
        inputs.append(dict(root=root,manifest_sha256=sha(raw)))
    payload['tools/engram_packed_policy.py']=(
        '# SPDX-License-Identifier: AGPL-3.0-only\n'+f'LAYOUT = {a.layout!r}\nINPUTS = {inputs!r}\n').encode()
    payload['tools/engram_packed_inputs.py']=bounded_read(HERE/'packed_inputs.py')
    edit('tools/portable_node.py',"    args = ['docker','create','--name',f\"{config['run_id']}-rank{rank}\",",
        '    from engram_packed_inputs import mounts as packed_mounts\n'
        '    mounts += packed_mounts(index)\n'
        "    args = ['docker','create','--name',f\"{config['run_id']}-rank{rank}\",")
    edit('tools/portable_node.py','    weight_check = check_model_receipt(config,index,manifest)\n',
        '    weight_check = check_model_receipt(config,index,manifest)\n'
        "    packed_check = module(kit,'packed_engram_check','tools/engram_packed_inputs.py',manifest)\n"
        '    packed_check.validate(index)\n')
    before,pins=assignment(backend,'PRIVATE_SOURCES')
    for name,data in payload.items():
        if name.startswith('serving/ds41/engram_io/'):
            pins[name.removeprefix('serving/')]=sha(data)
    edit(backend,before,'PRIVATE_SOURCES = '+repr(pins))
    history={name:{sha(raw)} for name,raw in old.items()
             if name.startswith(('serving/','tools/')) and name.endswith(('.py','.json','.so'))
             and not name.endswith('overlay-manifest.json')}
    replacements={}
    for _ in range(32):
        for name,values in history.items():
            values.add(sha(payload[name]))
        updates={digest:sha(payload[name]) for name,values in history.items()
                 for digest in values if digest!=sha(payload[name])}
        replacements.update(updates)
        changed=False
        for name,raw in list(payload.items()):
            if not name.startswith(('serving/','tools/')) or not name.endswith(('.py','.json')) or name.endswith('overlay-manifest.json'):
                continue
            for before,after in updates.items():
                raw=raw.replace(before.encode(),after.encode())
            if raw!=payload[name]:
                payload[name]=raw;changed=True
        if not changed:
            break
    else:
        raise ValueError('Source pins did not converge')
    requirements=json.loads(payload['runtime-requirements.json'])
    def refresh(value):
        if isinstance(value,dict):return {k:refresh(v) for k,v in value.items()}
        if isinstance(value,list):return [refresh(v) for v in value]
        return replacements.get(value,value) if isinstance(value,str) else value
    requirements=refresh(requirements)
    requirements['engram_io_candidate']=dict(policy=policy,status='unqualified',serving_qualified=False,
        parent_qualification_applies_to_parent_only=True,serving_settings_unchanged=True,
        packed_inputs=inputs,upstream_engram_sha256=a.upstream_sha256)
    requirements['loaded_backend_verification']['sha256']=sha(payload[backend])
    payload['runtime-requirements.json']=encoded(requirements)
    payload['serving/overlay-manifest.json']=encoded({name.removeprefix('serving/'):sha(raw)
        for name,raw in payload.items() if name.startswith('serving/') and name!='serving/overlay-manifest.json'})
    for file in HERE.glob('*'):
        if file.suffix in ('.py','.cpp','.md'):
            payload['experiments/engram_io/'+file.name]=bounded_read(file)
    if len(payload)>1000 or sum(map(len,payload.values()))>MAX_TOTAL or any(len(x)>MAX_FILE for x in payload.values()):
        raise ValueError('Candidate exceeds bounded kit size')
    for name,raw in payload.items():
        if name.endswith('.py'):compile(raw,name,'exec')
    out.mkdir(mode=0o700)
    for name,raw in payload.items():
        file=out/safe_name(name);file.parent.mkdir(parents=True,exist_ok=True)
        with file.open('xb') as f:f.write(raw)
    manifest=dict(format=previous['format'],standalone_runtime=False,clean_rebuild_qualified=False,
        publication_approved=False,serving_qualified=False,variant='experimental_engram_io',
        parent_manifest_sha256=a.parent_sha256,policy=policy,
        files={n:dict(bytes=len(raw),sha256=sha(raw)) for n,raw in sorted(payload.items())})
    raw=encoded(manifest)
    with (out/'bundle-manifest.json').open('xb') as f:f.write(raw)
    load_parent(out,sha(raw))
    return dict(candidate=str(out),manifest_sha256=sha(raw),policy=policy,deployed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('parent','library','output','rank0-manifest','rank1-manifest'):
        p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--parent-sha256',required=True)
    p.add_argument('--upstream-sha256',required=True)
    p.add_argument('--layout',choices=('original','dense','page15'),default='page15')
    p.add_argument('--mapped',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--overlap',action=argparse.BooleanOptionalAction,default=True)
    print(json.dumps(prepare(p.parse_args()),indent=2))


if __name__=='__main__':main()
