# SPDX-License-Identifier: AGPL-3.0-only
"""Add the complete public recipe source and inventory to a prepared OCI image.

No registry writes or Docker/GPU operations. The final image identity changes
only because of the additive source layer; serving binaries are not changed.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tarfile

import export as public_export
from repack_oci import Layers, ScannedReader, encoded

ROOT=Path(__file__).resolve().parents[1]


def finalize(layout, context):
    layout=layout.resolve(); context=context.resolve()
    output=layout/'publication'
    if output.exists():raise ValueError('Publication source layer already exists; preserve it')
    output.mkdir()
    files=[ROOT/name for name in public_export.FILES]
    for tree in public_export.TREES:
        files.extend(p for p in (ROOT/tree).rglob('*') if p.is_file()
            and '__pycache__' not in p.parts and not p.name.endswith(('.pyc','.local.json')))
    findings=[];source=output/'runtime-source.tar.gz'
    with tarfile.open(source,'w:gz',compresslevel=1,copybufsize=2**20) as archive:
        for path in sorted(files):
            if path.is_symlink():raise ValueError('Redirected public source')
            member=archive.gettarinfo(str(path),arcname='recipe/'+path.relative_to(ROOT).as_posix())
            member.uid=member.gid=0;member.uname=member.gname='';member.mtime=0
            with path.open('rb') as raw:
                archive.addfile(member,ScannedReader(raw,member.name,findings))
    source_row={'bytes':source.stat().st_size,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    assets=json.loads((context/'runtime-assets.json').read_bytes())
    assets['files']['runtime-source.tar.gz']=source_row
    assets['packaging']='bounded_layers_merged_filesystem_no_framework_rebase'
    (output/'runtime-assets.json').write_bytes(encoded(assets))
    layers=Layers(output/'oci')
    for name in ('runtime-source.tar.gz','runtime-assets.json'):
        p=output/name;member=tarfile.TarInfo('opt/ds41-release/'+name)
        member.size=p.stat().st_size;member.mode=0o644
        with p.open('rb') as raw:layers.add(member,raw)
    layers.finish()
    blobs=layout/'blobs/sha256'
    for p in layers.blobs.iterdir():
        target=blobs/p.name
        if target.exists():raise ValueError('Source-layer blob unexpectedly exists')
        os.link(p,target)
    index=json.loads((layout/'index.json').read_bytes())
    original=index['manifests'][0]
    manifest=json.loads((blobs/original['digest'].split(':')[1]).read_bytes())
    config=json.loads((blobs/manifest['config']['digest'].split(':')[1]).read_bytes())
    config['rootfs']['diff_ids']+=layers.diffids
    config['history'] += [{'created_by':'ds41 public recipe source and MiaAI AGPLv3 attribution'} for _ in layers.diffids]
    def blob(value,media):
        raw=encoded(value);digest=hashlib.sha256(raw).hexdigest()
        path=blobs/digest
        if path.exists() and path.read_bytes()!=raw:raise ValueError('Existing blob conflict')
        path.write_bytes(raw)
        return {'mediaType':media,'digest':'sha256:'+digest,'size':len(raw)}
    descriptor=blob(config,'application/vnd.oci.image.config.v1+json')
    manifest['config']=descriptor;manifest['layers']+=layers.manifests
    new_descriptor=blob(manifest,'application/vnd.oci.image.manifest.v1+json')
    index['manifests']=[{**new_descriptor,'annotations':original.get('annotations',{})}]
    (layout/'index.json').write_bytes(encoded(index))
    spec=importlib.util.spec_from_file_location('image_identity',ROOT/'release/runtime/tools/verify_runtime_image.py')
    checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    identity=checker.identity({'Os':config['os'],'Architecture':config['architecture'],
        'Variant':config.get('variant',''),'RootFS':{'Type':'layers','Layers':config['rootfs']['diff_ids']},'Config':config['config']})
    identity_raw=checker.encoded(identity)
    (output/'runtime-image-identity.json').write_bytes(identity_raw)
    report={'status':'oci_candidate_finalized_requires_audit_review','manifest_digest':new_descriptor['digest'],
        'image_config_digest':descriptor['digest'],'compressed_bytes':sum(x['size'] for x in manifest['layers']),
        'layers':len(manifest['layers']),'source_files':len(files),'source_findings':findings,
        'runtime_image_identity_sha256':hashlib.sha256(identity_raw).hexdigest(),
        'assets':assets,'native_file_bytes_changed':False,'fresh_clone_gpu_tested':False}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('assets','source_findings')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layout',type=Path,required=True);p.add_argument('--context',type=Path,required=True)
    a=p.parse_args();finalize(a.layout,a.context)
