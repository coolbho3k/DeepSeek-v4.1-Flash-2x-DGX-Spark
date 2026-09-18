# SPDX-License-Identifier: AGPL-3.0-only
"""Publish an audited OCI candidate with the existing Docker GHCR login.

No daemon import, container lifecycle changes, or GPU work. Does not change
package visibility or overwrite an existing different tag. Requires explicit
content-review receipt matching the final manifest, not merely the donor.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(2**20):
            h.update(block)
            os.posix_fadvise(f.fileno(),max(0,f.tell()-len(block)),len(block),os.POSIX_FADV_DONTNEED)
    return h.hexdigest()


def publish(layout,crane,review,receipt):
    layout=layout.resolve();crane=crane.resolve()
    if receipt.exists():raise ValueError('Publication receipt already exists; preserve it')
    report=json.loads((layout/'publication/report.json').read_bytes())
    reviewed=json.loads(review.read_bytes())
    if reviewed.get('manifest_digest')!=report['manifest_digest'] or reviewed.get('reviewed_for_publication') is not True:
        raise ValueError('Exact final-manifest content review is required')
    index=json.loads((layout/'index.json').read_bytes())
    descriptor=index['manifests'][0]
    if descriptor['digest']!=report['manifest_digest']:raise ValueError('Reviewed manifest changed')
    tag=descriptor['annotations']['org.opencontainers.image.ref.name']
    if not tag.startswith('ghcr.io/coolbho3k/deepseek-v4.1-flash-exl3-3bpw-2x-dgx-spark:'):
        raise ValueError('Unexpected publication destination')
    blobs=layout/'blobs/sha256'
    manifest=json.loads((blobs/descriptor['digest'].split(':')[1]).read_bytes())
    for item in [descriptor,manifest['config']]+manifest['layers']:
        path=blobs/item['digest'].split(':')[1]
        if path.is_symlink() or path.stat().st_size!=item['size'] or sha(path)!=item['digest'].split(':')[1]:
            raise ValueError('OCI blob no longer matches reviewed manifest')
    env={k:v for k,v in os.environ.items() if k not in ('HF_TOKEN_WRITE','HF_TOKEN','HUGGING_FACE_HUB_TOKEN','GH_TOKEN','GITHUB_TOKEN','CR_PAT')}
    env.update(GOMAXPROCS='2',GOMEMLIMIT='256MiB',GOGC='50')
    existing=subprocess.run([str(crane),'digest',tag],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if existing.returncode==0 and existing.stdout.strip()!=descriptor['digest']:
        raise ValueError('Existing tag refers to another image; refusing overwrite')
    available=lambda:next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    if available()<1536*2**20:raise ValueError('Insufficient serving headroom for upload')
    log=receipt.with_suffix('.log');stopped=threading.Event();breached=threading.Event();minimum=[available()]
    started=time.monotonic()
    with log.open('xb') as output:
        child=subprocess.Popen([str(crane),'push',str(layout),tag],env=env,stdout=output,stderr=subprocess.STDOUT)
        def watch():
            while not stopped.wait(1):
                free=available();minimum[0]=min(minimum[0],free)
                try:rss=next(int(x.split()[1])*1024 for x in (Path('/proc')/str(child.pid)/'status').read_text().splitlines() if x.startswith('VmRSS:'))
                except (FileNotFoundError,StopIteration):return
                if free<1536*2**20 or rss>512*2**20:
                    breached.set();child.terminate();return
        monitor=threading.Thread(target=watch,daemon=True);monitor.start()
        try:
            while child.poll() is None:
                print(json.dumps({'stage':'upload_running','elapsed_s':round(time.monotonic()-started),
                    'mem_available_mib':round(available()/2**20)}),flush=True)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:pass
        finally:
            stopped.set();monitor.join(timeout=3)
            if child.poll() is None:child.terminate();child.wait(timeout=30)
    if child.returncode or breached.is_set():raise RuntimeError('Upload stopped; preserve log, server unchanged')
    digest=subprocess.check_output([str(crane),'digest',tag],env=env,text=True).strip()
    if digest!=descriptor['digest']:raise ValueError('Registry digest differs from reviewed candidate')
    reference=tag.rsplit(':',1)[0]+'@'+digest
    subprocess.run([str(crane),'validate','--remote',reference,'--fast'],env=env,check=True)
    result={'status':'uploaded_registry_metadata_verified','image':reference,'tag':tag,
        'compressed_bytes':report['compressed_bytes'],'layers':report['layers'],
        'runtime_image_identity_sha256':report['runtime_image_identity_sha256'],
        'files':report['assets']['files'],'cache_manifest_sha256':report['assets']['cache_manifest_sha256'],
        'elapsed_s':time.monotonic()-started,'minimum_mem_available_bytes':minimum[0],
        'anonymous_pull_verified':False,'fresh_clone_gpu_tested':False,'server_stopped':False}
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layout',type=Path,required=True);p.add_argument('--crane',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();publish(a.layout,a.crane,a.review,a.receipt)
