# SPDX-License-Identifier: AGPL-3.0-only
"""Repackage a pristine compiled image into bounded OCI layers; never start it.

Streams docker export from an owned, stopped, mount-free CPU container. Keeps
file bytes/metadata, scans credential patterns, includes source/cache archives,
and limits layers to 1 GiB before compression. No framework compilation or
model/GPU operations. Review the redacted audit before publishing the output.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import threading
import time
import uuid

TOKEN = re.compile(rb'(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})')
KEY = re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')
SENSITIVE = re.compile(r'(^|/)(?:\.ssh|\.aws|\.netrc|\.git-credentials|\.docker)(/|$)|(^|/)\.config/gh(/|$)|(^|/)(?:id_rsa|id_ed25519|token)$|^(?:home/emi|work/(?:reports|calibration|artifacts))/')
GIB = 2**30


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'))+'\n').encode()


def safe_name(name):
    name = name.removeprefix('./').rstrip('/')
    path = PurePosixPath(name)
    if not name or name == '.':
        return '.'
    if path.is_absolute() or '..' in path.parts or '\0' in name:
        raise ValueError('Unsafe export member path')
    return name


class HashWriter:
    def __init__(self, stream):
        self.stream, self.sha, self.size = stream, hashlib.sha256(), 0
    def write(self, data):
        self.sha.update(data); self.size += len(data)
        return self.stream.write(data)
    def flush(self):
        return self.stream.flush()


class ScannedReader:
    def __init__(self, stream, name, findings):
        self.stream, self.name, self.findings = stream, name, findings
        self.sha, self.tail, self.seen = hashlib.sha256(), b'', set()
    def read(self, size=-1):
        if size < 0 or size > 2**20:
            raise ValueError('Unbounded file read refused')
        block = self.stream.read(size)
        self.sha.update(block)
        data = self.tail + block
        for pattern, reason in ((TOKEN, 'token_like'), (KEY, 'private_key_header')):
            for match in pattern.finditer(data):
                # A token ending at a read boundary may continue in the next
                # chunk. Keep it in the overlap instead of recording prefixes.
                if block and match.end() == len(data) and len(match.group()) < 512:
                    continue
                digest = hashlib.sha256(match.group()).hexdigest()
                if (reason, digest) not in self.seen:
                    self.seen.add((reason, digest))
                    self.findings.append(dict(file=self.name, reason=reason, match_sha256=digest))
        self.tail = data[-512:]
        return block


class Layers:
    def __init__(self, output, limit=GIB):
        self.output, self.limit = output, limit
        self.blobs = output/'blobs/sha256'; self.blobs.mkdir(parents=True)
        self.manifests, self.diffids, self.current, self.total = [], [], None, 0

    def start(self):
        self.partial = self.blobs/('layer-%03d.partial' % len(self.manifests))
        self.raw = self.partial.open('xb')
        self.compressed = HashWriter(self.raw)
        self.gz = gzip.GzipFile(filename='', mode='wb', compresslevel=1, mtime=0, fileobj=self.compressed)
        self.plain = HashWriter(self.gz)
        self.current = tarfile.open(fileobj=self.plain, mode='w|', format=tarfile.PAX_FORMAT, copybufsize=2**20)

    def add(self, member, source=None):
        # Individual files must also fit; the donor is expected to have no
        # single file >1 GiB. Refuse, rather than creating an oversized layer.
        estimate = member.size + 16384
        if estimate > self.limit:
            raise ValueError('Member exceeds the layer budget: ' + member.name)
        if self.current and self.plain.size + estimate > self.limit:
            self.finish()
        if self.current is None:
            self.start()
        self.current.addfile(member, source)
        self.current.members.clear()

    def finish(self):
        if self.current is None:
            return
        self.current.close(); self.gz.close(); self.raw.flush()
        os.posix_fadvise(self.raw.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        self.raw.close()
        digest = self.compressed.sha.hexdigest()
        target = self.blobs/digest
        if target.exists():
            raise ValueError('Unexpected duplicate layer; preserve artifacts for review')
        self.partial.rename(target)
        self.manifests.append({'mediaType':'application/vnd.oci.image.layer.v1.tar+gzip',
                               'digest':'sha256:'+digest, 'size':self.compressed.size})
        self.diffids.append('sha256:'+self.plain.sha.hexdigest())
        self.total += self.compressed.size
        print(json.dumps(dict(stage='layer_complete', layers=len(self.manifests),
            compressed_mib=round(self.total/2**20,1))), flush=True)
        self.current = None

    def blob(self, value, media):
        raw = encoded(value); digest = hashlib.sha256(raw).hexdigest()
        (self.blobs/digest).write_bytes(raw)
        return {'mediaType':media,'digest':'sha256:'+digest,'size':len(raw)}


def build(source_image, context, output, tag, memory_floor_mib=1536):
    if not re.fullmatch('sha256:[0-9a-f]{64}', source_image):
        raise ValueError('Use an immutable installed image ID')
    if not re.fullmatch(r'ghcr\.io/coolbho3k/deepseek-v4\.1-flash-exl3-3bpw-2x-dgx-spark:[a-z0-9._-]+', tag):
        raise ValueError('Use the authorized GHCR repository and an explicit candidate tag')
    output = output.absolute(); context = context.resolve()
    if output.exists() or output.is_symlink() or output.resolve()!=output:
        raise ValueError('Use a new output directory')
    node = json.loads(subprocess.check_output(['docker','image','inspect',source_image]))[0]
    if node['Architecture']!='arm64' or node['Os']!='linux' or node['Config'].get('Volumes'):
        raise ValueError('Expected an ARM64 image without implicit volumes')
    assets = json.loads((context/'runtime-assets.json').read_bytes())
    if assets['donor_image_id'] != source_image:
        raise ValueError('Prepared assets do not match donor')
    from package_ghcr import dockerfile
    dockerfile(node)  # Preserve only explicitly supported, credential-free configuration.
    if any(TOKEN.search(value.encode()) for value in node['Config'].get('Env',[])):
        raise ValueError('Token-like image environment refused')
    available = lambda: next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    if available() < memory_floor_mib*2**20:
        raise ValueError('Insufficient headroom for background packaging')
    if os.statvfs(output.parent).f_bavail*os.statvfs(output.parent).f_frsize < 64*GIB:
        raise ValueError('Preserve at least 64 GiB workspace headroom')
    output.mkdir(); layers = Layers(output)
    findings=[]; seen=set(); count=0; added=0
    container = subprocess.check_output(['docker','create','--runtime=runc','--network=none',
        '--label=ds41.release-export='+uuid.uuid4().hex,'--entrypoint=/bin/true',source_image],text=True).strip()
    if not re.fullmatch('[0-9a-f]{64}', container):
        raise ValueError('Unexpected container ID')
    stopped=threading.Event(); breached=threading.Event(); minimum=[available()]
    with (output/'export-stderr.log').open('xb') as errors:
        child=subprocess.Popen(['docker','export',container],stdout=subprocess.PIPE,stderr=errors)
        def watch():
            while not stopped.wait(1):
                free=available();minimum[0]=min(minimum[0],free)
                rss=next(int(x.split()[1])*1024 for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:'))
                if free<memory_floor_mib*2**20 or rss>512*2**20:
                    breached.set();child.terminate();return
        monitor=threading.Thread(target=watch,daemon=True);monitor.start()
        try:
            with (output/'filesystem-inventory.jsonl').open('x') as inventory:
                with tarfile.open(fileobj=child.stdout,mode='r|') as archive:
                    for member in archive:
                        if breached.is_set():raise RuntimeError('Packaging stopped to preserve serving headroom')
                        name=safe_name(member.name);member.name=name
                        if name in seen:raise ValueError('Duplicate exported file: '+name)
                        if member.islnk() and safe_name(member.linkname) not in seen:
                            raise ValueError('Forward hard link needs explicit handling: '+name)
                        seen.add(name)
                        if SENSITIVE.search(name):findings.append(dict(file=name,reason='sensitive_path',kind='directory' if member.isdir() else 'file',bytes=member.size))
                        stream=ScannedReader(archive.extractfile(member),name,findings) if member.isfile() else None
                        layers.add(member,stream)
                        row=dict(path=name,type=member.type.decode(),size=member.size,mode=member.mode,
                            uid=member.uid,gid=member.gid,link=member.linkname)
                        if stream:row['sha256']=stream.sha.hexdigest()
                        inventory.write(json.dumps(row,sort_keys=True)+'\n')
                        count+=1;archive.members.clear()
                if child.wait()!=0:raise RuntimeError('Pristine image export failed')
                layers.finish()
                for name in ('kernel-cache.tar','runtime-source.tar.gz','runtime-assets.json'):
                    path=context/name
                    if path.is_symlink():raise ValueError('Redirected release asset')
                    member=tarfile.TarInfo('opt/ds41-release/'+name)
                    member.size=path.stat().st_size;member.mode=0o644;member.mtime=0
                    with path.open('rb') as raw:
                        stream=ScannedReader(raw,member.name,findings);layers.add(member,stream)
                    if name in assets['files'] and (stream.sha.hexdigest()!=assets['files'][name]['sha256'] or member.size!=assets['files'][name]['bytes']):
                        raise ValueError('Release asset hash changed')
                    added+=1
                layers.finish()
        finally:
            stopped.set();monitor.join(timeout=3)
            if child.poll() is None:child.terminate();child.wait(timeout=30)
            child.stdout.close()
            # Never force-remove or signal any pre-existing container.
            subprocess.run(['docker','rm',container],check=True,stdout=subprocess.DEVNULL)
    config=dict(node['Config']);config['Labels']={
        'org.opencontainers.image.title':'DeepSeek V4.1 Flash EXL3 3bpw — two DGX Sparks',
        'org.opencontainers.image.licenses':'AGPL-3.0-only',
        'org.opencontainers.image.description':'MiaAI-based serving adaptations; original vision, DSpark, FP4 KV. Third-party components retain their licenses. Fresh-clone GPU qualification pending.',
        'io.ds41.upstream':'https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks',
        'io.ds41.corresponding-source':'/opt/ds41-release/runtime-source.tar.gz',
        'io.ds41.donor':source_image}
    image_config={'architecture':'arm64','os':'linux','config':config,
        'rootfs':{'type':'layers','diff_ids':layers.diffids},
        'history':[{'created_by':'ds41 bounded pristine-filesystem repack; no compilation'} for _ in layers.diffids]}
    if node.get('Variant'):image_config['variant']=node['Variant']
    descriptor=layers.blob(image_config,'application/vnd.oci.image.config.v1+json')
    manifest={'schemaVersion':2,'mediaType':'application/vnd.oci.image.manifest.v1+json',
        'config':descriptor,'layers':layers.manifests}
    manifest_descriptor=layers.blob(manifest,'application/vnd.oci.image.manifest.v1+json')
    index={'schemaVersion':2,'manifests':[{**manifest_descriptor,'annotations':{
        'org.opencontainers.image.ref.name':tag,'io.containerd.image.name':tag}}]}
    (output/'index.json').write_bytes(encoded(index));(output/'oci-layout').write_bytes(encoded({'imageLayoutVersion':'1.0.0'}))
    report={'status':'prepared_requires_content_review','source_image':source_image,'tag':tag,
        'manifest_digest':manifest_descriptor['digest'],'image_config_digest':descriptor['digest'],
        'layers':len(layers.manifests),'compressed_bytes':layers.total,'exported_entries':count,
        'added_assets':added,'minimum_mem_available_bytes':minimum[0],'findings':findings,
        'server_stopped':False,'gpu_accessed':False,'framework_rebased':False,'fresh_clone_gpu_tested':False}
    (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='findings'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-image',required=True);p.add_argument('--context',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tag',required=True)
    p.add_argument('--memory-floor-mib',type=int,default=1536)
    a=p.parse_args();build(a.source_image,a.context,a.output,a.tag,a.memory_floor_mib)
