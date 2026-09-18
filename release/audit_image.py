# SPDX-License-Identifier: AGPL-3.0-only
"""Maintainer-only, bounded-memory inspection of prebuilt image distribution.

Scans every layer, including deleted files. It does not import/start the image.
Matched credential text is never printed or saved. Review uses redacted context.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import tarfile

TOKEN = re.compile(rb'hf_[A-Za-z0-9]{20,}')
KEY = re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')
SENSITIVE = re.compile(r'(^|/)(?:\.ssh|\.aws|\.netrc|\.git-credentials)(/|$)|(^|/)(?:id_rsa|id_ed25519|token)$|^(?:home/emi|work/(?:reports|calibration|artifacts))/')


class Reader:
    def __init__(self, source, prefix=b'', advise=False):
        self.source, self.prefix, self.advise = source, prefix, advise
        self.digest = hashlib.sha256()
    def read(self, n=-1):
        if n < 0:raise ValueError('Unbounded read refused')
        first, self.prefix = self.prefix[:n], self.prefix[n:]
        block = self.source.read(n-len(first))
        self.digest.update(block)
        if self.advise:
            offset=self.source.tell()
            os.posix_fadvise(self.source.fileno(),max(0,offset-len(block)),len(block),os.POSIX_FADV_DONTNEED)
        return first+block


def scan(archive, output):
    resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20))
    findings=[]; layers=0; entries=0; configs=[]
    with archive.open('rb',buffering=0) as raw:
        source=Reader(raw,advise=True)
        with tarfile.open(fileobj=source,mode='r|gz') as outer:
            for item in outer:
                if not item.isfile():continue
                stream=outer.extractfile(item); prefix=stream.read(512)
                if prefix.lstrip().startswith((b'{',b'[')) and item.size<4*2**20:
                    data=prefix+stream.read()
                    if TOKEN.search(data) or KEY.search(data):
                        findings.append(dict(layer=item.name,reason='credential_in_image_metadata'))
                    try:value=json.loads(data)
                    except ValueError:continue
                    if isinstance(value,dict) and 'rootfs' in value:
                        configs.append(dict(blob=item.name,architecture=value.get('architecture'),
                            layers=len(value['rootfs'].get('diff_ids',[]))))
                    continue
                try:layer=tarfile.open(fileobj=Reader(stream,prefix),mode='r|*')
                except tarfile.ReadError:continue
                layers+=1
                with layer:
                    for entry in layer:
                        entries+=1; name=entry.name.removeprefix('./').lstrip('/')
                        if SENSITIVE.search(name):
                            findings.append(dict(layer=item.name,file=name,reason='sensitive_path',
                                kind='directory' if entry.isdir() else 'file',bytes=entry.size))
                        if not entry.isfile():continue
                        content=layer.extractfile(entry);tail=b'';seen=set()
                        while block:=content.read(2**20):
                            data=tail+block
                            for pattern,kind in ((TOKEN,'token_like'),(KEY,'private_key_header')):
                                for match in pattern.finditer(data):
                                    digest=hashlib.sha256(match.group()).hexdigest()
                                    if digest in seen:continue
                                    seen.add(digest)
                                    before=data[max(0,match.start()-80):match.start()]
                                    after=data[match.end():match.end()+100]
                                    context=TOKEN.sub(b'<redacted>',before+b'<MATCH REDACTED>'+after)
                                    findings.append(dict(layer=item.name,file=name,reason=kind,
                                        match_sha256=digest,context=context.decode('utf-8',errors='replace')))
                            tail=data[-512:]
                print(json.dumps(dict(layers=layers,entries=entries,findings=len(findings))),flush=True)
        while source.read(2**20):pass
        result=dict(archive_sha256=source.digest.hexdigest(),bytes=archive.stat().st_size,
                    layers=layers,entries=entries,configs=configs,findings=findings,
                    status='manual_review_required' if findings else 'no_findings')
        output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ('findings','configs')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();scan(a.archive,a.output)
