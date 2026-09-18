# SPDX-License-Identifier: AGPL-3.0-only
"""Read-only CPU check of repacked image files against the donor export.

Run inside a no-GPU, read-only container with a small memory cgroup. Container-
managed hosts/hostname/resolv.conf are deliberately excluded. Never import
torch/vLLM or call CUDA. All other exported regular file bytes are compared.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

started=time.monotonic();files=0;total=0;links=0
excluded={'etc/hosts','etc/hostname','etc/resolv.conf'}
with Path(sys.argv[1]).open() as inventory:
    for line in inventory:
        row=json.loads(line);name=row['path']
        if name in excluded or name.split('/',1)[0] in {'dev','proc','sys'}:continue
        path=Path('/')/name
        if row['type']=='2':
            if not path.is_symlink() or os.readlink(path)!=row['link']:
                raise ValueError('Repacked symlink differs: '+name)
            links+=1
        elif 'sha256' in row:
            if path.is_symlink() or not path.is_file() or path.stat().st_size!=row['size']:
                raise ValueError('Repacked file type/size differs: '+name)
            h=hashlib.sha256()
            with path.open('rb') as f:
                while block:=f.read(2**20):h.update(block)
                os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
            if h.hexdigest()!=row['sha256']:
                raise ValueError('Repacked file bytes differ: '+name)
            files+=1;total+=row['size']
            if files%50000==0:print(json.dumps({'stage':'files_verified','files':files,'bytes':total}),flush=True)
print(json.dumps({'status':'donor_regular_file_bytes_and_symlinks_preserved',
    'files':files,'symlinks':links,'bytes':total,'elapsed_s':time.monotonic()-started,
    'excluded_container_managed_files':sorted(excluded),
    'excluded_runtime_mounts':['dev','proc','sys'],'gpu_accessed':False}),flush=True)
