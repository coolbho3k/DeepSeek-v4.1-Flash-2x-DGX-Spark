# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only, pinned local packed-artifact checks for the experimental launcher.

No rehashing 100 GiB at startup: packing verified every output byte and its
receipt is independently pinned. Verify receipt plus unchanged inode metadata,
source identity, exact ownership/header and read-only container mounts.
"""
import hashlib
import json
from pathlib import Path
import struct

from engram_packed_policy import INPUTS, LAYOUT


def fingerprint(info):
    return [info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns]


def mounts(rank):
    if LAYOUT=='original':
        return []
    if rank not in (0,1):
        raise ValueError('Invalid packed rank')
    root=Path(INPUTS[rank]['root'])
    return [(str(root/f'engram-layer-{layer:02}-rank{rank}-{LAYOUT}.bin'),
             f'/opt/ds41-engram-packed/engram-layer-{layer:02}-rank{rank}-{LAYOUT}.bin',True)
            for layer in (1,14)]


def validate(rank):
    if LAYOUT=='original':
        return dict(layout=LAYOUT,additional_files=0)
    root=Path(INPUTS[rank]['root'])
    receipt=root/'rank-manifest.json'
    if root.resolve()!=root or receipt.resolve()!=receipt or receipt.stat().st_size>32768:
        raise ValueError('Unexpected or redirected packed receipt')
    raw=receipt.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=INPUTS[rank]['manifest_sha256']:
        raise ValueError('Packed manifest changed')
    manifest=json.loads(raw)
    if manifest['status']!='complete' or manifest['rank']!=rank or not manifest['independent_partition_match']:
        raise ValueError('Incomplete packed artifacts')
    selected=[s for s in manifest['shards'] if s['layout']==LAYOUT]
    if len(selected)!=2 or {s['layer'] for s in selected}!={1,14}:
        raise ValueError('Expected exactly two selected packed tables')
    for shard in selected:
        path=root/f'engram-layer-{shard["layer"]:02}-rank{rank}-{LAYOUT}.bin'
        if (str(path)!=shard['path'] or path.resolve()!=path or not path.is_file()
                or fingerprint(path.stat())!=shard['packed_fingerprint']
                or not shard['complete_byte_readback']):
            raise ValueError('Packed data changed after byte verification')
        source=Path(shard['source']['path'])
        if source.resolve()!=source or fingerprint(source.stat())!=shard['source']['source_fingerprint']:
            raise ValueError('Original weights changed after packing')
        bounds=manifest['partitions'][str(shard['layer'])]['ranges'][rank]
        if bounds!=[shard['lo'],shard['hi']]:
            raise ValueError('Packed ownership differs from verified vLLM partition')
        with path.open('rb') as f:
            header=struct.unpack('<8Q',f.read(64))
        magic=0x3247504531345344 if LAYOUT=='page15' else 0x31344e4531565344
        expected=(magic,shard['layer'],*bounds,shard['total_rows'],264,
                  15 if LAYOUT=='page15' else 0,4096 if LAYOUT=='page15' else 0)
        if header!=expected:
            raise ValueError('Unexpected packed header')
    return dict(layout=LAYOUT,additional_files=2,source_and_packed_identity_unchanged=True)
