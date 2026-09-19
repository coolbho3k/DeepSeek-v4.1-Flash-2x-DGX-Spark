# SPDX-License-Identifier: AGPL-3.0-only
"""Lossless immutable Engram packing; MiaAI dense layout plus local page15 layout.

The dense header and 264-byte row format are from MiaAI-Lab's row_store.cpp,
commit 979e68a62c90b24d928f5638596e0ceed90e9f34. All local code AGPLv3.
Source checkpoints are only read. Output is exclusive and published after
bounded write/readback verification; no overwrite or automatic deletion.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import time

PAGE, ROW, PER_PAGE = 4096, 264, 15
MAGIC = {'dense': 0x31344e4531565344, 'page15': 0x3247504531345344}


def fingerprint(info):
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def table(path, layer):
    path = Path(path).absolute()
    if path.resolve(strict=True) != path or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('Expected an unredirected regular checkpoint file')
    with path.open('rb') as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError('Truncated safetensors header prefix')
        size = struct.unpack('<Q', prefix)[0]
        if not 0 < size <= 2**20:
            raise ValueError('Invalid bounded safetensors header')
        header = json.loads(f.read(size))
    stem = f'layers.{layer}.engram.embed.'
    weight, scale = header[stem+'weight'], header[stem+'scale']
    rows = weight['shape'][0]
    if (weight['dtype'] != 'F8_E4M3' or scale['dtype'] != 'F8_E8M0'
            or weight['shape'] != [rows, 256] or scale['shape'] != [rows, 8]
            or not 0 < rows < 2**32):
        raise ValueError('Unexpected Engram shape or dtype')
    for item, width in ((weight, 256), (scale, 8)):
        lo, hi = item['data_offsets']
        if lo < 0 or hi-lo != rows*width or hi+8+size > path.stat().st_size:
            raise ValueError('Invalid tensor extent')
    w0,w1 = weight['data_offsets'];s0,s1 = scale['data_offsets']
    if not (w1 <= s0 or s1 <= w0):
        raise ValueError('Overlapping tensors')
    return dict(path=str(path), layer=layer, rows=rows,
        weight_offset=8+size+w0, scale_offset=8+size+s0,
        source_fingerprint=fingerprint(path.stat()))


def partitions(config):
    """Reproduce documented prime bucket sizes without importing vLLM/CUDA."""
    cfg = config.get('text_config') or config
    seen, result = set(), {}
    def prime(n):
        if n < 2:return False
        if n % 2 == 0:return n == 2
        return all(n % d for d in range(3, math.isqrt(n)+1, 2))
    for layer, rows in zip(cfg['engram_layer_ids'], cfg['engram_num_embeddings'], strict=True):
        sizes = []
        for _ in range(cfg['engram_max_ngram_size']-1):
            current = cfg['engram_vocab_size']-1
            for _ in range(cfg['engram_n_heads']):
                current += 1
                while current in seen or not prime(current):current += 1
                sizes.append(current);seen.add(current)
        if sum(sizes) != rows or len(sizes) % 2:
            raise ValueError('Expected exact two-rank complete-head partition')
        split = sum(sizes[:len(sizes)//2])
        result[layer] = dict(head_sizes=sizes, ranges=((0, split), (split, rows)))
    return result


def header_bytes(layer, total_rows, lo, hi, layout):
    if layout not in MAGIC or not 0 <= lo < hi <= total_rows:
        raise ValueError('Invalid packed shape or layout')
    values = (MAGIC[layout], layer, lo, hi, total_rows, ROW,
              PER_PAGE if layout == 'page15' else 0, PAGE if layout == 'page15' else 0)
    return struct.pack('<8Q', *values).ljust(PAGE, b'\0')


def packed_offset(row, lo, layout):
    relative = row-lo
    if relative < 0 or layout not in MAGIC:raise ValueError('Invalid packed row')
    return PAGE + (relative*ROW if layout == 'dense'
                   else relative//PER_PAGE*PAGE + relative%PER_PAGE*ROW)


def pack(source, destination, layer, lo, hi, layout, *, chunk_rows=65520):
    import numpy as np
    info = table(source, layer)
    header = header_bytes(layer, info['rows'], lo, hi, layout)
    if not 15 <= chunk_rows <= 65520 or chunk_rows % PER_PAGE:
        raise ValueError('Bounded chunk_rows must be a multiple of fifteen')
    dest = Path(destination).absolute()
    part = dest.with_name(dest.name+'.partial')
    proof = dest.with_name(dest.name+'.json')
    if dest.parent.resolve(strict=True) != dest.parent or any(p.exists() or p.is_symlink() for p in (dest,part,proof)):
        raise ValueError('Use fresh output paths in an existing real directory')
    started = time.time()
    packed_hash, w_hash, s_hash = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    source_fd = os.open(info['path'], os.O_RDONLY | os.O_NOFOLLOW)
    output_fd = os.open(part, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    def write_verified(data):
        start = os.lseek(output_fd, 0, os.SEEK_CUR)
        view = memoryview(data).cast('B')
        while view:
            count = os.write(output_fd, view)
            if count <= 0:raise RuntimeError('Short packed write')
            view = view[count:]
        os.fsync(output_fd)
        check = os.pread(output_fd, len(memoryview(data).cast('B')), start)
        if check != memoryview(data).cast('B'):
            raise RuntimeError('Packed file readback differs')
        packed_hash.update(check)
        os.posix_fadvise(output_fd, start, len(check), os.POSIX_FADV_DONTNEED)
    try:
        if fingerprint(os.fstat(source_fd)) != info['source_fingerprint']:
            raise ValueError('Source changed while opening')
        write_verified(header)
        tick = time.monotonic()
        for first in range(lo, hi, chunk_rows):
            count = min(chunk_rows, hi-first)
            woff, soff = info['weight_offset']+first*256, info['scale_offset']+first*8
            wraw, sraw = os.pread(source_fd, count*256, woff), os.pread(source_fd, count*8, soff)
            if len(wraw) != count*256 or len(sraw) != count*8:raise RuntimeError('Short source read')
            w_hash.update(wraw);s_hash.update(sraw)
            weights=np.frombuffer(wraw,dtype=np.uint8).reshape(count,256)
            scales=np.frombuffer(sraw,dtype=np.uint8).reshape(count,8)
            if layout == 'dense':
                output=np.empty((count,ROW),dtype=np.uint8)
                output[:,:256]=weights;output[:,256:]=scales
            else:
                pages=(count+PER_PAGE-1)//PER_PAGE
                output=np.zeros((pages,PAGE),dtype=np.uint8)
                grid=np.ndarray((pages,PER_PAGE,ROW),dtype=np.uint8,buffer=output,
                                strides=(PAGE,ROW,1))
                full, tail=divmod(count,PER_PAGE)
                grid[:full,:,:256]=weights[:full*PER_PAGE].reshape(full,PER_PAGE,256)
                grid[:full,:,256:]=scales[:full*PER_PAGE].reshape(full,PER_PAGE,8)
                if tail:
                    grid[full,:tail,:256]=weights[full*PER_PAGE:]
                    grid[full,:tail,256:]=scales[full*PER_PAGE:]
            write_verified(output)
            for offset,length in ((woff,len(wraw)),(soff,len(sraw))):
                os.posix_fadvise(source_fd,offset,length,os.POSIX_FADV_DONTNEED)
            if time.monotonic()-tick >= 10:
                print(json.dumps(dict(stage='packing',layer=layer,layout=layout,
                    completed_rows=first+count-lo,total_rows=hi-lo,
                    elapsed_seconds=time.time()-started)),flush=True)
                tick=time.monotonic()
        position=os.lseek(output_fd,0,os.SEEK_CUR)
        if position % PAGE:write_verified(bytes(PAGE-position%PAGE))
        if (fingerprint(os.fstat(source_fd)) != info['source_fingerprint']
                or fingerprint(Path(source).stat()) != info['source_fingerprint']):
            raise RuntimeError('Source changed during packing; output remains partial')
        size=os.fstat(output_fd).st_size
    finally:
        os.close(source_fd);os.close(output_fd)
    # Exclusive publication. Partial is our own exact inode; never overwrite.
    os.link(part,dest,follow_symlinks=False);part.unlink()
    receipt=dict(format='ds41_engram_packed_v1',layout=layout,layer=layer,lo=lo,hi=hi,
        total_rows=info['rows'],source=info,packed_sha256=packed_hash.hexdigest(),
        owned_weight_sha256=w_hash.hexdigest(),owned_scale_sha256=s_hash.hexdigest(),
        bytes=size,complete_byte_readback=True,packed_fingerprint=fingerprint(dest.stat()),
        elapsed_seconds=time.time()-started)
    with proof.open('x') as f:
        json.dump(receipt,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    fd=os.open(dest.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layer',type=int,required=True)
    p.add_argument('--lo',type=int,required=True);p.add_argument('--hi',type=int,required=True)
    p.add_argument('--layout',choices=tuple(MAGIC),required=True)
    a=p.parse_args()
    print(json.dumps(pack(a.source,a.output,a.layer,a.lo,a.hi,a.layout)),flush=True)


if __name__=='__main__':main()
