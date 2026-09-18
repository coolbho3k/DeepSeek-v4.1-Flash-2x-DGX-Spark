# SPDX-License-Identifier: AGPL-3.0-only
"""Verified CPU ABI for native O_DIRECT draft records; no CUDA import."""
import ctypes as C
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import threading

from .draft_expert_records import COMPONENTS, RECORD_BYTES
from .safetensor_pack import stat_fingerprint

SLOTS = 9


class Work(C.Structure):
    _fields_ = [('store',C.c_void_p),('ids',C.c_void_p),('records',C.c_void_p),
                ('mapped_ids',C.c_void_p),('status',C.c_void_p),
                ('layer',C.c_uint64),('count',C.c_uint64)]


def checked_sha(path, expected):
    if (not isinstance(expected,str) or not re.fullmatch('[0-9a-f]{64}',expected)
            or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected):
        raise ValueError('Native draft input checksum mismatch')


def verify_bank(bank, manifest, expected_manifest_sha, rank):
    bank, manifest = Path(bank).absolute(), Path(manifest).absolute()
    if bank.resolve()!=bank or manifest.resolve()!=manifest:
        raise ValueError('Canonical bank/manifest paths required')
    checked_sha(manifest,expected_manifest_sha)
    value = json.loads(manifest.read_bytes())
    expected_components = json.loads(json.dumps(COMPONENTS))
    if (value.get('format')!='ds41_draft_raw_mxfp4_tp2_records_v1'
            or value.get('status')!='lossless_draft_tp2_bank_written_and_fully_rehashed'
            or value.get('rank')!=rank or value.get('tp_size')!=2
            or value.get('file')!=bank.name or value.get('file_bytes')!=384*RECORD_BYTES
            or value.get('record_count')!=384 or value.get('record_bytes')!=RECORD_BYTES
            or value.get('components')!=expected_components
            or value.get('requantized') is not False
            or value.get('native_deepgemm_scale_conversion_included') is not False):
        raise ValueError('Unreviewed lossless draft record manifest')
    entries = value.get('records')
    if not isinstance(entries,list) or len(entries)!=384:
        raise ValueError('Incomplete draft bank')
    for index,item in enumerate(entries):
        if (item.get('layer')!=index//128 or item.get('expert')!=index%128
                or item.get('offset')!=index*RECORD_BYTES or item.get('bytes')!=RECORD_BYTES
                or not re.fullmatch('[0-9a-f]{64}',item.get('sha256',''))):
            raise ValueError('Invalid expert bank index')
    identity = stat_fingerprint(bank.stat())
    if list(identity)!=value.get('file_fingerprint'):
        raise ValueError('Draft bank identity changed since packing')
    # Full integrity check with8MiB anonymous staging, no retained file mapping
    # or buffered whole-bank read. Record hashes are checked at bank creation;
    # this gate pins the complete file contents before callbacks are admitted.
    fd = os.open(bank,os.O_RDONLY|os.O_DIRECT|os.O_NOFOLLOW)
    digest = hashlib.sha256()
    try:
        if stat_fingerprint(os.fstat(fd))!=identity:
            raise ValueError('Draft bank changed during open')
        with mmap.mmap(-1,8*2**20) as buffer:
            for offset in range(0,identity[2],len(buffer)):
                size = min(len(buffer),identity[2]-offset)
                with memoryview(buffer)[:size] as view:
                    if os.preadv(fd,[view],offset)!=size:
                        raise OSError('Short direct bank checksum read')
                    digest.update(view)
        if digest.hexdigest()!=value.get('file_sha256'):
            raise ValueError('Draft bank payload checksum mismatch')
        if stat_fingerprint(os.fstat(fd))!=identity or stat_fingerprint(bank.stat())!=identity:
            raise ValueError('Draft bank changed during checksum')
    finally:
        os.close(fd)
    return bank,identity,value


def load_library(path,expected_sha):
    checked_sha(path,expected_sha)
    lib = C.CDLL(str(Path(path).resolve()))
    ptr,u64,i64 = C.c_void_p,C.c_uint64,C.c_int64
    lib.ds41_draft_record_abi.restype=u64
    if lib.ds41_draft_record_abi()!=1:
        raise ValueError('Unexpected native draft ABI')
    lib.ds41_draft_record_open.argtypes=[C.c_char_p,u64,u64,u64,i64,i64,u64,u64]
    lib.ds41_draft_record_open.restype=ptr
    lib.ds41_draft_record_lookup.argtypes=[ptr]
    lib.ds41_draft_record_lookup.restype=None
    lib.ds41_draft_record_stats.argtypes=[ptr,C.POINTER(u64)]
    lib.ds41_draft_record_stats.restype=None
    lib.ds41_draft_record_close.argtypes=[ptr]
    lib.ds41_draft_record_close.restype=None
    return lib


class NativeDraftRecords:
    def __init__(self,bank,manifest,library,expected_manifest_sha,expected_binary_sha,*,rank,threads=3):
        if type(rank) is not int or rank not in (0,1) or type(threads) is not int or not 0<=threads<=4:
            raise ValueError('Require TP2 rank and0..4 native reader threads')
        self.path,self.identity,self.manifest=verify_bank(bank,manifest,expected_manifest_sha,rank)
        self.lib=load_library(library,expected_binary_sha)
        self.lock=threading.Lock()
        self.store=self.lib.ds41_draft_record_open(str(self.path).encode(),*self.identity,rank,threads)
        if not self.store:
            raise RuntimeError('Native draft record open failed')
        self.rank,self.threads=rank,threads

    def execute(self,work):
        with self.lock:
            if not self.store or type(work) is not Work or work.store!=self.store:
                raise RuntimeError('Closed or foreign draft callback')
            self.lib.ds41_draft_record_lookup(C.byref(work))
            status=C.cast(work.status,C.POINTER(C.c_uint32))[0]
            if status:
                raise RuntimeError(f'Native draft callback failure flags={status}')

    def stats(self):
        with self.lock:
            if not self.store:
                raise RuntimeError('Closed draft record reader')
            result=(C.c_uint64*8)()
            self.lib.ds41_draft_record_stats(self.store,result)
            return list(result)

    def close(self):
        with self.lock:
            if self.store:
                self.lib.ds41_draft_record_close(self.store)
                self.store=None

    def __enter__(self): return self
    def __exit__(self,*_): self.close()
