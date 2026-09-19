# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded CPU binding for MiaAI-derived packed Engram component tests."""
import ctypes as C
import os


class Work(C.Structure):
    _fields_=[('store',C.c_void_p),('ids',C.c_void_p),('weights',C.c_void_p),
              ('scales',C.c_void_p),('count',C.c_uint64)]


class Reader:
    def __init__(self,library,info,lo,hi,packed=None,budget=2**20,threads=4):
        if not 0 <= lo < hi <= info['rows'] or not 0 <= budget <= 64*2**20:
            raise ValueError('Invalid bounded reader configuration')
        if not 1 <= threads <= 96:raise ValueError('Invalid thread count')
        os.environ.update(OFFLOAD_MODE='ssd',DSV41_RESIDENT_SCALES='0',DSV41_IO_THREADS=str(threads))
        self.lib=C.CDLL(str(library))
        P,U=C.c_void_p,C.c_uint64
        self.lib.ds41_row_store_open.argtypes=[C.c_char_p,U,U,U,U]
        self.lib.ds41_row_store_open.restype=P
        self.lib.ds41_row_store_range.argtypes=[P,U,U]
        self.lib.ds41_row_store_attach_packed.argtypes=[P,C.c_char_p,U]
        self.lib.ds41_row_store_attach_packed.restype=C.c_int
        self.lib.ds41_row_store_lookup.argtypes=[P]
        self.lib.ds41_row_store_close.argtypes=[P]
        self.lib.row_store_stats.argtypes=[P,C.POINTER(U)]
        self.lib.ds41_row_store_profile.argtypes=[P,C.POINTER(U)]
        self.lib.ds41_row_store_clear_cache.argtypes=[P]
        self.lib.ds41_row_store_abi.restype=U
        if self.lib.ds41_row_store_abi()!=2:raise ValueError('Unexpected reader ABI')
        self.store=self.lib.ds41_row_store_open(info['path'].encode(),info['rows'],
            info['weight_offset'],info['scale_offset'],budget)
        if not self.store:raise RuntimeError('Cannot open Engram source')
        self.lib.ds41_row_store_range(self.store,lo,hi)
        if packed is not None and self.lib.ds41_row_store_attach_packed(
                self.store,str(packed).encode(),info['layer'])!=1:
            self.close();raise ValueError('Explicit packed shard did not attach')

    def close(self):
        if self.store:self.lib.ds41_row_store_close(self.store);self.store=None

    def clear(self):self.lib.ds41_row_store_clear_cache(self.store)

    def stats(self):
        a,b=(C.c_uint64*9)(),(C.c_uint64*4)()
        self.lib.row_store_stats(self.store,a);self.lib.ds41_row_store_profile(self.store,b)
        return dict(zip(('hits','misses','reads','cache_bytes','slots','scale_bytes','ways',
                         'threads','packed','requested_io_bytes','lookup_ns','lookups','layout'),[*a,*b]))

    def lookup(self,ids):
        import numpy as np
        ids=np.asarray(ids,dtype=np.int64).reshape(-1)
        if len(ids)>1056*144:raise ValueError('Excessive lookup staging')
        w,s=np.empty((len(ids),256),dtype=np.uint8),np.empty((len(ids),8),dtype=np.uint8)
        work=Work(self.store,ids.ctypes.data,w.ctypes.data,s.ctypes.data,len(ids))
        self.lib.ds41_row_store_lookup(C.byref(work))
        return w,s
