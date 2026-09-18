# SPDX-License-Identifier: AGPL-3.0-only
"""Exact length-aware radix selection; binary initialized before graph capture."""
import ctypes as C
import hashlib
import json
from pathlib import Path
from .length_aware_topk import wrap as prototype_wrap

class Native:
    def __init__(self, root):
        import torch
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Native top-k must prewarm before graph capture')
        root=Path(root);receipt=json.loads((root/'complete.json').read_bytes())
        binary=root/'topk.so'
        if hashlib.sha256(binary.read_bytes()).hexdigest()!=receipt['binary_sha256']:
            raise RuntimeError('Changed native top-k binary')
        self.library=C.CDLL(str(binary))
        self.library.ds41_topk_prepare.argtypes=[C.POINTER(C.c_int)]
        self.library.ds41_topk_prepare.restype=C.c_int
        self.library.ds41_topk_stage.argtypes=[C.c_void_p]*4+[C.c_int]*8+[C.c_void_p]
        self.library.ds41_topk_stage.restype=C.c_int
        info=(C.c_int*6)()
        status=self.library.ds41_topk_prepare(info)
        if status:raise RuntimeError(('Native top-k initialization failed',status))
        self.resources=list(info)

    def select(self,logits,counts,output,k):
        import torch
        rows,width=logits.shape;block=4096 if k<=1024 else 8192
        source=logits;level=0
        stream=C.c_void_p(torch.cuda.current_stream().cuda_stream)
        while True:
            blocks=(width+block-1)//block;final=blocks==1;next_width=blocks*k
            destination=output if final else torch.empty((rows,next_width),device=logits.device,dtype=torch.int64)
            status=self.library.ds41_topk_stage(source.data_ptr(),counts.data_ptr(),destination.data_ptr(),
                output.data_ptr(),rows,width,source.stride(0),next_width,k,level,int(level==0),int(final),stream)
            if status:raise RuntimeError(('Native top-k launch failed',status))
            if final:return
            source,width=destination,next_width;level+=1

def wrap(original, root='/work/artifacts/topk-build-v1'):
    result=prototype_wrap(original)
    holder={}
    def select(logits,counts,output,k):
        if 'native' not in holder:holder['native']=Native(root)
        return holder['native'].select(logits,counts,output,k)
    result.__globals__['_length_aware_select']=select
    result._ds41_native_topk_holder=holder
    return result
