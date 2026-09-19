# SPDX-License-Identifier: AGPL-3.0-only
"""Draft-only FP32 grouped MoE candidate inside the real dispatcher owner.

Retains the original route sorting, FP32 route weights, per-expert FP16
output boundary and FP32 accumulation. Reuses the four existing scratch
buffers; metadata occupies a disjoint tail of the first buffer.
"""
import ctypes as C
import hashlib
from pathlib import Path


def eligible(x_shape,ids_shape,experts):
    return (len(x_shape)==2 and 1<=x_shape[0]<=30 and x_shape[1]==5120
        and tuple(ids_shape)==(x_shape[0],3) and 1<=experts<=128)


def metadata_interval(rows):
    if type(rows) is not int or not 1<=rows<=30:raise ValueError('Draft rows outside K5/C6 envelope')
    capacity=6*128*5120*2
    start=capacity-5120*2
    end=start+3*rows*8
    if not rows*3*5120*2<=start<end<=capacity:raise ValueError('Overlapping draft metadata')
    return start,end


class NativeTop3:
    def __init__(self,work,binary,digest):
        import torch
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('Prewarm draft native resources before capture')
        binary=Path(binary)
        if hashlib.sha256(binary.read_bytes()).hexdigest()!=digest:raise ValueError('Changed draft binary')
        self.device=work.device;self.library=C.CDLL(str(binary))
        if self.library.ds41_draft_top3_abi()!=1:raise ValueError('Unknown top3 ABI')
        self.library.ds41_draft_top3_info.argtypes=[C.POINTER(C.c_int)]
        self.library.ds41_draft_top3_info.restype=C.c_int
        info=(C.c_int*12)()
        status=self.library.ds41_draft_top3_info(info)
        if (status or tuple(info)[10:]!=(30,90)
                or any(info[i]!=256 or info[i+1]!=8192 or info[i+3]>8 or info[i+4]<2 for i in (0,5))):
            raise RuntimeError(f'Unexpected FP32 grouped draft resources: {status}, {list(info)}')
        self.resources=tuple(info)
        self.launch=self.library.ds41_draft_top3_launch
        self.launch.argtypes=[C.POINTER(C.c_void_p),C.c_int,C.c_int,C.c_void_p]
        self.launch.restype=C.c_int

    def __call__(self,work,bank,x,ids,weights):
        import torch
        if not eligible(x.shape,ids.shape,len(bank.keys)) or x.device!=self.device:
            raise ValueError('Unsupported top3 native shape')
        shapes=((6,128,5120),(6,128,5120),(6,128,1152),(6,128,1152))
        if (len(work.temps)!=4 or len(bank.ptrs)!=9
                or any(tuple(t.shape)!=s or t.dtype!=torch.float16 or not t.is_contiguous()
                       or t.device!=x.device for t,s in zip(work.temps,shapes))):
            raise ValueError('Keep the original serialized draft workspace')
        metadata_interval(len(x))
        meta=work.temps[0][-1,-1,:12*len(x)].view(torch.int64)
        # Exact native async route preparation; no histogram readback.
        flat=ids.reshape(-1).long()
        safe=torch.where((flat>=0)&(flat<384),flat,384)
        mapped=bank.mapping.index_select(0,safe)
        order=torch.argsort(mapped,stable=True)
        counts=torch.zeros(len(bank.keys)+1,device=x.device,dtype=torch.int64)
        counts.scatter_add_(0,mapped,torch.ones_like(mapped))
        tokens=torch.div(order,3,rounding_mode='floor')
        sorted_weights=weights.reshape(-1).index_select(0,order).float().contiguous()
        out=torch.zeros(x.shape,device=x.device,dtype=torch.float32)
        tensors=[x.half().contiguous(),out,counts,tokens,sorted_weights,*bank.ptrs,*work.temps,meta]
        pointers=(C.c_void_p*19)(*[t.data_ptr() for t in tensors])
        status=self.launch(pointers,len(x),len(bank.keys),C.c_void_p(torch.cuda.current_stream(x.device).cuda_stream))
        if status:raise RuntimeError(f'Draft top3 CUDA failure: {status}; no retry')
        return out


def configure(binary,digest):
    """Caller already holds the existing dispatcher lock and ready-event fence."""
    def call(work,bank,x,ids,weights):
        native=getattr(work,'_ds41_draft_top3',None)
        if native is None:
            native=NativeTop3(work,binary,digest);work._ds41_draft_top3=native
        return native(work,bank,x,ids,weights).to(x.dtype)
    return call
