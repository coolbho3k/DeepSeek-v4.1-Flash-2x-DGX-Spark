# SPDX-License-Identifier: AGPL-3.0-only
"""One-to-four-token FP32 mHC projection and squared-norm producer.

Keep all24 projection outputs, all native split outputs and the unchanged
downstream normalization/Sinkhorn consumer. No weight conversion or output
precision reduction. Requires paired numerical and performance qualification.
"""
import functools
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _prenorm(X,F,O,S,K:tl.constexpr,SPLITS:tl.constexpr,ROWS:tl.constexpr,
             BN:tl.constexpr,BK:tl.constexpr):
    part=tl.program_id(0)
    row=tl.program_id(2)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    local_k=tl.arange(0,BK)
    span=K//SPLITS
    k=part*span+local_k
    x=tl.load(X+row*K+k,local_k<span,other=0.).to(tl.float32)
    f=tl.load(F+n[:,None]*K+k[None,:],
        (n[:,None]<24)&(local_k[None,:]<span),other=0.).to(tl.float32)
    value=tl.sum(f*x[None,:],1)
    tl.store(O+(part*ROWS+row)*24+n,value,n<24)
    if tl.program_id(1)==0:
        tl.store(S+part*ROWS+row,tl.sum(x*x,0))


def forward(x,fn,out,sqrsum,splits,*,tile_n=4,warps=4):
    if (x.ndim!=2 or not 1<=x.shape[0]<=4 or x.shape[1] not in (5120,20480) or x.dtype!=torch.bfloat16
            or not x.is_cuda or not x.is_contiguous()
            or fn.shape!=(24,x.shape[1]) or fn.dtype!=torch.float32
            or not fn.is_contiguous() or type(splits) is not int
            or splits not in (4,16) or tile_n not in (4,8) or warps not in (4,8)
            or out.shape!=(splits,x.shape[0],24) or sqrsum.shape!=(splits,x.shape[0])
            or any(t.device!=x.device or t.dtype!=torch.float32 or not t.is_contiguous()
                   for t in (fn,out,sqrsum))):
        raise ValueError('Expected the bounded native small-row mHC producer contract')
    _prenorm[(splits,tr.cdiv(24,tile_n),x.shape[0])](x,fn,out,sqrsum,x.shape[1],splits,x.shape[0],
        tile_n,tr.next_power_of_2(x.shape[1]//splits),num_warps=warps,
        enable_fp_fusion=False)


def wrap(original,*,tile_n=4,warps=4):
    """Select decode/verification producers; larger prefills stay native."""
    if tile_n not in (4,8) or warps not in (4,8):
        raise ValueError('Unqualified mHC launch variant')
    @functools.wraps(original)
    def selected(x,fn,out,sqrsum,num_split):
        if len(x.shape)!=2 or not 1<=x.shape[0]<=4 or x.shape[1] not in (5120,20480):
            return original(x,fn,out,sqrsum,num_split)
        return forward(x,fn,out,sqrsum,num_split,tile_n=tile_n,warps=warps)
    selected.__ds41_mhc_decode_variant__=(tile_n,warps)
    return selected
