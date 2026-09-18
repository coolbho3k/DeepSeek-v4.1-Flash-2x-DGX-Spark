# SPDX-License-Identifier: AGPL-3.0-only
"""Fused DCP2 metadata/merge and a single packed output/LSE exchange.

No serving hooks or allocations on import. Cache access and its synchronous
bounds validation remain in the unchanged attention/mapper implementations.
These small kernels are graph-compatible, retain sparse order/duplicates,
and preserve FP32 partials until the caller's final output cast.
"""
import math

import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice

MAX_ROWS=64
MAX_WIDTH=8192


@tr.jit
def _partition(Indices,Lengths,Output,Counts,
               WIDTH:tl.constexpr,I0:tl.constexpr,I1:tl.constexpr,L0:tl.constexpr,
               RANK:tl.constexpr,LOCALIZE:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    col=tl.arange(0,BLOCK)
    value=tl.load(Indices+row*I0+col*I1,col<WIDTH,other=-1).to(tl.int64)
    length=tl.load(Lengths+row*L0)
    live=(col<WIDTH)&(col<length)&(value>=0)&(value%2==RANK)
    count=tl.sum(live.to(tl.int32),0)
    target=tl.cumsum(live.to(tl.int32),0)-1
    # Disjoint tail/prefix writes, including across warps.
    tl.store(Output+row*WIDTH+col,-1,(col<WIDTH)&(col>=count))
    if LOCALIZE:value=value//2
    tl.store(Output+row*WIDTH+target,value,live)
    tl.store(Counts+row,count)


def partition_indices(indices,lengths,rank,world_size,*,localize):
    if (type(rank) is not int or rank not in (0,1) or world_size!=2
            or type(localize) is not bool or indices.ndim not in (2,3)
            or not indices.is_cuda or indices.dtype not in (torch.int32,torch.int64)
            or lengths.device!=indices.device or lengths.dtype not in (torch.int32,torch.int64)
            or indices.shape[-1]>MAX_WIDTH):
        raise ValueError('Expected bounded integer DCP2 sparse coordinates')
    width=indices.shape[-1]
    rows=math.prod(indices.shape[:-1])
    if rows>MAX_ROWS or lengths.numel()!=rows:
        raise ValueError('Expected one length per bounded sparse row')
    output=torch.empty(indices.shape,dtype=indices.dtype,device=indices.device)
    counts=torch.empty(rows,dtype=torch.int32,device=indices.device)
    if not rows:return output,counts
    if not width:
        counts.zero_();return output,counts
    flat=indices.reshape(rows,width)
    lens=lengths.reshape(rows)
    _partition[(rows,)](flat,lens,output,counts,WIDTH=width,I0=flat.stride(0),
        I1=flat.stride(1),L0=lens.stride(0),RANK=rank,LOCALIZE=localize,
        BLOCK=tr.next_power_of_2(width),num_warps=4)
    return output,counts


@tr.jit
def _merge(Outputs,Lses,Result,ResultLse,
           OR:tl.constexpr,OT:tl.constexpr,OH:tl.constexpr,OC:tl.constexpr,
           LR:tl.constexpr,LT:tl.constexpr,LH:tl.constexpr,
           HEADS:tl.constexpr,LOG_BASE:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    token=row//HEADS;head=row%HEADS
    col=tl.arange(0,BLOCK)
    a=tl.load(Lses+token*LT+head*LH).to(tl.float32)*LOG_BASE
    b=tl.load(Lses+LR+token*LT+head*LH).to(tl.float32)*LOG_BASE
    # Match the reference's natural-log FP32 logsumexp/exp arithmetic,
    # including empty, +inf and NaN rows. Never let 0*NaN poison a sum.
    maximum=tl.maximum(a,b)
    shift=tl.where(tl.abs(maximum)==float('inf'),0.,maximum)
    normalizer=libdevice.log(libdevice.exp(a-shift)+libdevice.exp(b-shift))+shift
    wa=libdevice.exp(a-normalizer);wb=libdevice.exp(b-normalizer)
    wa=tl.where((wa==wa)&(tl.abs(wa)!=float('inf')),wa,0.)
    wb=tl.where((wb==wb)&(tl.abs(wb)!=float('inf')),wb,0.)
    x=tl.load(Outputs+token*OT+head*OH+col*OC,col<512,other=0).to(tl.float32)
    y=tl.load(Outputs+OR+token*OT+head*OH+col*OC,col<512,other=0).to(tl.float32)
    result=tl.where(wa>0,x*wa,0.)+tl.where(wb>0,y*wb,0.)
    tl.store(Result+row*512+col,result,col<512)
    tl.store(ResultLse+row,normalizer/LOG_BASE)


def merge_outputs(outputs,lses,*,lse_base):
    if (outputs.ndim!=4 or outputs.shape[0]!=2 or outputs.shape[-1]!=512
            or not 0<=outputs.shape[1]<=MAX_ROWS or outputs.shape[2] not in (32,64)
            or outputs.shape[:-1]!=lses.shape or lse_base not in (2,math.e)
            or not outputs.is_cuda or outputs.dtype not in (torch.float32,torch.bfloat16,torch.float16)
            or lses.device!=outputs.device or lses.dtype!=torch.float32):
        raise ValueError('Expected two bounded DCP partials with FP32 normalizers')
    result=torch.empty(outputs.shape[1:],device=outputs.device,dtype=outputs.dtype)
    normalizer=torch.empty(lses.shape[1:],device=outputs.device,dtype=torch.float32)
    rows,heads=outputs.shape[1:3]
    if rows:
        _merge[(rows*heads,)](outputs,lses,result,normalizer,
            OR=outputs.stride(0),OT=outputs.stride(1),OH=outputs.stride(2),OC=outputs.stride(3),
            LR=lses.stride(0),LT=lses.stride(1),LH=lses.stride(2),HEADS=heads,
            LOG_BASE=math.log(lse_base),BLOCK=512,num_warps=4,enable_fp_fusion=False)
    return result,normalizer


def pack_result(output,lse):
    if (output.ndim!=3 or output.shape[-1]!=512 or not 0<=output.shape[0]<=MAX_ROWS
            or output.shape[1]!=64 or not output.is_cuda or output.dtype!=torch.float32
            or lse.shape!=output.shape[:-1] or lse.device!=output.device or lse.dtype!=torch.float32):
        raise ValueError('Expected bounded FP32 DCP attention output and LSE')
    return torch.cat((output,lse.unsqueeze(-1)),dim=-1).unsqueeze(0)


def merge_packed(packed):
    if (packed.ndim!=4 or packed.shape[0]!=2 or packed.shape[2:]!=(64,513)
            or packed.dtype!=torch.float32 or not packed.is_contiguous()):
        raise ValueError('Expected the contiguous two-rank packed collective result')
    return merge_outputs(packed[...,:512],packed[...,512],lse_base=2)


def forward_replacements():
    """Exact source substitutions for the existing atomic FP4 installation.

    The second replacement eliminates one NCCL call per compressed layer and
    chunk. SWA-only attention, query exchange, cache mapping and image metadata
    are unchanged. Source compilation rejects missing/non-unique matches.
    """
    return [(
        'all_outputs = group.all_gather(partial.unsqueeze(0), dim=0)\n'
        '                all_lses = group.all_gather(lse.unsqueeze(0), dim=0)\n'
        '                partial, _ = merge_outputs(all_outputs, all_lses, lse_base=2)',
        'all_packed = group.all_gather(_ds41_pack_result(partial, lse), dim=0)\n'
        '                partial, _ = _ds41_merge_packed(all_packed)')]
