# SPDX-License-Identifier: AGPL-3.0-only
"""DCP2 decode candidate blocks with device lengths and fixed graph storage.

Prefill keeps its existing packed-row implementation. Decode uses the native
width-derived candidate capacity, block maxima and newest-block pinning.
No scores, quantization, image visibility or candidate count are approximated.
"""
import torch
import triton
import triton.language as tl

from . import dcp_candidates as legacy
from .graph_validation import check_flags, require_capture_owner


@triton.jit
def _maximum(a,b):
    return tl.maximum(a,b,propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _scores(X,Lengths,Scores,WIDTH:tl.constexpr,STRIDE:tl.constexpr,
            NB:tl.constexpr,BS:tl.constexpr,RANK:tl.constexpr,REPEAT:tl.constexpr,
            TILE:tl.constexpr,LANES:tl.constexpr):
    row=tl.program_id(0)
    block=tl.program_id(1)*TILE+tl.arange(0,TILE)
    offset=tl.arange(0,LANES)
    global_col=block[:,None]*BS+offset[None,:]
    local_col=global_col//2
    length=tl.load(Lengths+row//REPEAT)
    valid=(block[:,None]<NB)&(offset[None,:]<BS)&(global_col%2==RANK)
    valid&=(local_col<length)&(local_col<WIDTH)
    values=tl.load(X+row*STRIDE+local_col,valid,other=-float('inf'))
    score=tl.reduce(values,1,_maximum)
    tl.store(Scores+row*NB+block,score,block<NB)


@triton.jit
def _mask(X,Lengths,Flags,WIDTH:tl.constexpr,STRIDE:tl.constexpr,
          NB:tl.constexpr,BS:tl.constexpr,RANK:tl.constexpr,REPEAT:tl.constexpr,
          TILE:tl.constexpr):
    row=tl.program_id(0);col=tl.program_id(1)*TILE+tl.arange(0,TILE)
    length=tl.load(Lengths+row//REPEAT)
    block=(col*2+RANK)//BS
    keep=tl.load(Flags+row*(NB+1)+block,col<WIDTH,other=0)&(col<length)
    value=tl.load(X+row*STRIDE+col,col<WIDTH,other=-float('inf'))
    tl.store(X+row*STRIDE+col,tl.where(keep,value,-float('inf')),col<WIDTH)


def _validate(logits,ends,block_size,world,rank,row_repeat):
    require_capture_owner()
    if (world!=2 or rank not in (0,1) or type(row_repeat) is not int or row_repeat<1
            or type(block_size) is not int or not 1<=block_size<=4096
            or not logits.is_cuda or logits.dtype!=torch.float32 or logits.ndim!=2
            or not 1<=logits.shape[0]<=24 or not 0<logits.shape[1]<=1048576
            or logits.stride(1)!=1 or ends.device!=logits.device
            or ends.dtype not in (torch.int32,torch.int64) or not ends.is_contiguous()
            or (logits.shape[0]-1)//row_repeat>=ends.numel()):
        raise ValueError('Expected bounded DCP2 decode candidate metadata')
    ends=ends.reshape(-1)
    errors=((ends<0)|(ends>logits.shape[1])).to(torch.int32)
    check_flags(errors,((1,'Invalid local causal bounds'),))
    return ends.clamp(0,logits.shape[1])


def select_candidate_blocks(logits,starts,ends,topk_blocks,block_size,out,group,row_repeat=1):
    if starts is not None:
        return legacy.select_candidate_blocks(logits,starts,ends,topk_blocks,block_size,out,group,row_repeat)
    world,rank=group.world_size,group.rank_in_group
    lengths=_validate(logits,ends,block_size,world,rank,row_repeat)
    rows,width=logits.shape
    if (type(topk_blocks) is not int or topk_blocks<1 or out.shape!=(rows,topk_blocks)
            or out.device!=logits.device or out.dtype not in (torch.int32,torch.int64)):
        raise ValueError('Invalid decode candidate output')
    nb=triton.cdiv(width*world,block_size)
    row_ids=torch.arange(rows,device=logits.device)//row_repeat
    local_lengths=lengths[row_ids].int().unsqueeze(1)
    global_lengths=group.all_gather(local_lengths,dim=1).sum(1)
    local_scores=torch.empty((rows,nb),device=logits.device,dtype=torch.float32)
    _scores[(rows,triton.cdiv(nb,16))](logits,lengths,local_scores,
        WIDTH=width,STRIDE=logits.stride(0),NB=nb,BS=block_size,RANK=rank,
        REPEAT=row_repeat,TILE=16,LANES=triton.next_power_of_2(block_size),num_warps=4)
    scores=group.all_gather(local_scores,dim=1).reshape(rows,world,nb).amax(1)
    newest=((global_lengths-1)//block_size).clamp(0,nb-1).unsqueeze(1)
    pinned=torch.where(global_lengths[:,None]>0,torch.inf,scores.gather(1,newest))
    scores.scatter_(1,newest,pinned)
    top=scores.topk(min(topk_blocks,nb),dim=-1)
    chosen=torch.where(top.values>-torch.inf,top.indices,-1).to(out.dtype)
    out.fill_(-1)
    out[:,:chosen.shape[1]].copy_(chosen)


def apply_candidate_mask(logits,starts,ends,candidates,block_size,world,rank,row_repeat=1):
    if starts is not None:
        return legacy.apply_candidate_mask(logits,starts,ends,candidates,block_size,world,rank,row_repeat)
    lengths=_validate(logits,ends,block_size,world,rank,row_repeat)
    rows,width=logits.shape;nb=triton.cdiv(width*world,block_size)
    if (candidates.ndim!=2 or candidates.shape[0]!=rows or candidates.device!=logits.device
            or candidates.dtype not in (torch.int32,torch.int64)):
        raise ValueError('Invalid decode candidate mask')
    flags=torch.zeros((rows,nb+1),device=logits.device,dtype=torch.bool)
    valid=(candidates>=0)&(candidates<nb)
    flags.scatter_(1,torch.where(valid,candidates,nb).long(),True)
    _mask[(rows,triton.cdiv(width,256))](logits,lengths,flags,WIDTH=width,
        STRIDE=logits.stride(0),NB=nb,BS=block_size,RANK=rank,REPEAT=row_repeat,
        TILE=256,num_warps=4)
