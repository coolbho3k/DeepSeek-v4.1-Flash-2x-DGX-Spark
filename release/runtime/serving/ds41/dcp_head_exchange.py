# SPDX-License-Identifier: AGPL-3.0-only
"""Exchange only the32 attention heads needed by the other DCP rank.

Original FP32 partials, FP32 LSEs and merge arithmetic are preserved. Each
rank keeps its own32 heads and all-gathers only its contribution to the
peer's32 heads. No new collective primitive, quantization or persistent
workspace. This module is not registered in serving until GPU-qualified.
"""
import math

import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice

MAX_ROWS = 512
MAX_SEND_BYTES = MAX_ROWS*32*513*4


def allocate_attention_outputs(query, split_k, packed=True):
    """Let unsplit prefill write its exact FP32 exchange payload in place.

    Backing storage is[64,rows,513]. Output/LSE remain ordinary tensor views;
    each peer's32 heads are contiguous, so packing needs no GPU copy. Total
    allocated elements equal the original output plus LSE allocations.
    """
    if packed and split_k==1 and query.shape[1]==64 and 32<=len(query)<=MAX_ROWS:
        storage=torch.empty((64,len(query),513),device=query.device,dtype=torch.float32)
        return storage[...,:512].transpose(0,1),storage[...,512].transpose(0,1)
    return (torch.empty(query.shape,device=query.device,dtype=torch.float32),
            torch.empty(query.shape[:2],device=query.device,dtype=torch.float32))


def head_major_layout(output,lse):
    rows=len(output)
    return (32<=rows<=MAX_ROWS and output.stride()==(513,rows*513,1)
        and lse.stride()==(513,rows*513) and output.storage_offset()==0
        and lse.storage_offset()==512
        and output.untyped_storage().data_ptr()==lse.untyped_storage().data_ptr())


def validate_local(output,lse,rank):
    if (type(rank) is not int or rank not in (0,1)
            or output.ndim != 3 or output.shape[1:] != (64,512)
            or not 0 <= len(output) <= MAX_ROWS or not output.is_cuda
            or output.dtype != torch.float32 or lse.shape != output.shape[:2]
            or lse.dtype != torch.float32 or lse.device != output.device
            or any(s <= 0 for s in (*output.stride(),*lse.stride()))):
        raise ValueError('Expected bounded64-head FP32 DCP partials and normalizers')


def pack_result(output,lse,rank):
    validate_local(output,lse,rank)
    start = (1-rank)*32
    if head_major_layout(output,lse):
        rows=len(output)
        return output.as_strided((1,32,rows,513),(32*rows*513,rows*513,513,1),
                                 storage_offset=start*rows*513)
    return torch.cat((output[:,start:start+32],lse[:,start:start+32,None]),dim=-1).unsqueeze(0)


@tr.jit
def _merge_peers(Local,Lse,Peers,Output,OutputLse,
                 OT:tl.constexpr,OH:tl.constexpr,OC:tl.constexpr,
                 LT:tl.constexpr,LH:tl.constexpr,ROWS:tl.constexpr,RANK:tl.constexpr,
                 DT:tl.constexpr,DH:tl.constexpr,DC:tl.constexpr,
                 LOG_BASE:tl.constexpr,HEAD_MAJOR:tl.constexpr=False):
    row = tl.program_id(0)
    token,head = row//32,row%32
    col = tl.arange(0,512)
    remote = ((1-RANK)*ROWS*32+row)*513
    if HEAD_MAJOR:
        remote=((1-RANK)*32*ROWS+head*ROWS+token)*513
    local_head = RANK*32+head
    local_lse = tl.load(Lse+token*LT+local_head*LH).to(tl.float32)*LOG_BASE
    peer_lse = tl.load(Peers+remote+512).to(tl.float32)*LOG_BASE
    if RANK == 0:
        a,b = local_lse,peer_lse
    else:
        a,b = peer_lse,local_lse
    maximum = tl.maximum(a,b)
    shift = tl.where(tl.abs(maximum)==float('inf'),0.,maximum)
    normalizer = libdevice.log(libdevice.exp(a-shift)+libdevice.exp(b-shift))+shift
    wa,wb = libdevice.exp(a-normalizer),libdevice.exp(b-normalizer)
    wa = tl.where((wa==wa)&(tl.abs(wa)!=float('inf')),wa,0.)
    wb = tl.where((wb==wb)&(tl.abs(wb)!=float('inf')),wb,0.)
    own = tl.load(Local+token*OT+local_head*OH+col*OC).to(tl.float32)
    peer = tl.load(Peers+remote+col).to(tl.float32)
    if RANK == 0:
        x,y = own,peer
    else:
        x,y = peer,own
    value = tl.where(wa>0,x*wa,0.)+tl.where(wb>0,y*wb,0.)
    tl.store(Output+token*DT+head*DH+col*DC,value)
    tl.store(OutputLse+row,normalizer/LOG_BASE)


def merge_packed(output,lse,peers,rank,destination=None):
    validate_local(output,lse,rank)
    head_major=head_major_layout(output,lse)
    expected=(2,32,len(output),513) if head_major else (2,len(output),32,513)
    if (peers.shape != expected or peers.dtype != torch.float32
            or peers.device != output.device or not peers.is_contiguous()):
        raise ValueError('Expected the exact two-rank32-head packed exchange')
    if destination is not None and (destination.shape!=(len(output),32,512)
            or destination.device!=output.device or destination.dtype not in
            (torch.float32,torch.bfloat16,torch.float16)
            or any(s<=0 for s in destination.stride())):
        raise ValueError('Expected the original final attention output slice')
    result = (torch.empty((len(output),32,512),device=output.device,dtype=torch.float32)
              if destination is None else destination)
    normalizer = torch.empty((len(output),32),device=output.device,dtype=torch.float32)
    if len(output):
        _merge_peers[(len(output)*32,)](output,lse,peers,result,normalizer,
            *output.stride(),*lse.stride(),len(output),rank,*result.stride(),math.log(2.),
            HEAD_MAJOR=head_major,num_warps=4,enable_fp_fusion=False)
    return result,normalizer


def forward_replacements():
    return [(
        'all_outputs = group.all_gather(partial.unsqueeze(0), dim=0)\n'
        '                all_lses = group.all_gather(lse.unsqueeze(0), dim=0)\n'
        '                partial, _ = merge_outputs(all_outputs, all_lses, lse_base=2)\n'
        '                partial = partial[:, rank * 32:(rank + 1) * 32]\n'
        '            output[rows].copy_(partial)',
        'all_packed = group.all_gather(_ds41_pack_result(partial, lse, rank), dim=0)\n'
        '                _ds41_merge_packed(partial, lse, all_packed, rank, output[rows])\n'
        '            else:\n'
        '                output[rows].copy_(partial)')]
