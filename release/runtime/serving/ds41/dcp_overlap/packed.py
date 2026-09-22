# SPDX-License-Identifier: AGPL-3.0-only
# Address-only adaptations of this recipe's online_sparse_attention.attention
# and online_decode_attention._merge, in the MiaAI-derived AGPLv3 stack.
"""Write FP32 attention and LSE directly into the wire payload.

Arithmetic, reduction order and tile sizes are identical to the parent. Only
output addressing changes: each 512-channel row is followed by its LSE.
"""
import triton as tr
import triton.language as tl
from ds41.online_sparse_attention import _segment


@tr.jit
def attention(query,swa,si,sl,main,ci,cl,sinks,output,normalizers,error,
              HEADS:tl.constexpr,Q0:tl.constexpr,Q1:tl.constexpr,Q2:tl.constexpr,
              SW:tl.constexpr,SC:tl.constexpr,SP:tl.constexpr,SS:tl.constexpr,
              CW:tl.constexpr,CC:tl.constexpr,CP:tl.constexpr,CS:tl.constexpr,
              MAIN:tl.constexpr,MAIN_FP4:tl.constexpr,SINKS:tl.constexpr,
              SINK_STRIDE:tl.constexpr,SCALE:tl.constexpr,BH:tl.constexpr,BN:tl.constexpr,
              SINGLE_ACC:tl.constexpr=False,OUTPUT_HEAD_MAJOR:tl.constexpr=False,SB:tl.constexpr=8,CB:tl.constexpr=8):
    token,tile=tl.program_id(0),tl.program_id(1)
    head=tile*BH+tl.arange(0,BH)
    channel=tl.arange(0,512)
    q=tl.load(query+token*Q0+head[:,None]*Q1+channel[None,:]*Q2)
    length=tl.minimum(tl.maximum(tl.load(sl+token),0),SW)
    if SINKS:
        maximum=tl.load(sinks+head*SINK_STRIDE).to(tl.float32)
        total=tl.full((BH,),1.,tl.float32)
    else:
        maximum=tl.full((BH,),float('-inf'),tl.float32)
        total=tl.full((BH,),0.,tl.float32)
    high=tl.full((BH,512),0.,tl.float32)
    low=tl.full((BH,512),0.,tl.float32)
    maximum,total,high,low=_segment(q,swa,si+token*SW,length,maximum,total,high,low,
        error,SW,SC,SP,SS,False,SCALE,BN,SINGLE_ACC,SB)
    if MAIN:
        main_length=tl.minimum(tl.maximum(tl.load(cl+token),0),CW)
        maximum,total,high,low=_segment(q,main,ci+token*CW,main_length,maximum,total,high,low,
            error,CW,CC,CP,CS,MAIN_FP4,SCALE,BN,SINGLE_ACC,CB)
    lse=tl.where(maximum==float('-inf'),float('-inf'),maximum+tl.log(total))
    result=tl.where((total>0)[:,None]&(tl.abs(lse)<float('inf'))[:,None],
        (high+low)/total[:,None],0.)
    if OUTPUT_HEAD_MAJOR:
        rows=tl.num_programs(0)
        tl.store(output+(head[:,None]*rows+token)*513+channel[None,:],result)
        tl.store(normalizers+(head*rows+token)*513,lse*1.4426950408889634)
    else:
        tl.store(output+(token*HEADS+head[:,None])*513+channel[None,:],result)
        tl.store(normalizers+(token*HEADS+head)*513,lse*1.4426950408889634)


@tr.jit
def merge(partial, local_lse, output, normalizers, SPLITS: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    d = tile * 128 + tl.arange(0, 128)
    split = tl.arange(0, SPLITS)
    lse = tl.load(local_lse + row * SPLITS + split)
    maximum = tl.max(lse, 0)
    safe = tl.where(tl.abs(maximum) < float('inf'), maximum, 0.)
    weights = tl.where(tl.abs(lse) < float('inf'), tl.exp(lse - safe), 0.)
    total = tl.sum(weights, 0)
    weights = tl.where((total > 0) & (tl.abs(maximum) < float('inf')), weights / total, 0.)
    values = tl.load(partial + (row * SPLITS + split[:, None]) * 512 + d[None, :])
    tl.store(output + row * 513 + d, tl.sum(values * weights[:, None], 0))
    if tile == 0:
        normalizer = tl.where(tl.abs(maximum) == float('inf'), maximum, maximum + tl.log(total))
        tl.store(normalizers + row * 513, normalizer * 1.4426950408889634)
