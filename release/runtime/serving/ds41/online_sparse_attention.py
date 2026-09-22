# SPDX-License-Identifier: AGPL-3.0-only
"""Candidate one-pass prefill attention; not installed in serving yet.

Reuse the exact packed FP4/FP8 decoder and error masking. Online softmax
avoids the second QK/dequantization pass, retaining BF16 queries, two BF16
probability terms and FP32 accumulation. Numerical-order parity needs GPU
qualification. Small/decode split attention remains on its original path.
"""
import triton
import triton.language as tl

from .fused_sparse_attention import _selected


class PrefillDispatch:
    """Keep small-row tiles; reuse KV across32 heads for large prefill."""
    def __getitem__(self, grid):
        def launch(*args, **options):
            if options['BH'] != 16 or options['BN'] != 32 or options['num_stages'] != 1:
                raise ValueError('Unexpected baseline attention launch envelope')
            # Allocated sparse width can include image capacity even for text.
            # Query rows determine occupancy; do not gate on padded key width.
            wide = grid[0] >= 32
            heads = 32 if wide else 16
            head_major=(options['HEADS']==64 and
                args[-3].stride()==(513,grid[0]*513,1))
            return attention[(grid[0], options['HEADS'] // heads)](
                *args, **dict(options, BH=heads, SINGLE_ACC=wide,
                              OUTPUT_HEAD_MAJOR=head_major))
        return launch


prefill = PrefillDispatch()


@triton.jit
def _segment(q,cache,indices,length,maximum,total,high,low,error,
             WIDTH:tl.constexpr,CAPACITY:tl.constexpr,PAGE_STRIDE:tl.constexpr,
             STATES:tl.constexpr,FP4:tl.constexpr,SCALE:tl.constexpr,BN:tl.constexpr,
             SINGLE_ACC:tl.constexpr,SCALE_BYTES:tl.constexpr=8):
    for begin in range(tl.cdiv(length,BN)):
        position=begin*BN+tl.arange(0,BN)
        kv,live,invalid=_selected(cache,indices,position,length,WIDTH,
                                  CAPACITY,PAGE_STRIDE,STATES,FP4,SCALE_BYTES)
        tl.atomic_or(error,1,mask=tl.sum(invalid.to(tl.int32),0)>0,sem='relaxed')
        scores=tl.dot(q,tl.trans(kv)).to(tl.float32)*SCALE
        scores=tl.where(live[None,:],scores,float('-inf'))
        next_maximum=tl.maximum(maximum,tl.max(scores,1))
        safe=tl.where(next_maximum==float('-inf'),0.,next_maximum)
        finite=tl.abs(next_maximum)<float('inf')
        alpha=tl.where(finite,tl.exp(maximum-safe),0.)
        probabilities=tl.where(live[None,:]&finite[:,None],tl.exp(scores-safe[:,None]),0.)
        total=total*alpha+tl.sum(probabilities,1)
        total=tl.where(next_maximum==float('inf'),1.,total)
        p_hi=probabilities.to(tl.bfloat16)
        p_lo=(probabilities-p_hi.to(tl.float32)).to(tl.bfloat16)
        high=tl.dot(p_hi,kv,high*alpha[:,None])
        if SINGLE_ACC:
            high=tl.dot(p_lo,kv,high)
        else:
            low=tl.dot(p_lo,kv,low*alpha[:,None])
        maximum=next_maximum
    return maximum,total,high,low


@triton.jit
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
        tl.store(output+(token*HEADS+head[:,None])*512+channel[None,:],result)
        tl.store(normalizers+token*HEADS+head,lse*1.4426950408889634)
