# SPDX-License-Identifier: AGPL-3.0-only
"""Experimental full-vocabulary Markov addition and native Gumbel fusion.

Native sampling/RNG/cache semantics are reused from Apache-2.0 vLLM. The
optional head fusion is an FP32 reduction of the original BF16 weights, with
the original BF16 head and addition rounding boundaries. No top-k pruning.
Not selected in serving until component and end-to-end qualification.
"""
import torch
from vllm.triton_utils import tl,triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_block_argmax


@triton.jit
def _sample(Base,Bias,Embed,Head,Map,Temperature,Seed,Position,Cache,Column,LocalIds,LocalMax,
            BASE_ROW:tl.constexpr,BIAS_ROW:tl.constexpr,EMBED_ROW:tl.constexpr,
            CACHE_ROW:tl.constexpr,CACHE_COL:tl.constexpr,VOCAB:tl.constexpr,PARTS:tl.constexpr,
            HEAD:tl.constexpr,FP64:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);part=tl.program_id(1)
    col=part*BLOCK+tl.arange(0,BLOCK);live=col<VOCAB
    base=tl.load(Base+row*BASE_ROW+col,live,other=-float('inf'))
    if HEAD:
        k=tl.arange(0,256)
        w=tl.load(Head+col[:,None]*256+k[None,:],live[:,None],other=0).to(tl.float32)
        e=tl.load(Embed+row*EMBED_ROW+k).to(tl.float32)
        bias=tl.sum(w*e[None,:],1).to(tl.bfloat16)
    else:
        bias=tl.load(Bias+row*BIAS_ROW+col,live,other=0)
    # PyTorch creates a BF16 sum when both native logits inputs are BF16.
    logits=(base.to(tl.float32)+bias.to(tl.float32)).to(Base.dtype.element_ty).to(tl.float32)
    value,idx=gumbel_block_argmax(logits,col,live,row,Map,Temperature,Seed,Position,
        Cache,CACHE_ROW,CACHE_COL,Column,VOCAB,IS_DRAFTING=True,APPLY_TEMPERATURE=True,
        USE_FP64=FP64,PER_TOKEN_COL=False)
    tl.store(LocalIds+row*PARTS+part,part*BLOCK+idx)
    tl.store(LocalMax+row*PARTS+part,value)


def sample(base,bias,idx_map,temperature,seeds,positions,cache,column,*,embedding=None,head=None,use_fp64=False):
    rows,vocab=base.shape
    if (base.dtype!=torch.bfloat16 or not 1<=rows<=6 or vocab!=129280 or base.stride(1)!=1
            or cache is None or cache.shape[-1]<vocab or column.ndim!=0):
        raise ValueError('Expected native full-vocabulary BF16 C1..C6 draft layout')
    fused=head is not None
    if fused:
        if (head.shape!=(vocab,256) or head.dtype!=torch.bfloat16 or not head.is_contiguous()
                or embedding.shape!=(rows,256) or embedding.dtype!=torch.bfloat16 or embedding.stride(1)!=1):
            raise ValueError('Expected original BF16 rank256 Markov head')
    elif bias.shape!=base.shape or bias.dtype!=base.dtype or bias.stride(1)!=1:
        raise ValueError('Keep original Markov bias precision')
    block=128 if fused else 1024;parts=triton.cdiv(vocab,block)
    ids=torch.empty(rows,parts,device=base.device,dtype=torch.int64)
    maxima=torch.empty(rows,parts,device=base.device,dtype=torch.float64 if use_fp64 else torch.float32)
    _sample[(rows,parts)](base,bias,embedding,head,idx_map.contiguous(),temperature,seeds,positions.contiguous(),
        cache,column,ids,maxima,base.stride(0),bias.stride(0) if bias is not None else 0,
        embedding.stride(0) if embedding is not None else 0,cache.stride(0),cache.stride(1),vocab,parts,
        fused,use_fp64,block,num_warps=8 if fused else 4,enable_fp_fusion=False)
    return ids.gather(-1,maxima.argmax(-1,keepdim=True)).view(-1)
