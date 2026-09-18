"""Fused bounded gather for native FP8 indexer logits; no runtime hooks.

Preserves segregated values/scales, padded page strides and native arithmetic.
Validates all used page IDs once, then gathers up to8192 states per launch.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _gather(cache, table, keys, scales, start, count,
            STATES:tl.constexpr, PAGE_STRIDE:tl.constexpr, TABLE_STRIDE:tl.constexpr,
            PAGES:tl.constexpr, BLOCK:tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
    pos = start+i
    block = tl.load(table+(pos//STATES)*TABLE_STRIDE,i<count,other=-1).to(tl.int64)
    valid = (i<count) & (block>=0) & (block<PAGES)
    offset = tl.where(valid,block,0)*PAGE_STRIDE
    row = pos%STATES
    channel = tl.arange(0,128)
    value = tl.load(cache+offset[:,None]+row[:,None]*128+channel[None,:],valid[:,None],other=0)
    scale_ptr = (cache+offset+STATES*128+row*4).to(tl.pointer_type(tl.float32))
    scale = tl.load(scale_ptr,valid,other=0)
    tl.store(keys+i[:,None]*128+channel[None,:],value,i[:,None]<count)
    tl.store(scales+i,scale,i<count)


def paged_logits(q, kv, weights, lengths, table, schedule_metadata, *,
                 max_model_len, clean_logits=False, indices=None, state_chunk=8192):
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits
    values, q_scale = q
    if (q_scale is not None or values.ndim!=4 or values.shape[1:]!=(1,32,128)
            or values.dtype!=torch.float8_e4m3fn or indices is not None
            or lengths.shape!=(values.shape[0],1) or type(state_chunk) is not int
            or not 1<=state_chunk<=8192 or type(max_model_len) is not int
            or not 1<=max_model_len<=1048576):
        raise ValueError('Expected bounded FP8 next_n1 indexer queries and context')
    if (not kv.is_cuda or kv.dtype!=torch.uint8 or kv.ndim!=4
            or kv.shape[2:]!=(1,132) or kv.shape[1] not in (64,128)
            or kv.stride(1)!=132 or kv.stride(-1)!=1
            or kv.stride(0)<kv.shape[1]*132 or kv.stride(0)%4):
        raise ValueError('Expected aligned segregated FP8 indexer pages')
    if (weights.shape!=(values.shape[0],32) or weights.dtype!=torch.float32
            or table.ndim!=2 or table.shape[0]!=values.shape[0]
            or table.dtype not in (torch.int32,torch.int64)
            or lengths.dtype not in (torch.int32,torch.int64)
            or any(x.device!=kv.device for x in (values,weights,lengths,table))):
        raise ValueError('Invalid indexer weights, lengths, table or devices')
    counts = lengths[:,0].cpu().tolist()
    if any(n<0 or n>max_model_len for n in counts):
        raise ValueError('Local context exceeds bounded logits allocation')
    required = [(n+kv.shape[1]-1)//kv.shape[1] for n in counts]
    columns = max(required,default=0)
    if columns>table.shape[1]:
        raise ValueError('Indexer block table is too short')
    if columns:
        used = torch.arange(columns,device=kv.device)[None,:] < torch.tensor(required,device=kv.device)[:,None]
        page_ids = table[:,:columns]
        if (used & ((page_ids<0) | (page_ids>=kv.shape[0]))).any().item():
            raise ValueError('Invalid physical indexer page ID')
    output = torch.full((len(counts),max_model_len),-torch.inf,device=kv.device,dtype=torch.float32)
    capacity = min(state_chunk,max(counts,default=0))
    if not capacity:
        return output
    keys = torch.empty((capacity,128),device=kv.device,dtype=torch.uint8)
    scales = torch.empty(capacity,device=kv.device,dtype=torch.float32)
    row_start = torch.zeros(1,device=kv.device,dtype=torch.int32)
    full_end = torch.full((1,),capacity,device=kv.device,dtype=torch.int32)
    for request,length in enumerate(counts):
        query = values[request].contiguous()
        weight = weights[request:request+1].contiguous()
        for start in range(0,length,capacity):
            n = min(capacity,length-start)
            _gather[(triton.cdiv(n,32),)](kv,table[request],keys,scales,start,n,
                STATES=kv.shape[1],PAGE_STRIDE=kv.stride(0),TABLE_STRIDE=table.stride(1),
                PAGES=kv.shape[0],BLOCK=32,num_warps=4)
            row_end = full_end if n==capacity else torch.full((1,),n,device=kv.device,dtype=torch.int32)
            logits = fp8_fp4_mqa_logits((query,None),(keys[:n].view(torch.float8_e4m3fn),scales[:n]),
                weight,row_start,row_end,clean_logits=False)
            output[request,start:start+n].copy_(logits[0,:n])
    return output
