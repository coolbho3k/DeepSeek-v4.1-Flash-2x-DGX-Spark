"""Stable DCP2 sparse-page mapping in one kernel; no serving hooks on import.

Preserves the eager mapper's candidate order, duplicates and interior padding.
All device addresses are masked before loading. One bounded error transfer
retains synchronous request/page validation without several scalar GPU checks.
No cache payload, quantization, image visibility or ownership rule changes.
"""
import torch
import triton
import triton.language as tl

MAX_ROWS = 64
MAX_WIDTH = 8192


@triton.jit
def _map(indices, lengths, requests, table, mapped, counts, errors,
         WIDTH:tl.constexpr, I0:tl.constexpr, I1:tl.constexpr,
         L0:tl.constexpr, R0:tl.constexpr, T0:tl.constexpr, T1:tl.constexpr,
         REQUESTS:tl.constexpr, COLUMNS:tl.constexpr, STATES:tl.constexpr,
         WORLD:tl.constexpr, RANK:tl.constexpr, BLOCK:tl.constexpr):
    row=tl.program_id(0)
    col=tl.arange(0,BLOCK)
    index=tl.load(indices+row*I0+col*I1,col<WIDTH,other=-1).to(tl.int64)
    length=tl.load(lengths+row*L0).to(tl.int64)
    request=tl.load(requests+row*R0).to(tl.int64)
    request_ok=(request>=0)&(request<REQUESTS)
    owned=(col<WIDTH)&(col<length)&(index>=0)&(index%WORLD==RANK)
    local=index//WORLD
    column=local//STATES
    address_ok=owned&request_ok&(column>=0)&(column<COLUMNS)
    safe_request=tl.where(request_ok,request,0)
    safe_column=tl.where(address_ok,column,0)
    page=tl.load(table+safe_request*T0+safe_column*T1,address_ok,other=-1).to(tl.int64)
    valid=address_ok&(page>=0)
    count=tl.sum(valid.to(tl.int32),0)
    destination=tl.cumsum(valid.to(tl.int32),0)-1
    physical=page*STATES+local%STATES
    # The sentinel tail and compacted prefix are disjoint, including across
    # warps: no barrier or overlapping initialization/scatter stores needed.
    tl.store(mapped+row*WIDTH+col,-1,(col<WIDTH)&(col>=count))
    tl.store(mapped+row*WIDTH+destination,physical,valid)
    tl.store(counts+row,count)
    flags=tl.where(request_ok,0,1)
    flags=flags|tl.where(tl.sum((owned&(column>=COLUMNS)).to(tl.int32),0)>0,2,0)
    flags=flags|tl.where(tl.sum((address_ok&(page<0)).to(tl.int32),0)>0,4,0)
    tl.store(errors+row,flags)


def sparse_global_to_local_slots(indices,lengths,req_ids,block_table,
                                 storage_block_size,world_size,rank):
    """Drop unowned entries and map the remaining entries in original order."""
    if (type(storage_block_size) is not int or not 1<=storage_block_size<=4096
            or type(world_size) is not int or world_size!=2
            or type(rank) is not int or rank not in (0,1)):
        raise ValueError('Expected bounded DCP2 storage coordinates')
    if (indices.ndim!=2 or not indices.is_cuda or indices.dtype not in (torch.int32,torch.int64)
            or lengths.shape!=(indices.shape[0],) or req_ids.shape!=(indices.shape[0],)
            or block_table.ndim!=2 or indices.shape[0]>MAX_ROWS or indices.shape[1]>MAX_WIDTH
            or any(x.dtype not in (torch.int32,torch.int64) or x.device!=indices.device
                   for x in (lengths,req_ids,block_table))
            or any(s<0 for x in (indices,lengths,req_ids,block_table) for s in x.stride())):
        raise ValueError('Expected bounded integer sparse metadata on one CUDA device')
    rows,width=indices.shape
    mapped=torch.empty((rows,width),device=indices.device,dtype=indices.dtype)
    counts=torch.empty_like(lengths,memory_format=torch.contiguous_format)
    if not rows:
        return mapped,counts
    errors=torch.empty(rows,device=indices.device,dtype=torch.int32)
    _map[(rows,)](indices,lengths,req_ids,block_table,mapped,counts,errors,
        WIDTH=width,I0=indices.stride(0),I1=indices.stride(1),
        L0=lengths.stride(0),R0=req_ids.stride(0),T0=block_table.stride(0),T1=block_table.stride(1),
        REQUESTS=block_table.shape[0],COLUMNS=block_table.shape[1],STATES=storage_block_size,
        WORLD=world_size,RANK=rank,BLOCK=triton.next_power_of_2(max(width,1)),num_warps=4)
    # At most256 bytes. Retain the eager exception boundary before returning
    # addresses to attention; this path does not defer or disable validation.
    flags=errors.cpu().tolist()
    if any(value&1 for value in flags):
        raise ValueError('Invalid request index')
    if any(value&2 for value in flags):
        raise ValueError('Sparse candidate exceeds allocated block table')
    if any(value&4 for value in flags):
        raise ValueError('Unallocated sparse candidate page')
    return mapped,counts
