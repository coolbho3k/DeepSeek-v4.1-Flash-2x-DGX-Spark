# SPDX-License-Identifier: AGPL-3.0-only
"""Fuse the existing one-to-four-token route preparation, exactly.

Keep stable compact-expert ordering, the trailing missing-expert sentinel,
FP32 route weights, FP16 input conversion and a zeroed FP32 accumulator.
There is no native-kernel, expert-arithmetic, workspace or stream-policy
change. Requires paired GPU and registered-router qualification before use.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _prepare(X,Ids,Weights,Mapping,Counts,Tokens,SortedWeights,Out,HalfX,
             ROUTES:tl.constexpr,ROWS:tl.constexpr,EXPERTS:tl.constexpr,
             XR:tl.constexpr,XS:tl.constexpr,IR:tl.constexpr,IS:tl.constexpr,
             WR:tl.constexpr,WS:tl.constexpr,BS:tl.constexpr,BC:tl.constexpr):
    slot=tl.arange(0,BS)
    count_routes=ROWS*ROUTES
    raw=tl.load(Ids+(slot//ROUTES)*IR+(slot%ROUTES)*IS,slot<count_routes,other=-1).to(tl.int64)
    safe=tl.where((raw>=0)&(raw<384),raw,384)
    mapped=tl.load(Mapping+safe)
    # Rank the <=24 live assignments by (mapped expert, original position),
    # exactly matching torch.argsort(mapped, stable=True), including ties.
    earlier=(mapped[None,:]<mapped[:,None])|(
        (mapped[None,:]==mapped[:,None])&(slot[None,:]<slot[:,None]))
    order=tl.sum((earlier&(slot[None,:]<count_routes)).to(tl.int32),1)
    weight=tl.load(Weights+(slot//ROUTES)*WR+(slot%ROUTES)*WS,slot<count_routes,other=0.).to(tl.float32)
    tl.store(SortedWeights+order,weight,slot<count_routes)
    tl.store(Tokens+order,slot//ROUTES,slot<count_routes)
    expert=tl.arange(0,512)
    count=tl.sum(((expert[:,None]==mapped[None,:])&
        (slot[None,:]<count_routes)).to(tl.int32),1)
    tl.store(Counts+expert,count.to(tl.int64),expert<=EXPERTS)
    col=tl.arange(0,BC)
    value=tl.load(X+(col//5120)*XR+(col%5120)*XS,col<ROWS*5120,other=0.)
    tl.store(HalfX+col,value.to(tl.float16),col<ROWS*5120)
    tl.store(Out+col,0.,col<ROWS*5120)


def prepare(x,ids,weights,mapping,experts):
    if (x.ndim!=2 or not 1<=x.shape[0]<=4 or x.shape[1]!=5120 or x.dtype not in (torch.float16,torch.bfloat16)
            or not x.is_cuda or ids.ndim!=2 or ids.shape[0]!=x.shape[0]
            or not 1<=ids.shape[1]<=6 or ids.dtype not in (torch.int32,torch.int64)
            or weights.shape!=ids.shape or weights.dtype not in
            (torch.float16,torch.bfloat16,torch.float32)
            or any(t.device!=x.device for t in (ids,weights,mapping))
            or type(experts) is not int or not 1<=experts<=384
            or mapping.shape!=(385,) or mapping.dtype!=torch.int64
            or not mapping.is_contiguous()):
        raise ValueError('Expected the bounded small-row sparse expert bank')
    counts=torch.empty(experts+1,device=x.device,dtype=torch.int64)
    tokens=torch.empty(ids.numel(),device=x.device,dtype=torch.int64)
    sorted_weights=torch.empty(ids.numel(),device=x.device,dtype=torch.float32)
    out=torch.empty(x.shape,device=x.device,dtype=torch.float32)
    half_x=torch.empty(x.shape,device=x.device,dtype=torch.float16)
    _prepare[(1,)](x,ids,weights,mapping,counts,tokens,sorted_weights,out,half_x,
        ids.shape[1],x.shape[0],experts,x.stride(0),x.stride(1),
        ids.stride(0),ids.stride(1),weights.stride(0),weights.stride(1),
        tr.next_power_of_2(ids.numel()),tr.next_power_of_2(x.numel()),
        num_warps=8,enable_fp_fusion=False)
    return counts,tokens,sorted_weights,out,half_x


def forward_replacements():
    old='''flat = ids.reshape(-1).long()
                safe = torch.where((flat >= 0) & (flat < 384), flat, 384)
                mapped = bank.mapping.index_select(0, safe)
                order = torch.argsort(mapped, stable=True)
                # Fixed-size scatter avoids both bincount's dynamic-size
                # synchronization and the explicit counts.cpu() of the base.
                counts = torch.zeros(len(bank.keys)+1, device=x.device, dtype=torch.int64)
                counts.scatter_add_(0, mapped, torch.ones_like(mapped))
                tokens = torch.div(order, ids.shape[1], rounding_mode='floor')
                sorted_weights = weights.reshape(-1).index_select(0, order).float().contiguous()
                out = torch.zeros(x.shape, device=x.device, dtype=torch.float32)'''
    # inspect.getsource followed by dedent removes the class's four spaces.
    old=old.replace('\n                ','\n            ')
    original='\n'.join('    '+line for line in old.splitlines())
    # Original first line lacks its leading indentation inside this anchor.
    original=' '*12+original
    new='''if 1<=len(x)<=4:
                counts,tokens,sorted_weights,out,half_x = _ds41_prepare_routes(
                    x,ids,weights,bank.mapping,len(bank.keys))
            else:
'''+original+'''
                half_x=x.half().contiguous()'''
    return [
        ('return super().__call__(experts, x, ids, weights, chunk_tokens)',
         'return base.Dispatcher.__call__(self, experts, x, ids, weights, chunk_tokens)'),
        (old,new),
        ('self.module.forward(x.half().contiguous(), out, counts, tokens,',
         'self.module.forward(half_x, out, counts, tokens,'),
    ]
