# SPDX-License-Identifier: AGPL-3.0-only
# Uses the MiaAI-derived grouped kernels; source and notices are retained in
# vendor/miaai-dsv41-agpl and the additive grouped-prefill build artifact.
"""Device-only thin/fat routing; no startup registration or serving activation."""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

import torch
import triton as tr
import triton.language as tl
import spark_fused_moe as base
from spark_fused_moe_async import AsyncSmallDispatcher

BINARY_SHA='35f11df05fc5b870128db1d521513618a9f7a2ca05953b19a6c1d1194230c097'
MAX_ROWS=1056*6
MAX_SEGMENTS=384+(MAX_ROWS+63)//64
FAT_MIN=16
_installed=None


@tr.jit
def _fat_rows(Order, Mapped, Weights, ThinRows, FatRows, Tokens, Experts, SortedWeights,
              TOP:tl.constexpr, CAP:tl.constexpr, BLOCK:tl.constexpr):
    row=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    count=tl.load(FatRows)
    live=(row<CAP)&(row<count)
    source=tl.load(Order+tl.load(ThinRows)+row,live,other=0)
    expert=tl.load(Mapped+source,live,other=0)
    weight=tl.load(Weights+source,live,other=0).to(tl.float32)
    tl.store(Tokens+row,source//TOP,live)
    tl.store(Experts+row,expert,live)
    tl.store(SortedWeights+row,weight,live)


@tr.jit
def _fat_segments(Counts, RowStarts, SegStarts, Experts, Rows, Lengths,
                  CAP:tl.constexpr, BLOCK:tl.constexpr):
    expert=tl.program_id(0)
    count=tl.load(Counts+expert)
    tile=tl.arange(0,BLOCK)
    target=tl.load(SegStarts+expert)+tile
    live=(tile<(count+63)//64)&(target<CAP)
    tl.store(Experts+target,expert,live)
    tl.store(Rows+target,tl.load(RowStarts+expert)+tile*64,live)
    tl.store(Lengths+target,tl.minimum(64,count-tile*64),live)


def load_kernel(path):
    path=Path(path).resolve()
    if path.name!='ds41_miaai_fat_moe_v1.so' or hashlib.sha256(path.read_bytes()).hexdigest()!=BINARY_SHA:
        raise RuntimeError('Unqualified grouped kernel binary')
    name='ds41_miaai_fat_moe_v1'
    module=sys.modules.get(name)
    if module is None:
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        sys.modules[name]=module
    if (Path(module.__file__).resolve()!=path or module.abi()!=1003
            or module.tile_rows_gateup()!=64 or module.tile_rows_down()!=64):
        raise RuntimeError('Unexpected grouped kernel origin or ABI')
    return module


class FatWorkspace:
    def __init__(self,device):
        self.h13g=torch.empty((MAX_ROWS,5120),device=device,dtype=torch.float16)
        self.h13u=torch.empty_like(self.h13g)
        self.h2=torch.empty((MAX_ROWS,1152),device=device,dtype=torch.float16)
        self.tokens=torch.empty(MAX_ROWS,device=device,dtype=torch.int64)
        self.experts=torch.empty(MAX_ROWS,device=device,dtype=torch.int32)
        self.weights=torch.empty(MAX_ROWS,device=device,dtype=torch.float32)
        self.seg_experts=torch.empty(MAX_SEGMENTS,device=device,dtype=torch.int32)
        self.seg_rows=torch.empty_like(self.seg_experts)
        self.seg_lengths=torch.empty_like(self.seg_experts)
        self.bytes=sum(t.numel()*t.element_size() for t in vars(self).values() if isinstance(t,torch.Tensor))
        if self.bytes!=144466596:raise RuntimeError('Unexpected grouped scratch allocation')


class GroupedDispatcher(AsyncSmallDispatcher):
    def __init__(self,thin_module,fat_module):
        super().__init__(thin_module)
        self.fat_module=fat_module
        self.fat_workspace=None
        self.last_plan=None

    def __call__(self,experts,x,ids,weights,chunk_tokens=1024):
        base.validate_inputs(experts,x,ids,weights,chunk_tokens)
        if not len(x) or not experts or ids.numel()<=128 or __import__('spark_fused_moe_async').__dict__.get('_ds41_coop_eligible', lambda *_: False)(x.shape,ids.shape):
            self.last_plan=None
            return super().__call__(experts,x,ids,weights,chunk_tokens)
        with self.lock,torch.cuda.device(x.device):
            if self.failed:raise RuntimeError('Grouped dispatcher is poisoned')
            try:
                stream=torch.cuda.current_stream(x.device)
                if self.workspace is None:self.workspace=base.Workspace(self.module,x.device)
                work=self.workspace
                if work.device!=x.device:raise ValueError('Exactly one visible GPU per worker required')
                if work.pending and work.stream_id!=stream.cuda_stream:
                    stream.wait_event(work.ready);self.stream_waits+=1
                if self.fat_workspace is None:self.fat_workspace=FatWorkspace(x.device)
                fat=self.fat_workspace
                bank=self.banks.get(id(experts))
                if bank is None:
                    if len(self.banks)>=40:raise ValueError('More than forty immutable expert banks')
                    bank=base.Bank(experts,x.device);self.banks[id(experts)]=bank
                if bank.owner is not experts or len(experts)!=len(bank.keys):
                    raise ValueError('Expert bank changed after native pointer capture')
                n=len(bank.keys)
                flat=ids.reshape(-1).long()
                safe=torch.where((flat>=0)&(flat<384),flat,384)
                mapped=bank.mapping.index_select(0,safe)
                counts=torch.zeros(n+1,device=x.device,dtype=torch.int64)
                counts.scatter_add_(0,mapped,torch.ones_like(mapped))
                valid=torch.arange(n+1,device=x.device)<n
                thin_counts=torch.where(valid&(counts<FAT_MIN),counts,0)
                fat_counts=torch.where(valid&(counts>=FAT_MIN),counts,0).int()
                routed_counts=counts.index_select(0,mapped)
                key=torch.where(mapped==n,2*n,torch.where(routed_counts>=FAT_MIN,mapped+n,mapped))
                order=torch.argsort(key,stable=True)
                tokens=torch.div(order,ids.shape[1],rounding_mode='floor')
                flat_weights=weights.reshape(-1).float().contiguous()
                sorted_weights=flat_weights.index_select(0,order)
                thin_rows=thin_counts.sum(dtype=torch.int32)
                fat_rows=fat_counts.sum(dtype=torch.int32).reshape(1)
                row_starts=torch.cumsum(fat_counts,0,dtype=torch.int32)-fat_counts
                tiles=torch.div(fat_counts+63,64,rounding_mode='floor')
                seg_starts=torch.cumsum(tiles,0,dtype=torch.int32)-tiles
                num_segs=tiles.sum(dtype=torch.int32).reshape(1)
                _fat_rows[(tr.cdiv(MAX_ROWS,256),)](order,mapped,flat_weights,thin_rows,fat_rows,
                    fat.tokens,fat.experts,fat.weights,ids.shape[1],MAX_ROWS,256)
                _fat_segments[(n,)](fat_counts,row_starts,seg_starts,fat.seg_experts,
                    fat.seg_rows,fat.seg_lengths,MAX_SEGMENTS,128)
                out=torch.zeros(x.shape,device=x.device,dtype=torch.float32)
                inputs=x.half().contiguous()
                # The sorted prefix contains precisely the compacted thin rows.
                # Zero counts retire in the existing device scheduler, with no host readback.
                self.module.forward(inputs,out,thin_counts,tokens,sorted_weights,
                    bank.ptrs,work.temps,work.locks)
                p=bank.ptrs;m=self.fat_module
                m.gather(inputs,fat.tokens,fat.experts,p[1],fat.h13g,fat_rows)
                m.gather(inputs,fat.tokens,fat.experts,p[4],fat.h13u,fat_rows)
                m.gateup(fat.h13g,fat.h13u,p[0],p[3],p[2],p[5],p[7],fat.h2,fat.weights,
                    fat.seg_experts,fat.seg_rows,fat.seg_lengths,num_segs,10.,3,2)
                m.down(fat.h2,p[6],p[8],out,fat.tokens,fat.weights,
                    fat.seg_experts,fat.seg_rows,fat.seg_lengths,num_segs,3,2)
                result=out.to(x.dtype)
                self.last_plan=dict(thin_counts=thin_counts,fat_counts=fat_counts,
                    thin_rows=thin_rows,fat_rows=fat_rows,num_segments=num_segs,
                    mapped=mapped,order=order)
                self.last_schedule=dict(mode='device_only_grouped_prefill',threshold=FAT_MIN,
                    assignments=ids.numel(),host_count_readback=False,workspace_bytes=fat.bytes)
                work.ready.record(stream);work.pending=True;work.stream_id=stream.cuda_stream
                return result
            except Exception:
                self.failed=True
                raise


def register(thin_kernel_path=None,fat_kernel_path=None):
    """Startup-only opt-in; preserve the adapter callable and small-decode math."""
    global _installed
    mode=os.environ.get('DS41_ENABLE_GROUPED_PREFILL','0')
    if mode not in ('0','1'):raise ValueError('Grouped prefill mode must be exactly 0 or 1')
    if mode=='0':
        if _installed is not None:raise RuntimeError('Grouped prefill cannot be disabled after startup')
        return
    if _installed is not None and (base._dispatcher is not _installed
            or type(_installed) is not GroupedDispatcher or FAT_MIN!=16):
        raise RuntimeError('Grouped prefill binding or threshold changed')
    from spark_fused_moe_async import register as register_small
    register_small(thin_kernel_path)
    library=Path(__file__).resolve().parent/'ds41_miaai_fat_moe_v1.so' if fat_kernel_path is None else Path(fat_kernel_path)
    module=load_kernel(library)
    if _installed is not None:
        if _installed.fat_module is not module:raise RuntimeError('Grouped kernel binding changed')
        return
    previous=base._dispatcher
    if type(previous) is not AsyncSmallDispatcher or previous.workspace is not None or previous.banks:
        raise RuntimeError('Install grouped prefill before any expert forward')
    _installed=GroupedDispatcher(previous.module,module)
    base._dispatcher=_installed
    base.register(thin_kernel_path)
