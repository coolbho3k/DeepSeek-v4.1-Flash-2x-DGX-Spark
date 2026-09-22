# SPDX-License-Identifier: Apache-2.0
"""Bounded SWA32/BF16 native-writer, reader, graph and timing qualification."""
import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path
import random
import statistics
import sys
import types

import torch


def oracle(x, group):
    # Independent CPU float64 quantizer: pow2 scale, E4M3 nearest-even.
    values = x[:, :448].double().reshape(len(x), 448 // group, group)
    exponent = torch.ceil(torch.log2(values.abs().amax(-1).clamp_min(1e-4) / 448))
    scales = torch.exp2(exponent)
    codes = (values / scales[..., None]).to(torch.float8_e4m3fn)
    return codes.view(torch.uint8).reshape(len(x), 448), (exponent + 127).byte(), (codes.double() * scales[..., None]).reshape(len(x), 448)


def time_pair(functions, nodes=32, repeats=17):
    graphs = []
    for fn in functions:
        for _ in range(3): fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(nodes): fn()
        graphs.append(graph)
    for _ in range(3):
        for g in graphs: g.replay()
    samples = [[] for _ in functions]
    rng = random.Random(416)
    for _ in range(repeats):
        order = list(range(len(graphs))); rng.shuffle(order)
        for i in order:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); graphs[i].replay(); b.record(); b.synchronize()
            samples[i].append(a.elapsed_time(b) * 1000 / nodes)
    return [dict(median_us=statistics.median(v), min_us=min(v), max_us=max(v)) for v in samples]


def check_pair(name, count, heads, padded, states, stride, old, new, *, display=None, timings=False):
    torch.manual_seed(416 + count)
    q = torch.randn(count, heads, 512, device='cuda', dtype=torch.bfloat16)
    kv = torch.randn(count, 512, dtype=torch.bfloat16)
    if name == 'mixed_scales':
        values = kv[:, :448].reshape(count, 7, 64)
        values[:, :, :32] *= 1e-4
        values[:, :, 32:] = 10
    if name == 'ties_and_zeros':
        kv[0].zero_(); kv[0, 1::2] = -0.0
        kv[1, :448] = torch.tensor([0.00000001, -0.00000001, .5, -.5, 1., 1.0625, 448., -448.] * 56).bfloat16()
    cpu_values = kv.clone(); kv = kv.cuda()
    slots = torch.arange(count, dtype=torch.int64)
    if count > 4 and not timings: slots[1::5] = -1
    # DP padding: query processing extends beyond the inserted rows.
    if name == 'dp_padding': slots = slots[:max(0, count - 3)]
    gpu_slots = slots.cuda(); positions = torch.arange(count, device='cuda')
    phase = torch.arange(max(1, count)*32, device='cuda').reshape(max(1, count), 32).float() * .037
    cs = torch.cat((phase.cos(), phase.sin()), -1)
    pages = max(1, (count + states - 1)//states)
    stores = []
    for i, width in enumerate((584, 592)):
        if display is None:
            backing = torch.full((pages, stride), 165, device='cuda', dtype=torch.uint8)
        else:
            half = display.numel() // 2
            assert pages * stride <= half
            backing = display[i*half:i*half+pages*stride].view(pages, stride)
            backing.fill_(165)
        cache = backing[:, :states*width].view(pages, states, width)
        stores.append((backing, cache))
    args = (gpu_slots, positions, cs, padded, 1e-6, states, False)
    outputs = [fn(q, kv, backing[:, :states*width], *args) for fn,(backing,_),width in zip((old,new),stores,(584,592))]
    torch.cuda.synchronize()
    assert torch.equal(outputs[0], outputs[1]), (name, 'Q/RoPE mismatch')
    if padded > heads: assert not outputs[1][:, heads:].view(torch.int16).any()
    live = [(i,int(slot)) for i,slot in enumerate(slots) if slot >= 0]
    snapshots = [backing.cpu() for backing,_ in stores]
    reconstructed = []
    for group, width, snapshot in zip((64,32),(584,592),snapshots):
        codes, scales, decoded = oracle(cpu_values,group)
        expected = torch.full_like(snapshot,165)
        for row,slot in live:
            page,state = divmod(slot,states); offset=state*576
            expected[page,offset:offset+448] = codes[row]
            # Original native RoPE rounding is the exact byte reference.
            expected[page,offset+448:offset+576] = snapshots[0][page,offset+448:offset+576]
            base=states*576+state*(width-576)
            expected[page,base:base+448//group] = scales[row]
            expected[page,base+448//group:base+width-576] = 0
        if not torch.equal(snapshot,expected):
            loc=(snapshot!=expected).nonzero()[:32]
            raise AssertionError((name,group,'cache bytes or canary changed',[(a,b,int(snapshot[a,b]),int(expected[a,b])) for a,b in loc.tolist()]))
        reconstructed.append(decoded)
    x=cpu_values[:,:448].double()
    errors=[((v-x)**2).reshape(count,14,32).sum(-1) for v in reconstructed]
    assert torch.all(errors[1] <= errors[0]), (name,'SWA32 error regressed')
    if name=='mixed_scales': assert errors[1].sum() < errors[0].sum()
    result=dict(case=name,rows=count,heads=heads,padded_heads=padded,states=states,page_stride=stride,
                compared_groups=errors[0].numel(),improved_groups=int((errors[1]<errors[0]).sum()),
                regressed_groups=int((errors[1]>errors[0]).sum()),group64_sse=float(errors[0].sum()),group32_sse=float(errors[1].sum()),
                q_bit_exact=True,rope_bit_exact=True,canaries_preserved=True)
    from ds41.dcp_cache_gather import gather
    selected=torch.tensor([slot for _,slot in live]+[-1],device='cuda')
    restored=gather(stores[1][1],selected).cpu()
    for i,(row,slot) in enumerate(live):
        assert torch.equal(restored[i,:448],reconstructed[1][row].bfloat16())
        page,state=divmod(slot,states)
        expected_rope=snapshots[0][page,state*576+448:state*576+576].contiguous().view(torch.bfloat16)
        assert torch.equal(restored[i,448:].view(torch.int16),expected_rope.view(torch.int16))
    assert not restored[-1].any()
    del outputs
    if timings:
        functions=[lambda fn=fn,backing=backing,width=width: fn(q,kv,backing[:,:states*width],*args)
                   for fn,(backing,_),width in zip((old,new),stores,(584,592))]
        speed=time_pair(functions)
        result['timings_us']=dict(group64=speed[0],group32=speed[1])
        for (backing,_),expected in zip(stores,snapshots):assert torch.equal(backing.cpu(),expected)
        result['cuda_graph_replay']=True
    print(json.dumps(result),flush=True)
    return result


def check_readers(old,new):
    from ds41 import fused_sparse_attention as fused, online_sparse_attention as prefill, online_decode_attention as decode, fp4_main_kv as fp4
    from ds41.dcp_overlap import packed
    from ds41.dcp_cache_gather import gather
    n=96;states=32;stride=115200
    values=torch.randn(n,512,device='cuda',dtype=torch.bfloat16)
    q_unused=torch.zeros(n,32,512,device='cuda',dtype=torch.bfloat16)
    slots=torch.arange(n,device='cuda');cs=torch.zeros(n,64,device='cuda');cs[:,:32]=1
    backing=torch.zeros(3,stride,device='cuda',dtype=torch.uint8)
    cache=backing[:,:states*592].view(3,states,592)
    new(q_unused,values,backing[:,:states*592],slots,slots,cs,32,1e-6,states,False)
    main=torch.zeros(1,128,288,device='cuda',dtype=torch.uint8)
    fp4.store(main,values[:16],slots[:16],check_bounds=False)
    results=[]
    for count,width in ((1,16),(8,96),(32,512)):
        query=torch.randn(count,32,512,device='cuda',dtype=torch.bfloat16)*.1
        si=(torch.arange(width,device='cuda')%n).repeat(count,1);si[:,-2:]=-1
        sl=torch.full((count,),width,device='cuda',dtype=torch.int32)
        ci=torch.arange(16,device='cuda').repeat(count,1);cl=torch.full((count,),16,device='cuda',dtype=torch.int32)
        sinks=torch.zeros(32,device='cuda')
        keys=torch.cat((gather(cache,si),fp4.gather(main,ci)),1).double()
        valid=torch.cat((si>=0,ci>=0),1)
        scores=torch.einsum('qhd,qkd->qhk',query.double(),keys)*(512**-.5)
        scores=scores.masked_fill(~valid[:,None,:],-torch.inf)
        dense_lse=torch.logsumexp(torch.cat((scores,torch.zeros(count,32,1,device='cuda',dtype=torch.float64)),2),2)
        expected=torch.einsum('qhk,qkd->qhd',torch.exp(scores-dense_lse[...,None]),keys)
        common=dict(HEADS=32,Q0=query.stride(0),Q1=query.stride(1),Q2=1,
            SW=width,SC=n,SP=stride,SS=states,SB=16,CW=16,CC=128,CP=main.stride(0),CS=128,CB=8,
            MAIN=True,MAIN_FP4=True,SINKS=True,SINK_STRIDE=1,SCALE=512**-.5,BH=16,BN=32,
            num_warps=8,num_stages=1,enable_fp_fusion=False)
        args=(query,cache,si,sl,main,ci,cl,sinks)
        error=torch.zeros((),device='cuda',dtype=torch.int32)
        variants={}
        variants['two_pass']=fused.packed_sparse_attention_with_lse(query,cache,si,sl,compressed_cache=main,compressed_indices=ci,compressed_lengths=cl,sinks=sinks)
        for label,fn in (('prefill',prefill.attention),('dcp_prefill',packed.attention)):
            if label=='dcp_prefill':
                buf=torch.empty(count,32,513,device='cuda');out,lse=buf[...,:512],buf[...,512]
            else:out,lse=torch.empty_like(query,dtype=torch.float32),torch.empty(count,32,device='cuda')
            fn[(count,2)](*args,out,lse,error,**common)
            variants[label]=(out,lse)
        partial=torch.empty(count,32,2,512,device='cuda');local=torch.empty(count,32,2,device='cuda')
        out=torch.empty_like(query,dtype=torch.float32);lse=torch.empty(count,32,device='cuda')
        decode._online[(count,2,2)](*args,partial,local,error,**common,SPLITS=2)
        decode._merge[(count*32,4)](partial,local,out,lse,SPLITS=2,num_warps=4,enable_fp_fusion=False)
        variants['decode']=(out,lse)
        for label,(out,lse) in variants.items():
            nmse=float(((out.double()-expected)**2).sum()/expected.square().sum().clamp_min(1e-30))
            assert nmse<1e-8,(label,count,nmse)
            assert torch.allclose(lse.double(),dense_lse/torch.log(torch.tensor(2.,device='cuda',dtype=torch.float64)),atol=3e-5,rtol=1e-5)
            results.append(dict(path=label,rows=count,visible_width=width,nmse=nmse))
        assert error.item()==0
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--display-library',type=Path)
    args=parser.parse_args();torch.set_num_threads(2)
    package=types.ModuleType('ds41');package.__path__=[str(args.runtime/'serving/ds41')];sys.modules['ds41']=package
    torch.ops.load_library('/opt/ds41-venv/lib/python3.12/site-packages/vllm/_C_stable_libtorch.abi3.so')
    torch.ops.load_library(str(args.runtime/'serving/libds41_swa32.so'))
    old=torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert;new=torch.ops.ds41_swa32.insert
    cases=[]
    for name,n,heads,padded,states in [('normal',129,32,32,32),('mixed_scales',129,32,32,32),('ties_and_zeros',8,8,16,32),('dp_padding',24,32,64,64)]:
        cases.append(check_pair(name,n,heads,padded,states,115200,old,new))
        gc.collect();torch.cuda.empty_cache()
    readers=check_readers(old,new);gc.collect();torch.cuda.empty_cache()
    display=None
    if args.display_library:
        from bench_nvfp4_kv import display_buffer
        display,owner,lib,handle=display_buffer(args.display_library)
    for n in (1,8,24,128,512,2048,3072):
        cases.append(check_pair('timing',n,32,32,32,19008,old,new,display=display,timings=True))
        gc.collect();torch.cuda.empty_cache()
    if display is not None:
        torch.cuda.synchronize();del display,owner;gc.collect();lib.ds41_display_destroy(handle)
    result=dict(status='pass',group32_rope_dtype='bfloat16',group64_rope_dtype='bfloat16',
                group32_state_bytes=592,group64_state_bytes=584,aligned_32_token_page_bytes=19008,
                cases=cases,readers=readers,actual_display_allocation_tested=args.display_library is not None,
                max_allocated_bytes=torch.cuda.max_memory_allocated(),full_model_quality_ab=False,
                source_sha256={n:hashlib.sha256((args.runtime/'serving'/n).read_bytes()).hexdigest() for n in ('libds41_swa32.so','ds41/fused_sparse_attention.py','ds41/online_sparse_attention.py','ds41/online_decode_attention.py','ds41/dcp_overlap/packed.py')})
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(status=result['status'],cases=len(cases),readers=len(readers),max_allocated_bytes=result['max_allocated_bytes'])),flush=True)


if __name__=='__main__':main()
