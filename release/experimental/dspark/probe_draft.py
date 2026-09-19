# SPDX-License-Identifier: AGPL-3.0-only
"""Maintenance-only real-drafter top3 MoE and zero-copy KV projection checks.

Runs with idle GPUs, read-only weights, bounded memory and no networking.
Native reference math, changed routes, poisoned scratch and graph timings are
checked separately from full-model performance. No serving qualification here.
"""
import argparse
import gc
import json
from pathlib import Path
import runpy
import statistics
import subprocess
from types import SimpleNamespace


def error(torch,actual,expected):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError('Nonfinite component output')
    d=actual.float()-expected.float()
    return dict(equal=torch.equal(actual,expected),nmse=float(d.square().sum()/expected.float().square().sum().clamp_min(1e-30)),
        max_absolute=float(d.abs().max()))


def measure(torch,functions,mutate,poison):
    graphs={};outputs={}
    for name,fn in functions.items():
        for _ in range(3):fn()
        torch.cuda.synchronize()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):out=fn()
        graphs[name]=g;outputs[name]=out
    mutate();errors={}
    for name,fn in functions.items():
        expected=fn().clone();poison();graphs[name].replay();torch.cuda.synchronize()
        errors[name]=error(torch,outputs[name],expected)
        if errors[name]['nmse']>2e-12:raise ValueError('Changed-input replay disagrees: '+str(errors))
    comparison=error(torch,outputs['candidate'],outputs['reference'])
    if comparison['nmse']>2e-5:raise ValueError('Candidate/reference numerical bound: '+str(comparison))
    flush=torch.zeros(48*2**20,device='cuda',dtype=torch.uint8)
    samples={name:[] for name in functions};names=list(functions)
    for iteration in range(24):
        for name in names[::1 if iteration%2 else -1]:
            flush.add_(1)
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record();graphs[name].replay();end.record();end.synchronize()
            if iteration>=4:samples[name].append(begin.elapsed_time(end))
    for g in graphs.values():g.reset()
    return dict(changed_replay=errors,comparison=comparison,median_ms={k:statistics.median(v) for k,v in samples.items()},
        samples_ms=samples)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank',type=int,choices=(0,1),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--only',choices=('all','projection'),default='all')
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve earlier evidence')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('GPU occupied; this test never stops workloads')
    import torch
    from safetensors import safe_open
    torch.set_num_threads(2);torch.manual_seed(419185)
    runpy.run_path('/opt/ds41-serving/serve.py',run_name='dspark_draft_component')
    import spark_combined_miaai as combined
    combined.register();torch.cuda.set_per_process_memory_fraction(.06)
    import spark_fused_moe as base
    from ds41.exl3_moe import PackedExpert,eager_moe
    from draft_top3 import NativeTop3
    from kv_projection import KVProjection
    report=dict(status='running',rank=a.rank,moe=[],projection=[],full_model_qualified=False,
        synthetic_activations=True,distinct_real_draft_experts_per_layer=128)
    def save():a.output.write_text(json.dumps(report,indent=2)+'\n')
    save()
    index=json.loads(Path('/draft-exl3/model.safetensors.index.json').read_bytes())['weight_map']
    def load(root,index,keys):
        tensors={}
        for shard in sorted({index[k] for k in keys}):
            with safe_open(str(root/shard),framework='pt',device='cpu') as source:
                for k in keys:
                    if index[k]==shard:tensors[k]=source.get_tensor(k).contiguous().cuda()
        return tensors
    with torch.inference_mode():
        for layer in (range(3) if a.only=='all' else ()):
            experts={}
            for expert in range(128):
                prefix=f'mtp.{layer}.ffn.experts.{expert}'
                keys=[f'{prefix}.{proj}.{field}' for proj in ('w1','w3','w2') for field in ('trellis','suh','svh','mul1')]
                tensors=load(Path('/draft-exl3'),index,keys)
                experts[expert]=PackedExpert(tensors,prefix,a.rank,2,limit=10.)
            x=torch.randn(30,5120,device='cuda',dtype=torch.bfloat16)*.2
            ids=torch.arange(90,device='cuda').reshape(30,3)%128
            weights=torch.rand(30,3,device='cuda')/3
            dispatcher=base._dispatcher
            dispatcher(experts,x[:3],ids[:3],weights[:3]);work=dispatcher.workspace
            bank=dispatcher.banks[id(experts)]
            build=json.loads(Path('/candidate/complete.json').read_bytes())
            candidate=NativeTop3(work,Path('/candidate/dspark_draft_top3.so'),build['binary_sha256'])
            def reference(x,ids,weights):
                flat=ids.reshape(-1).long();safe=torch.where((flat>=0)&(flat<384),flat,384)
                mapped=bank.mapping.index_select(0,safe);order=torch.argsort(mapped,stable=True)
                counts=torch.zeros(len(bank.keys)+1,device='cuda',dtype=torch.int64)
                counts.scatter_add_(0,mapped,torch.ones_like(mapped))
                tokens=torch.div(order,3,rounding_mode='floor')
                rw=weights.reshape(-1).index_select(0,order).float().contiguous()
                out=torch.zeros(x.shape,device='cuda',dtype=torch.float32)
                dispatcher.module.forward(x.half().contiguous(),out,counts,tokens,rw,bank.ptrs,work.temps,work.locks)
                return out
            for rows in (1,3,4,5,6,10,12,15,18,20,24,25,30):
                for topology in ('distinct','shared','partial','missing','duplicate','empty','zero_weight'):
                    xi=x[:rows];ii=ids[:rows];ww=weights[:rows]
                    ii.copy_(torch.arange(rows*3,device='cuda').reshape(rows,3)%128);ww.fill_(1/3)
                    if topology=='shared':ii.copy_(torch.arange(3,device='cuda')[None,:].expand(rows,3))
                    elif topology=='partial':ii.copy_((torch.arange(rows,device='cuda')[:,None]+torch.arange(3,device='cuda')[None,:])%128)
                    elif topology=='missing':ii[:,0]=-1;ii[:,1]=128
                    elif topology=='duplicate':ii.zero_()
                    elif topology=='empty':ii.fill_(-1)
                    elif topology=='zero_weight':ww.zero_()
                    actual=candidate(work,bank,xi,ii,ww)
                    canonical=eager_moe(experts,xi,ii,ww)
                    check=error(torch,actual,canonical)
                    if check['nmse']>2e-5:raise ValueError('Draft canonical reference mismatch: '+str(check))
                    def mutate():
                        xi.mul_(-.99);ww.mul_(.97)
                        ii.copy_(torch.where((ii>=0)&(ii<128),(ii+7)%128,ii))
                    def poison():
                        for temp in work.temps:temp.fill_(float('nan'))
                    measured=measure(torch,dict(reference=lambda:reference(xi,ii,ww),
                        candidate=lambda:candidate(work,bank,xi,ii,ww)),mutate,poison)
                    row=dict(layer=layer,rows=rows,topology=topology,canonical=check,**measured)
                    report['moe'].append(row);save()
                    print(json.dumps(dict(stage='draft_top3',layer=layer,rows=rows,topology=topology,
                        median_ms=measured['median_ms'])),flush=True)
            # The bank owns its pointers; free only after all its graphs reset.
            del dispatcher.banks[id(experts)]
            del bank,experts,tensors,actual,canonical,candidate,x,ids,weights,xi,ii,ww
            gc.collect();torch.cuda.empty_cache()
        dense_index=json.loads(Path('/model/draft/model.safetensors.index.json').read_bytes())['weight_map']
        from b12x.gemm import mxfp8_linear
        from vllm.model_executor.kernels.linear.mxfp8.b12x import _apply_b12x_mxfp8_packed_linear
        from ds41.dense_fused_input import wrap_apply
        original=wrap_apply(lambda kernel,layer,x,bias:_apply_b12x_mxfp8_packed_linear(layer,x,bias))
        for layer in range(3):
            prefix=f'mtp.{layer}.attn'
            tensors=load(Path('/model/draft'),dense_index,[f'{prefix}.{p}.{f}' for p in ('wq_a','wkv') for f in ('weight','scale')])
            weight=torch.cat([tensors[f'{prefix}.{p}.weight'] for p in ('wq_a','wkv')])
            scale=torch.cat([tensors[f'{prefix}.{p}.scale'] for p in ('wq_a','wkv')])
            # Native ModelOpt _Mxfp8WeightScaleParam expands each 32-row
            # checkpoint scale block byte-for-byte, not a requantization.
            if tuple(scale.shape)!=(56,160):raise ValueError('Changed checkpoint scale block')
            scale=scale.view(torch.uint8).repeat_interleave(32,dim=0).view(scale.dtype)
            packed=mxfp8_linear.pack_weight(weight,scale)
            linear=SimpleNamespace(b12x_mxfp8_packed_weight=packed)
            attn=SimpleNamespace(q_lora_rank=1280,fused_wqa_wkv=linear)
            projection=KVProjection(attn)
            for rows in (1,3,4,5,6,12,18,24,30,36,128,512,2048):
                x=torch.randn(rows,5120,device='cuda',dtype=torch.bfloat16)*.2
                measured=measure(torch,dict(reference=lambda:original(None,linear,x,None)[:,1280:],
                    candidate=lambda:projection(attn,x)),lambda:x.mul_(-.99),lambda:None)
                row=dict(layer=layer,rows=rows,**measured);report['projection'].append(row);save()
                print(json.dumps(dict(stage='draft_kv_projection',layer=layer,rows=rows,
                    comparison=measured['comparison'],median_ms=measured['median_ms'])),flush=True)
            del projection,attn,linear,packed,tensors,weight,scale,x
            gc.collect();torch.cuda.empty_cache()
    report.update(status='draft_components_passed',peak_cuda_bytes=torch.cuda.max_memory_allocated())
    if report['peak_cuda_bytes']>6*2**30:raise ValueError('Exceeded component allocation bound')
    save()


if __name__=='__main__':main()
