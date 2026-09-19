# SPDX-License-Identifier: AGPL-3.0-only
"""Maintenance-only GPU comparison of the three existing MoE geometries.

Uses real distinct canonical experts (not an aliased six-expert bandwidth
fixture), the existing dispatcher bank/scratch, and the unchanged native
binary. No networking, publication, server lifecycle or quantization actions.
Run in an explicitly bounded test container with idle GPUs and read-only
model/serving mounts. Not suitable beside a resident model server.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import statistics
import subprocess
import types

from geometry import transform


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--serving',type=Path,required=True)
    p.add_argument('--rank',type=int,choices=(0,1),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--maintenance',action='store_true',required=True)
    p.add_argument('--experts',type=int,choices=(24,144),default=144)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve existing experiment results')
    # Admission must precede torch import/context creation. The outer launcher
    # also checks both recorded serving workers; never trust the flag alone.
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
            '--format=csv,noheader'],text=True,timeout=15).strip():
        raise ValueError('GPU is occupied; this probe never stops workloads')
    import torch
    from safetensors import safe_open
    torch.set_num_threads(2)
    runpy.run_path(str(a.serving/'serve.py'),run_name='moe_geometry_probe_entry')
    import spark_combined_miaai as combined
    combined.register()
    torch.cuda.set_per_process_memory_fraction(.04)
    import spark_fused_moe as base
    from ds41.exl3_moe import PackedExpert, eager_moe
    torch.manual_seed(41918)
    index=json.loads((a.model/'model.safetensors.index.json').read_bytes())['weight_map']
    experts={}
    for expert in range(a.experts):
        prefix=f'layers.0.ffn.experts.{expert}'
        keys=[f'{prefix}.{proj}.{field}' for proj in ('w1','w3','w2')
              for field in ('trellis','suh','svh','mul1')]
        tensors={}
        for name in sorted({index[key] for key in keys}):
            with safe_open(a.model/name,framework='pt',device='cpu') as source:
                for key in keys:
                    if index[key]==name:tensors[key]=source.get_tensor(key).contiguous().cuda()
        experts[expert]=PackedExpert(tensors,prefix,a.rank,2,limit=10.)
        del tensors
    x=torch.randn((24,5120),device='cuda',dtype=torch.bfloat16)*.2
    ids=torch.arange(144,device='cuda').reshape(24,6)%a.experts
    weights=torch.rand((24,6),device='cuda')/6
    dispatcher=base._dispatcher
    with torch.inference_mode():dispatcher(experts,x[:4],ids[:4],weights[:4])
    work=dispatcher.workspace;bank=dispatcher.banks[id(experts)]
    binary=a.serving/'cooperative_moe.so'
    receipt=json.loads((a.serving/'cooperative-native.json').read_bytes())
    source=(a.serving/'ds41/cooperative_moe.py').read_bytes()
    native={}
    for geometry in (0,1,2):
        module=types.ModuleType(f'ds41.geometry_{geometry}')
        module.__package__='ds41'
        exec(compile(transform(source,geometry),f'<geometry-{geometry}>','exec'),module.__dict__)
        native[geometry]=module.Native(work,binary,receipt['binary_sha256'])
    flush=torch.zeros(48*2**20,device='cuda',dtype=torch.uint8)
    results=[]
    def delta(actual,expected):
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise ValueError('Nonfinite geometry output')
        diff=actual.float()-expected.float()
        return dict(bitwise_equal=torch.equal(actual,expected),
            nmse=float(diff.square().sum()/expected.float().square().sum().clamp_min(1e-30)),
            max_absolute=float(diff.abs().max()))
    with torch.inference_mode():
        for rows in (1,4,6,8,12,16,24):
            for topology in ('distinct','shared','partial','missing','duplicates','empty'):
                xi=x[:rows];ii=ids[:rows];ww=weights[:rows]
                if topology=='distinct':ii.copy_(torch.arange(rows*6,device='cuda').reshape(rows,6)%a.experts)
                elif topology=='shared':ii.copy_(torch.arange(6,device='cuda')[None,:].expand(rows,6))
                elif topology=='partial':ii.copy_((torch.arange(rows,device='cuda')[:,None]*3+torch.arange(6,device='cuda')[None,:])%a.experts)
                elif topology=='missing':ii[:,::2]=-1;ii[:,1::3]=384
                elif topology=='duplicates':ii.zero_()
                else:ii.fill_(-1)
                torch.cuda.synchronize()
                expected=native[1](bank,xi,ii,ww).clone()
                canonical=eager_moe(experts,xi,ii,ww)
                graphs={};outputs={};errors={};canonical_errors={}
                for geometry in (1,0,2):
                    eager=native[geometry](bank,xi,ii,ww)
                    errors[geometry]=delta(eager,expected)
                    canonical_errors[geometry]=delta(eager,canonical)
                    if max(errors[geometry]['nmse'],canonical_errors[geometry]['nmse'])>2e-5:
                        raise ValueError('Geometry exceeded the existing component numerical bound')
                    # Prewarm allocations and kernel resources before capture.
                    for _ in range(3):native[geometry](bank,xi,ii,ww)
                    torch.cuda.synchronize()
                    g=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):out=native[geometry](bank,xi,ii,ww)
                    graphs[geometry]=g;outputs[geometry]=out
                    g.replay();torch.cuda.synchronize()
                    if not torch.equal(out,eager):raise ValueError('Geometry graph replay differs')
                # Changed activations, weights AND expert addresses: captured
                # graphs must not retain old routes. Preserve invalid sentinels.
                xi.mul_(-.99);ww.mul_(.97)
                ii.copy_(torch.where((ii>=0)&(ii<a.experts),(ii+7)%a.experts,ii))
                canonical=eager_moe(experts,xi,ii,ww)
                replay_errors={}
                for geometry in (0,1,2):
                    eager=native[geometry](bank,xi,ii,ww).clone()
                    # No kernel may depend on previous shared scratch values.
                    for temp in work.temps:temp.fill_(float('nan'))
                    graphs[geometry].replay();torch.cuda.synchronize()
                    if not torch.equal(outputs[geometry],eager):raise ValueError('Stale graph input/output')
                    replay_errors[geometry]=delta(outputs[geometry],canonical)
                    if replay_errors[geometry]['nmse']>2e-5:
                        raise ValueError('Changed-route graph differs from canonical experts')
                samples={g:[] for g in (0,1,2)}
                for iteration in range(33):
                    order=(1,0,2) if iteration%3==0 else (2,1,0) if iteration%3==1 else (0,2,1)
                    for geometry in order:
                        flush.add_(1)
                        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        begin.record();graphs[geometry].replay();end.record();end.synchronize()
                        if iteration>=3:samples[geometry].append(begin.elapsed_time(end))
                results.append(dict(rows=rows,topology=topology,errors=errors,
                    canonical_errors=canonical_errors,replay_canonical_errors=replay_errors,
                    changed_routes=True,scratch_poisoned_before_replay=True,
                    median_ms={g:statistics.median(v) for g,v in samples.items()},samples_ms=samples))
                del graphs,outputs
                torch.cuda.synchronize()
                if torch.cuda.max_memory_allocated()>4*2**30:raise ValueError('Component memory budget exceeded')
                print(json.dumps({k:v for k,v in results[-1].items() if k!='samples_ms'}),flush=True)
    report=dict(status='geometry_component_measured_not_serving_qualified',rank=a.rank,
        actual_distinct_experts=a.experts,aliasing=False,layer=0,
        source_adapter_sha256=hashlib.sha256(source).hexdigest(),
        binary_sha256=receipt['binary_sha256'],resources={g:n.resources for g,n in native.items()},
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),results=results,
        acceptance_measured=False,full_model_speed_measured=False,canonical_reference_compared=True)
    with a.output.open('x') as out:json.dump(report,out,indent=2)


if __name__=='__main__':main()
