# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded registered-router GPU proof: real K3 weights, graph/scratch reuse.

Six actual experts aliased across384 IDs test routing, not full-layer bandwidth.
No end-to-end speed/accuracy or new quantization qualification is implied.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import runpy
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host-index',type=int,choices=(0,1),required=True)
    p.add_argument('--registration-only',action='store_true')
    a=p.parse_args()
    import torch
    torch.set_num_threads(2)
    # CPU-only startup test does not initialize a CUDA context.
    runpy.run_path('/opt/ds41-serving/serve.py',run_name='cooperative_test_entry')
    import spark_combined_miaai as combined
    descriptor=combined.register()
    assert descriptor['cooperative_moe']
    assert not torch.cuda.is_initialized()
    if a.registration_only:
        print(json.dumps(dict(status='cooperative_cpu_registration_pass',descriptor=descriptor)),flush=True)
        return
    torch.cuda.set_per_process_memory_fraction(.0075)
    from safetensors import safe_open
    from safetensors.torch import load_file
    from ds41.exl3_moe import PackedExpert
    from ds41.cooperative_contract import eligible_shape,selected_shape
    from ds41.cooperative_routes import prepare
    from check_combined_miaai_gpu import NativeGraph
    from check_exl3_prefill_bench import CAPTURE,CAPTURE_SHA,digest,sort_once_moe
    import spark_fused_moe as base
    import spark_fused_moe_async as small
    import spark_grouped_prefill as grouped
    dispatch=base._dispatcher
    assert type(dispatch) is grouped.GroupedDispatcher
    model=Path('/work/artifacts')/('ds41-exl3-3bpw-candidate-v1' if a.host_index==0 else 'ds41-hf-release-v2')
    index=json.loads((model/'model.safetensors.index.json').read_bytes())['weight_map']
    experts={}
    for local,expert in enumerate((0,48,96,144,192,240)):
        prefix=f'layers.0.ffn.experts.{expert}'
        keys=[f'{prefix}.{projection}.{field}' for projection in ('w1','w3','w2')
            for field in ('trellis','suh','svh','mul1')]
        tensors={}
        for name in sorted({index[key] for key in keys}):
            with safe_open(model/name,framework='pt',device='cpu') as source:
                for key in keys:
                    if index[key]==name:tensors[key]=source.get_tensor(key).contiguous().cuda()
        experts[local]=PackedExpert(tensors,prefix,1-a.host_index,2,limit=10.)
        del tensors
    bank={i:experts[i%6] for i in range(384)}
    capture=Path('/work')/CAPTURE
    assert digest(capture)==CAPTURE_SHA
    inputs=load_file(capture)['inputs'].cuda()
    torch.manual_seed(41916)
    flush=torch.zeros(48*2**20,device='cuda',dtype=torch.uint8)
    cases=[];graphs=[];timings=[];preparation=[]

    def error(actual,expected):
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        diff=(actual.float()-expected.float())
        return dict(nmse=(diff.square().sum()/expected.float().square().sum().clamp_min(1e-30)).item(),
            peak_relative=(diff.abs().max()/expected.float().abs().max().clamp_min(1e-30)).item(),
            mismatches=int((actual!=expected).sum()))

    def invoke(candidate,x,ids,weights):
        # Diagnostic A/B selection only, single-threaded and under the real
        # dispatch lock inside __call__. Restore immediately, even on error.
        original=small._ds41_coop_eligible
        small._ds41_coop_eligible=eligible_shape if candidate else lambda *_:False
        try:return dispatch(bank,x,ids,weights)
        finally:small._ds41_coop_eligible=original

    def poison():
        for temp in dispatch.workspace.temps:temp.fill_(float('nan'))

    with torch.inference_mode():
        for rows in range(1,9):
            for dtype in (torch.float16,torch.bfloat16):
                x=inputs[:rows].to(dtype).contiguous()
                ids=torch.arange(rows*6,device='cuda').reshape(rows,6)*7
                weights=torch.rand((rows,6),device='cuda')/6
                for label in ('distinct','overlap','duplicates','mixed_missing','empty'):
                    if label=='distinct':ids.copy_(torch.arange(rows*6,device='cuda').reshape(rows,6)*7)
                    elif label=='overlap':ids.copy_(torch.arange(6,device='cuda')[None,:].expand(rows,6))
                    elif label=='duplicates':ids.zero_()
                    elif label=='mixed_missing':ids[:,::2]=-1;ids[:,1::3]=384
                    else:ids.fill_(-1)
                    expected=invoke(False,x,ids,weights).clone()
                    poison();actual=invoke(True,x,ids,weights).clone()
                    delta=error(actual,expected)
                    assert delta['nmse']<=2e-5,(rows,str(dtype),label,delta)
                    assert dispatch.last_schedule['mode']=='miaai_two_stage_cooperative'
                    # Reenter the original kernel after cooperative scratch
                    # use: catches accidental writes into its completion locks.
                    again=invoke(False,x,ids,weights)
                    assert torch.equal(again,expected),(rows,label,'fallback corrupted')
                    selected=dispatch(bank,x,ids,weights)
                    assert torch.equal(selected,actual if selected_shape(x.shape,ids.shape) else expected)
                    assert (dispatch.last_schedule['mode']=='miaai_two_stage_cooperative')==(rows>=5)
                    cases.append(dict(rows=rows,dtype=str(dtype),label=label,**delta))
                ids.copy_(torch.arange(rows*6,device='cuda').reshape(rows,6)*7)
                # Also compare the canonical expert implementation, separately
                # from the previous serving kernel's rounding behavior.
                canonical=sort_once_moe(torch,bank,x,ids,weights)
                delta=error(invoke(True,x,ids,weights),canonical)
                assert delta['nmse']<=2e-5,('canonical',rows,str(dtype),delta)
                cases.append(dict(rows=rows,dtype=str(dtype),label='canonical',**delta))
                native=dispatch.workspace._ds41_cooperative
                mapping=dispatch.banks[id(bank)].mapping
                # Both ID dtypes, strided tensors, out-of-range sentinels and
                # exact CPU-equivalent FP16 conversion/counter initialization.
                for id_dtype in (torch.int32,torch.int64):
                    ix=torch.randn((rows,10240),device='cuda',dtype=dtype)[:,::2]
                    ii=torch.randint(-4,389,(rows,12),device='cuda',dtype=id_dtype)[:,::2]
                    iw=torch.rand((rows,12),device='cuda')[:,::2]
                    poison()
                    hx,local,rw=prepare(ix,ii,iw,mapping,native.scratch[-1])
                    safe=torch.where((ii>=0)&(ii<384),ii,384).long()
                    assert torch.equal(hx,ix.half())
                    assert torch.equal(local,mapping[safe])
                    assert torch.equal(rw,iw.half())
                    assert not torch.count_nonzero(native.scratch[-1])
                    preparation.append(dict(rows=rows,dtype=str(dtype),ids_dtype=str(id_dtype),exact=True))
                graph=NativeGraph(lambda:invoke(True,x,ids,weights),tokens=rows)
                streams=[torch.cuda.current_stream(),torch.cuda.Stream()]
                for step in range(4):
                    torch.cuda.synchronize()
                    with torch.cuda.stream(streams[step%2]):
                        ids.copy_(torch.randint(-1,390,ids.shape,device='cuda'))
                        if step%2:ids.zero_()
                        x.mul_(-.9);weights.mul_(.97)
                        eager=invoke(True,x,ids,weights).clone()
                        invoke(False,x,ids,weights) # overwrite shared scratch
                        poison()
                        replay=graph.replay().clone()
                        assert torch.equal(replay,eager),(rows,str(dtype),step,error(replay,eager))
                graph.close()
                graphs.append(dict(rows=rows,dtype=str(dtype),exact_replays=4,streams=2,
                    changed_route_topology=True,fallback_and_poison_between_replays=True))
                if dtype==torch.bfloat16:
                    for topology in ('distinct','shared_six','partial_overlap'):
                        if topology=='distinct':values=torch.arange(rows*6,device='cuda').reshape(rows,6)*7
                        elif topology=='shared_six':values=torch.arange(6,device='cuda')[None,:].expand(rows,6)
                        else:values=torch.arange(rows,device='cuda')[:,None]*3+torch.arange(6,device='cuda')[None,:]
                        ids.copy_(values)
                        samples={False:[],True:[]}
                        gs={mode:NativeGraph(lambda m=mode:invoke(m,x,ids,weights),tokens=rows) for mode in (False,True)}
                        for iteration in range(21):
                            for mode in ((False,True) if iteration%2 else (True,False)):
                                flush.add_(1)
                                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                                start.record();gs[mode].replay();end.record();end.synchronize()
                                samples[mode].append(start.elapsed_time(end))
                        for g in gs.values():g.close()
                        timings.append(dict(rows=rows,topology=topology,old_ms=statistics.median(samples[False]),
                            cooperative_ms=statistics.median(samples[True]),l2_flush_bytes=flush.numel(),
                            full_layer_bandwidth=False))
                torch.cuda.synchronize()
                del graph,canonical,expected,again,actual,eager,replay,hx,local,rw,ix,ii,iw,safe
                gc.collect()
                print(json.dumps(dict(stage='cooperative_rows_pass',rows=rows,dtype=str(dtype))),flush=True)
        # Explicit unsupported-row/top-k fallback, including real grouped prefill.
        fallbacks=[]
        for rows,topk in ((1,3),(9,6),(24,6)):
            x=inputs[:rows].bfloat16().contiguous()
            ids=torch.arange(rows*topk,device='cuda').reshape(rows,topk)%384
            weights=torch.rand(ids.shape,device='cuda')/topk
            expected=invoke(False,x,ids,weights).clone()
            actual=invoke(True,x,ids,weights)
            assert torch.equal(actual,expected) and not eligible_shape(x.shape,ids.shape)
            fallbacks.append(dict(rows=rows,topk=topk,exact=True))
        torch.cuda.synchronize()
        peak=torch.cuda.max_memory_allocated()
        assert peak<=512*2**20,peak
    result=dict(status='cooperative_component_gpu_pass',host_index=a.host_index,tp_rank=1-a.host_index,
        descriptor=descriptor,cases=cases,route_preparation=preparation,graphs=graphs,timings=timings,
        fallbacks=fallbacks,resources=dispatch.workspace._ds41_cooperative.resources,
        hybrid_selection_rows=[5,6,7,8],hybrid_selection_eager_verified=True,
        persistent_workspace_bytes=dispatch.workspace.bytes,extra_persistent_scratch_bytes=0,
        actual_weight_experts=6,aliased_expert_ids=384,calibration_capture_sha256=CAPTURE_SHA,
        source_model=str(model),source_index_sha256=digest(model/'model.safetensors.index.json'),
        peak_allocated_bytes=peak,serving_quality_qualified=False,end_to_end_performance_measured=False)
    with Path('/results/complete.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(dict(status=result['status'],host_index=a.host_index,
        max_nmse=max(c['nmse'] for c in cases),timings=timings)),flush=True)


if __name__=='__main__':main()
