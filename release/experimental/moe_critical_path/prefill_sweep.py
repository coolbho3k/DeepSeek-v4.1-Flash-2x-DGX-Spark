# SPDX-License-Identifier: AGPL-3.0-only
"""Isolated threshold sweep of unchanged MiaAI-derived prefill kernels.

Called only after the maintenance probe admits the idle GPU and loads the
pinned runtime and distinct weights. This mutates a process-local dispatcher
constant, never serving files, weights or memory bounds. Synthetic inputs are
not an end-to-end quality or speed qualification.
"""
import json
import statistics


def run(dispatcher,experts,output,rank,reference):
    import torch
    import spark_grouped_prefill as grouped
    if (type(dispatcher).__call__.__globals__ is not grouped.__dict__
            or grouped.FAT_MIN!=16 or len(experts)!=384):
        raise ValueError('Unexpected shared dispatcher or expert bank')
    thresholds=(2,4,8,16,32,64)
    flush=torch.zeros(48*2**20,device='cuda',dtype=torch.uint8)
    cases=[]

    def error(actual,expected):
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise ValueError('Nonfinite prefill output')
        diff=actual.float()-expected.float()
        value=dict(nmse=float(diff.square().sum()/expected.float().square().sum().clamp_min(1e-30)),
                   max_absolute=float(diff.abs().max()),bitwise_equal=torch.equal(actual,expected))
        if value['nmse']>2e-5:raise ValueError('Prefill exceeded component numerical bound')
        return value

    try:
        with torch.inference_mode():
            for rows in (128,512,2048):
                for topology in ('balanced','random','skewed'):
                    x=torch.randn((rows,5120),device='cuda',dtype=torch.bfloat16)*.2
                    slots=torch.arange(6,device='cuda')[None,:]
                    if topology=='balanced':
                        ids=torch.arange(rows*6,device='cuda').reshape(rows,6)%384
                    else:
                        starts=torch.randint(384,(rows,1),device='cuda')
                        ids=(starts+slots*53)%384
                        if topology=='skewed':
                            ids[:rows*3//4]=(starts[:rows*3//4]+slots*5)%32
                    weights=torch.rand((rows,6),device='cuda')/6
                    grouped.FAT_MIN=16
                    baseline=dispatcher(experts,x,ids,weights).clone()
                    # Sample actual rows from the entire batch. Reusing the
                    # same experts per token preserves this independent oracle.
                    selected=torch.linspace(0,rows-1,16,device='cuda').long()
                    canonical=reference(experts,x[selected],ids[selected],weights[selected])
                    errors={};canonical_errors={};plans={};samples={t:[] for t in thresholds}
                    for threshold in thresholds:
                        grouped.FAT_MIN=threshold
                        value=dispatcher(experts,x,ids,weights)
                        errors[threshold]=error(value,baseline)
                        canonical_errors[threshold]=error(value[selected],canonical)
                        plan=dispatcher.last_plan
                        plans[threshold]=dict(thin_rows=int(plan['thin_rows']),
                            fat_rows=int(plan['fat_rows']),segments=int(plan['num_segments']))
                        for _ in range(2):dispatcher(experts,x,ids,weights)
                    torch.cuda.synchronize()
                    for iteration in range(15):
                        order=thresholds[iteration%6:]+thresholds[:iteration%6]
                        if iteration%2:order=order[::-1]
                        for threshold in order:
                            grouped.FAT_MIN=threshold
                            flush.add_(1)
                            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            begin.record();dispatcher(experts,x,ids,weights);end.record();end.synchronize()
                            if iteration>=3:samples[threshold].append(begin.elapsed_time(end))
                    case=dict(rows=rows,topology=topology,errors=errors,
                        canonical_sample_errors=canonical_errors,canonical_sample_rows=16,
                        plans=plans,median_ms={t:statistics.median(v) for t,v in samples.items()},samples_ms=samples)
                    cases.append(case)
                    print(json.dumps(dict(stage='prefill_threshold',rows=rows,topology=topology,
                        median_ms=case['median_ms'],worst_nmse=max(v['nmse'] for v in errors.values()))),flush=True)
                    del value,baseline,canonical,x,ids,weights
                    if torch.cuda.max_memory_allocated()>6*2**30:raise ValueError('Component memory budget exceeded')
    finally:
        grouped.FAT_MIN=16
    result=dict(status='prefill_thresholds_component_only',rank=rank,experts=384,
        synthetic_activations=True,synthetic_routing=True,baseline_threshold=16,
        precision_and_kernels_unchanged=True,full_model_speed_measured=False,
        acceptance_measured=False,graph_replay_qualified=False,cases=cases,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    with output.open('x') as stream:json.dump(result,stream,indent=2)

