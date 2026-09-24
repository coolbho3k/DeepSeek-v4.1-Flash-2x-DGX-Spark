# SPDX-License-Identifier: AGPL-3.0-only
"""Compare a native decode candidate against the unchanged cooperative binary.

Uses full FP32 native outputs, not just final BF16 rounding, with real distinct
weights, sparse banks, edge routes, changed-input replay, poisoned scratch and
interleaved cache-flushed timings. Called only by the idle-GPU maintenance
probe. Passing does not establish serving acceptance or end-to-end speed.
"""
import ctypes
import hashlib
import json
import statistics
import sys
import types


def run(work,bank,experts,serving,candidate,output,rank,reference):
    import torch
    import spark_fused_moe as base
    original=json.loads((serving/'cooperative-native.json').read_bytes())
    built=json.loads((candidate/'complete.json').read_bytes())
    source=(serving/'ds41/cooperative_moe.py').read_bytes()
    if hashlib.sha256(source).hexdigest()!='22edf8a5aed2194ac0fd2050a9fd27ffea48dd82de021c40f68fd17109ec6bc5':
        raise ValueError('Changed native adapter')
    text=source.decode()
    if text.count('return out.to(x.dtype)')!=1:raise ValueError('Changed output boundary')
    if built['experiment']==102:
        if text.count('info[i] != 512')!=1:raise ValueError('Changed thread-count admission')
        text=text.replace('info[i] != 512','info[i] not in (512, 1024)')
    # Observe a stricter boundary than production: before BF16 output cast.
    module=types.ModuleType('ds41.native_component_fp32');module.__package__='ds41'
    exec(compile(text.replace('return out.to(x.dtype)','return out'),'<native-component-fp32>','exec'),module.__dict__)
    baseline=module.Native(work,serving/'cooperative_moe.so',original['binary_sha256'])
    candidate_module=module
    capacity=built['experiment']==401
    if capacity:
        import dspark_contracts as contracts
        routes_source=(serving/'ds41/cooperative_routes.py').read_bytes()
        if hashlib.sha256(routes_source).hexdigest()!='ae91ace2d4310f9409ade1367b341cadd90bc28b152e355831144f9b09dc7be7':
            raise ValueError('Changed route preparation source')
        if routes_source.count(b'ctr < 2547')!=1:raise ValueError('Changed counter reset boundary')
        route_name='ds41.capacity_component_routes'
        routes=types.ModuleType(route_name);routes.__package__='ds41'
        # Triton resolves source by filename. Use the pinned copied source,
        # with a distinct real file for this component-only specialization.
        route_file=output.with_name('capacity_component_routes.py')
        with route_file.open('xb') as f:f.write(routes_source.replace(b'ctr < 2547',b'ctr < 3819'))
        routes.__file__=str(route_file);sys.modules[route_name]=routes
        exec(compile(route_file.read_bytes(),str(route_file),'exec'),routes.__dict__)
        candidate_module=types.ModuleType('ds41.capacity_component_fp32');candidate_module.__package__='ds41'
        extended=text.replace('(48, 2547, 344)','(48, 3819, 344)').replace(
            'from .cooperative_routes import prepare','from .capacity_component_routes import prepare')
        exec(compile(extended.replace('return out.to(x.dtype)','return out'),'<capacity-component-fp32>','exec'),candidate_module.__dict__)
        candidate_module.LAYOUT=contracts.LAYOUT
        candidate_module.CAPACITIES=contracts.CAPACITIES
        candidate_module.intervals=contracts.intervals
        candidate_module.eligible_shape=lambda x,ids: len(x)==2 and 1<=x[0]<=36 and x[1]==5120 and tuple(ids)==(x[0],6)
        class ChunkedReference:
            resources=baseline.resources
            def __call__(self,selected_bank,x,ids,weights):
                if len(x)<=24:return baseline(selected_bank,x,ids,weights)
                return torch.cat([baseline(selected_bank,x[start:start+24],ids[start:start+24],weights[start:start+24])
                    for start in range(0,len(x),24)],dim=0)
        reference_native=ChunkedReference()
    else:reference_native=baseline
    variants={'baseline':reference_native,
              'candidate':candidate_module.Native(work,candidate/'cooperative_moe.so',built['binary_sha256'])}
    variants['candidate'].library.goal50_coop_experiment.restype=ctypes.c_int
    if variants['candidate'].library.goal50_coop_experiment()!=built['experiment']:
        raise ValueError('Changed native experiment marker')
    sparse={key:expert for key,expert in experts.items() if key%2==0}
    banks={'full':(bank,experts),'sparse':(base.Bank(sparse,work.device),sparse)}
    flush=torch.zeros(48*2**20,device=work.device,dtype=torch.uint8)
    report=dict(status='running',rank=rank,experiment=built['experiment'],variant=built['variant'],
        binary_sha256=built['binary_sha256'],baseline_binary_sha256=original['binary_sha256'],
        resources={name:value.resources for name,value in variants.items()},cases=[],
        distinct_weight_bank=True,synthetic_activations=True,synthetic_routing=True,
        full_model_speed_measured=False,acceptance_measured=False,observation_dtype='float32')

    def save():
        # This is a fresh experiment report, checkpointed after each case so
        # a failure preserves its exact completed scope and active case.
        temporary=output.with_suffix('.json.tmp')
        with temporary.open('w') as stream:json.dump(report,stream,indent=2)
        temporary.replace(output)

    def delta(actual,expected):
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():raise ValueError('Nonfinite native output')
        difference=actual.float()-expected.float()
        return dict(bitwise_equal=torch.equal(actual,expected),
            nmse=float(difference.square().sum()/expected.float().square().sum().clamp_min(1e-30)),
            max_absolute=float(difference.abs().max()))

    save()
    try:
        with torch.inference_mode():
            for bank_name,(selected_bank,selected_experts) in banks.items():
                row_cases=(1,2,3,4,5,6,8,10,12,15,16,18,20,24,25,30,36) if capacity else (1,2,3,4,6,8,12,16,24)
                for rows in row_cases:
                    topologies=('distinct','shared','partial','missing','duplicates','empty','zero_weights')
                    if bank_name=='sparse' and rows not in ((1,4,24,30,36) if capacity else (1,4,24)):continue
                    for topology in topologies:
                        amplitude=2.0 if topology=='duplicates' else .2
                        x=torch.randn((rows,5120),device=work.device,dtype=torch.bfloat16)*amplitude
                        ids=torch.arange(rows*6,device=work.device).reshape(rows,6)%384
                        weights=torch.rand((rows,6),device=work.device)/6
                        if topology=='shared':ids.copy_(torch.arange(6,device=work.device)[None,:].expand(rows,6))
                        elif topology=='partial':ids.copy_((torch.arange(rows,device=work.device)[:,None]*3+torch.arange(6,device=work.device)[None,:])%384)
                        elif topology=='missing':ids[:,::2]=-1;ids[:,1::3]=384
                        elif topology=='duplicates':ids.zero_()
                        elif topology=='empty':ids.fill_(-1)
                        elif topology=='zero_weights':weights.zero_()
                        report['active_case']=dict(bank=bank_name,rows=rows,topology=topology,amplitude=amplitude)
                        save()
                        expected=variants['baseline'](selected_bank,x,ids,weights).clone()
                        actual=variants['candidate'](selected_bank,x,ids,weights)
                        error=delta(actual,expected)
                        if not error['bitwise_equal']:
                            report['failed_comparison']=error
                            raise ValueError('Candidate differs at FP32 output boundary')
                        canonical=reference(selected_experts,x,ids,weights)
                        canonical_error=delta(actual.to(x.dtype),canonical)
                        if amplitude==.2 and canonical_error['nmse']>2e-5:
                            raise ValueError('Canonical numerical bound exceeded')
                        graphs={};values={}
                        for name,native in variants.items():
                            for _ in range(3):native(selected_bank,x,ids,weights)
                            torch.cuda.synchronize()
                            graph=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):value=native(selected_bank,x,ids,weights)
                            graphs[name]=graph;values[name]=value
                            graph.replay();torch.cuda.synchronize()
                            if not torch.equal(value,expected):raise ValueError('Initial graph replay differs')
                        x.mul_(-.97);weights.mul_(.93)
                        ids.copy_(torch.where((ids>=0)&(ids<384),(ids+7)%384,ids))
                        changed_expected=variants['baseline'](selected_bank,x,ids,weights).clone()
                        for name,graph in graphs.items():
                            for temp in work.temps:temp.fill_(float('nan'))
                            graph.replay();torch.cuda.synchronize()
                            if not torch.equal(values[name],changed_expected):raise ValueError('Changed-input poisoned replay differs: '+name)
                        samples={name:[] for name in variants}
                        for iteration in range(23):
                            order=('baseline','candidate') if iteration%2==0 else ('candidate','baseline')
                            for name in order:
                                flush.add_(1)
                                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                                begin.record();graphs[name].replay();end.record();end.synchronize()
                                if iteration>=3:samples[name].append(begin.elapsed_time(end))
                        result=dict(**report['active_case'],comparison=error,canonical=canonical_error,
                            changed_input_graph_exact=True,scratch_poisoned=True,samples_ms=samples,
                            reference_split_at_24_rows=bool(capacity and rows>24),
                            median_ms={name:statistics.median(v) for name,v in samples.items()})
                        report['cases'].append(result);save()
                        print(json.dumps({k:v for k,v in result.items() if k!='samples_ms'}),flush=True)
                        # One trace proves that the candidate really executes
                        # the new kernel, rather than accidentally timing a fallback.
                        if bank_name=='full' and rows==4 and topology=='distinct':
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA]) as profile:
                                graphs['candidate'].replay();torch.cuda.synchronize()
                            trace=output.with_name('native-candidate-trace.json')
                            if trace.exists():raise ValueError('Preserve existing trace')
                            profile.export_chrome_trace(str(trace))
                            events=json.loads(trace.read_bytes())['traceEvents']
                            kernels=[e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X']
                            if built['experiment'] in (101,102) and not any('fused_a_kernel' in e['name'] for e in kernels):
                                raise ValueError('Candidate did not execute fused gate/up')
                            if built['experiment']==201 and not any('persistent_kernel' in e['name'] for e in kernels):
                                raise ValueError('Candidate did not execute persistent pipeline')
                            if built['experiment']==301 and not all(any(name in e['name'] for e in kernels)
                                    for name in ('coop_dq_a_kernel','coop_dq_b_kernel')):
                                raise ValueError('Candidate did not execute dequantization pipeline')
                            report['kernel_trace']=[dict(name=e['name'],duration_us=e['dur']) for e in kernels]
                            del profile,events,kernels
                        del graphs,values,x,ids,weights,actual,expected,canonical,changed_expected
                        torch.cuda.synchronize()
                        if torch.cuda.max_memory_allocated()>6*2**30:raise ValueError('Native component memory budget exceeded')
        report.update(status='native_component_passed_not_serving_qualified',
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        report.pop('active_case',None)
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:save()
