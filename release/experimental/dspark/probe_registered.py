# SPDX-License-Identifier: AGPL-3.0-only
"""Maintenance check of the actual registered draft dispatcher and graph owner."""
import argparse
import gc
import json
from pathlib import Path
import runpy
import subprocess
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank',type=int,choices=(0,1),required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve evidence')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise ValueError('GPU occupied')
    import torch
    from safetensors import safe_open
    torch.set_num_threads(2);torch.manual_seed(419188)
    runpy.run_path('/opt/ds41-serving/serve.py',run_name='registered_draft_probe')
    torch.cuda.set_per_process_memory_fraction(.06)
    import spark_fused_moe as base
    from ds41.exl3_moe import PackedExpert,eager_moe
    from ds41.graph_validation import GraphOwner
    from ds41.dspark_experiment.kernel_integration import sample_bias
    from probe_draft import error
    index=json.loads(Path('/draft-exl3/model.safetensors.index.json').read_bytes())['weight_map']
    report=dict(status='running',rank=a.rank,cases=[],full_model_qualified=False)
    def save():a.output.write_text(json.dumps(report,indent=2)+'\n')
    save()
    with torch.inference_mode():
        for layer in range(3):
            experts={}
            for expert in range(128):
                prefix=f'mtp.{layer}.ffn.experts.{expert}'
                keys=[f'{prefix}.{proj}.{field}' for proj in ('w1','w3','w2') for field in ('trellis','suh','svh','mul1')]
                tensors={}
                for shard in sorted({index[k] for k in keys}):
                    with safe_open('/draft-exl3/'+shard,framework='pt',device='cpu') as f:
                        for key in keys:
                            if index[key]==shard:tensors[key]=f.get_tensor(key).contiguous().cuda()
                experts[expert]=PackedExpert(tensors,prefix,a.rank,2,limit=10.)
            dispatch=base._dispatcher
            for rows in (3,4,5,6,10,15,18,24,30):
                for topology in ('distinct','shared'):
                    x=torch.randn(rows,5120,device='cuda',dtype=torch.bfloat16)*.2
                    ids=torch.arange(rows*3,device='cuda').reshape(rows,3)%128
                    if topology=='shared':ids.remainder_(3)
                    weights=torch.rand(rows,3,device='cuda')/3
                    for _ in range(3):dispatch(experts,x,ids,weights)
                    expected=eager_moe(experts,x,ids,weights)
                    owner=GraphOwner(torch.device('cuda',0));graph=torch.cuda.CUDAGraph()
                    torch.cuda.synchronize()
                    with owner.execution(capture_only=True),torch.cuda.graph(graph):out=dispatch(experts,x,ids,weights)
                    if rows>=5 and dispatch.last_schedule['mode']!='ds41_draft_top3':raise ValueError('Registered draft kernel was not selected')
                    checks=[]
                    for iteration in range(3):
                        x.mul_(-.99);weights.mul_(.97);ids.add_(11).remainder_(128)
                        if iteration==2:ids[:,0]=-1
                        expected=eager_moe(experts,x,ids,weights)
                        for temp in dispatch.workspace.temps:temp.fill_(float('nan'))
                        with owner.execution(capture_only=False):graph.replay()
                        torch.cuda.synchronize();check=error(torch,out,expected);checks.append(check)
                        if check['nmse']>2e-5:raise ValueError('Registered graph differs from canonical experts: '+str(check))
                    owner.wait_before_graph_destruction();graph.reset();owner.release_after_graph_destruction()
                    report['cases'].append(dict(layer=layer,rows=rows,topology=topology,checks=checks));save()
                    print(json.dumps(dict(stage='registered_draft_pass',layer=layer,rows=rows,topology=topology)),flush=True)
                    del graph,owner,out,x,ids,weights,expected
                if torch.cuda.max_memory_allocated()>6*2**30:raise ValueError('Component memory bound exceeded')
            del dispatch.banks[id(experts)]
            del experts,tensors
            gc.collect();torch.cuda.empty_cache()
        # Native no-cache greedy and reduced-vocab paths must call the exact
        # original sampling entry, not the fused full-vocabulary path.
        calls=[]
        base_logits=torch.randn(2,129280,device='cuda',dtype=torch.bfloat16)
        bias=torch.randn_like(base_logits)
        def original(logits,mapping,pos,step):
            calls.append(step);return logits.argmax(-1)
        for cache,scatter in ((None,None),(object(),object())):
            spec=SimpleNamespace(draft_logits=cache,_d2t_scatter_index=scatter,_sample_logits=original)
            actual=sample_bias(spec,base_logits,bias,None,None,2)
            if not torch.equal(actual,(base_logits+bias).argmax(-1)):raise ValueError('Native fallback changed')
        assert calls==[2,2]
    report.update(status='registered_draft_components_passed',fallbacks_preserved=True,
        peak_cuda_bytes=torch.cuda.max_memory_allocated());save()


if __name__=='__main__':main()
