# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded maintenance probe of two full-vocabulary Markov fusion variants."""
import argparse
import json
from pathlib import Path
import runpy
import statistics
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank',type=int,choices=(0,1),required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve results')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise ValueError('GPU occupied')
    import torch
    from safetensors import safe_open
    torch.set_num_threads(2);torch.manual_seed(419187)
    runpy.run_path('/opt/ds41-serving/serve.py',run_name='dspark_markov_probe')
    torch.cuda.set_per_process_memory_fraction(.03)
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
    from markov_sampling import sample
    index=json.loads(Path('/model/draft/model.safetensors.index.json').read_bytes())['weight_map']
    keys=('mtp.2.markov_head.embed.weight','mtp.2.markov_head.head.weight');weights={}
    for key in keys:
        with safe_open('/model/draft/'+index[key],framework='pt',device='cpu') as f:weights[key]=f.get_tensor(key).cuda()
    embed_table,head=(weights[k] for k in keys)
    assert head.shape==(129280,256) and head.dtype==torch.bfloat16
    report=dict(status='running',rank=a.rank,cases=[],full_model_qualified=False)
    def save():a.output.write_text(json.dumps(report,indent=2)+'\n')
    save()
    with torch.inference_mode():
        for rows in (1,2,3,4,5,6):
            for temp in (0.,.6,1.,1.5):
                for fp64 in (False,True):
                    prev=torch.arange(rows,device='cuda')*7919+17
                    # Strided base logits exactly like base_logits[:, step].
                    base=torch.randn(rows,5,129280,device='cuda',dtype=torch.bfloat16)[:,2]
                    base.mul_(2)
                    mapping=torch.arange(rows,device='cuda',dtype=torch.int32).flip(0)
                    temperatures=torch.full((rows,),temp,device='cuda');seeds=torch.arange(rows,device='cuda',dtype=torch.int64)*1729+41
                    positions=torch.arange(rows,device='cuda',dtype=torch.int64)*193+524287
                    column=torch.tensor(2,device='cuda',dtype=torch.int32)
                    caches={n:torch.full((rows,5,129280),float('nan'),device='cuda',dtype=torch.bfloat16) for n in ('reference','add','head')}
                    def execute(name):
                        embedding=torch.nn.functional.embedding(prev,embed_table)
                        if name=='head':return sample(base,None,mapping,temperatures,seeds,positions,caches[name],column,
                            embedding=embedding,head=head,use_fp64=fp64)
                        bias=torch.nn.functional.linear(embedding,head)
                        if name=='add':return sample(base,bias,mapping,temperatures,seeds,positions,caches[name],column,use_fp64=fp64)
                        return gumbel_sample(base+bias,mapping,temperatures,seeds,positions,apply_temperature=True,is_drafting=True,
                            logits_cache=caches[name],logits_cache_col=column,use_fp64=fp64)
                    graphs={};outputs={}
                    for name in caches:
                        execute(name);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):out=execute(name)
                        graphs[name]=g;outputs[name]=out
                    checks=[]
                    for iteration in range(3):
                        prev.add_(11);positions.add_(1);base.mul_(-.99)
                        for name,g in graphs.items():
                            caches[name].fill_(float('nan'));g.replay()
                        torch.cuda.synchronize()
                        ref=caches['reference'][:,2];add=caches['add'][:,2];fused=caches['head'][:,2]
                        if not torch.equal(ref,add) or not torch.equal(outputs['reference'],outputs['add']):raise ValueError('Addition fusion changes logits or RNG')
                        # Verify actual fused proposal cache reproduces its draw,
                        # regardless of different parallel FP32 head reduction.
                        replay=gumbel_sample(fused[mapping.long()],mapping,temperatures,seeds,positions,
                            apply_temperature=True,is_drafting=True,use_fp64=fp64)
                        if not torch.equal(replay,outputs['head']):raise ValueError('Fused head cache does not reproduce proposal')
                        nmse=float((fused.float()-ref.float()).square().sum()/ref.float().square().sum().clamp_min(1e-30))
                        checks.append(dict(head_cache_equal=torch.equal(ref,fused),head_cache_nmse=nmse,
                            head_tokens_equal=torch.equal(outputs['reference'],outputs['head']),proposal_reproduced=True))
                        if not torch.isfinite(fused).all() or nmse>2e-5:raise ValueError('Fused head accuracy bound exceeded')
                    samples={n:[] for n in caches};flush=torch.zeros(48*2**20,device='cuda',dtype=torch.uint8)
                    for iteration in range(24):
                        order=('reference','add','head') if iteration%2 else ('head','add','reference')
                        for name in order:
                            flush.add_(1);begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            begin.record();graphs[name].replay();end.record();end.synchronize()
                            if iteration>=4:samples[name].append(begin.elapsed_time(end))
                    for g in graphs.values():g.reset()
                    row=dict(rows=rows,temperature=temp,fp64=fp64,checks=checks,median_ms={k:statistics.median(v) for k,v in samples.items()},samples_ms=samples)
                    report['cases'].append(row);save();print(json.dumps({k:v for k,v in row.items() if k not in ('samples_ms','checks')}),flush=True)
    report.update(status='markov_components_passed',peak_cuda_bytes=torch.cuda.max_memory_allocated());save()


if __name__=='__main__':main()
