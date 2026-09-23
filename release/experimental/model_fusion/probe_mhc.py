# SPDX-License-Identifier: AGPL-3.0-only
"""Check fused mHC residual/projection boundaries against current kernels."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rank',type=int,choices=(0,1),required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve previous evidence')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise ValueError('Idle GPU required')
    import torch
    from safetensors import safe_open
    from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang
    from ds41 import mhc_decode_prenorm as original
    import post_prenorm as candidate
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.04);torch.manual_seed(41924+args.rank)
    root=Path('/model');index=json.loads((root/'model.safetensors.index.json').read_bytes())['weight_map']
    report=dict(status='running',rank=args.rank,cases=[],real_checkpoint_weights=True,full_model_speed_measured=False)
    flush=torch.zeros(48*2**20,device='cuda',dtype=torch.uint8)
    def save():args.output.write_text(json.dumps(report,indent=2)+'\n')
    def graph_of(fn):
        for _ in range(3):fn()
        torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):value=fn()
        return g,value
    save()
    try:
        with torch.inference_mode():
            for layer in (0,19,39):
                name=f'layers.{layer}.hc_ffn_fn'
                with safe_open(root/index[name],framework='pt',device='cpu') as handle:
                    fn=handle.get_tensor(name).reshape(24,20480).contiguous().cuda()
                for rows in (1,2,3,4):
                    x=torch.randn((rows,5120),device='cuda',dtype=torch.bfloat16)
                    r=torch.randn((rows,4,5120),device='cuda',dtype=torch.bfloat16)
                    post=torch.rand((rows,4,1),device='cuda')*2
                    comb=torch.rand((rows,4,4),device='cuda');comb/=comb.sum(1,keepdim=True)
                    for splits in (4,16):
                        def baseline():
                            residual=mhc_post_tilelang(x,r,post,comb)
                            mixes=torch.empty((splits,rows,24),device='cuda',dtype=torch.float32)
                            squares=torch.empty((splits,rows),device='cuda',dtype=torch.float32)
                            original.forward(residual.reshape(rows,20480),fn,mixes,squares,splits,tile_n=4,warps=8)
                            return residual,mixes,squares
                        functions={'baseline':baseline,'fused_post':lambda:candidate.forward(x,r,post,comb,fn,splits,fused_post=True),
                                   'unfused_post':lambda:candidate.forward(x,r,post,comb,fn,splits,fused_post=False)}
                        graphs={name:graph_of(call) for name,call in functions.items()}
                        exact={name:True for name in functions}
                        errors={name:[] for name in functions}
                        for pattern in ('normal','cancellation','zero'):
                            if pattern=='cancellation':comb[:,:,1::2].mul_(-1);r[:,1].copy_(-r[:,0]);r[:,3].copy_(-r[:,2])
                            if pattern=='zero':x.zero_();r.zero_()
                            expected=baseline()
                            for label,(g,values) in graphs.items():
                                for v in values:v.fill_(float('nan'))
                                g.replay();torch.cuda.synchronize()
                                flags=[torch.equal(v,e) for v,e in zip(values,expected)]
                                exact[label] &= all(flags)
                                errors[label].append(dict(pattern=pattern,exact=flags))
                        # Restore finite nontrivial inputs before interleaved timings.
                        x.normal_();r.normal_()
                        samples={name:[] for name in graphs}
                        names=list(graphs)
                        for iteration in range(25):
                            for name in names[iteration%3:]+names[:iteration%3]:
                                flush.add_(1);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                                a.record();graphs[name][0].replay();b.record();b.synchronize()
                                if iteration>=5:samples[name].append(a.elapsed_time(b))
                        row=dict(layer=layer,rows=rows,splits=splits,exact=exact,boundary_checks=errors,
                                 median_ms={name:statistics.median(v) for name,v in samples.items()})
                        report['cases'].append(row);save();print(json.dumps({k:v for k,v in row.items() if k!='boundary_checks'}),flush=True)
                        del graphs,expected,functions
            report.update(status='component_measured_not_serving_qualified',
                          exact_variants=[name for name in ('fused_post','unfused_post') if all(c['exact'][name] for c in report['cases'])],
                          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:report.update(status='failed',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
