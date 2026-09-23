# SPDX-License-Identifier: AGPL-3.0-only
"""Maintenance-only packed-projection comparison with real checkpoint weights."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank',type=int,choices=(0,1),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve previous evidence')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('Idle GPU required')
    import torch
    from safetensors import safe_open
    import spark_packed_wo_a as original
    import packed_wo_a_rows as candidate
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.04)
    torch.manual_seed(41922)
    root=Path('/model');index=json.loads((root/'model.safetensors.index.json').read_bytes())['weight_map']
    report=dict(status='running',rank=args.rank,cases=[],real_checkpoint_weights=True,
                synthetic_activations=True,full_model_speed_measured=False,
                sources={str(Path(module.__file__)):hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                         for module in (original,candidate)})
    def save():args.output.write_text(json.dumps(report,indent=2)+'\n')
    flush=torch.zeros(48*2**20,device='cuda',dtype=torch.uint8)
    tr,gemv,_,finish,_=original.kernels()
    def baseline(x,w,s):
        out=torch.empty((len(x),4,1024),device=x.device,dtype=x.dtype)
        partial=torch.empty((16,len(x),4,1024),device=x.device,dtype=torch.float32)
        gemv[(64,len(x),64)](x,w,s,partial,len(x),4,1024,4096,*x.stride(),16,16,256,
                               num_warps=4,enable_fp_fusion=False)
        finish[(tr.cdiv(out.numel(),256),)](partial,out,out.numel(),16,256,num_warps=4)
        return out
    def capture(fn):
        for _ in range(3):fn()
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):value=fn()
        return graph,value
    save()
    try:
        with torch.inference_mode():
            for layer in (0,19,39):
                weights={}
                for field in ('weight','scale'):
                    name=f'layers.{layer}.attn.wo_a.{field}'
                    with safe_open(root/index[name],framework='pt',device='cpu') as handle:
                        value=handle.get_tensor(name)
                        if field=='scale':
                            if value.shape!=(256,128):raise ValueError('Changed 32x32 checkpoint scales')
                            value=value.view(torch.uint8)[args.rank*128:(args.rank+1)*128].repeat_interleave(32,dim=0)
                        else:value=value[args.rank*4096:(args.rank+1)*4096]
                        weights[field]=value.contiguous().cuda()
                w,s=weights['weight'],weights['scale'].view(torch.uint8)
                if w.shape!=(4096,4096) or s.shape!=(4096,128):raise ValueError('Changed TP2 weight shape')
                for rows in (1,2,3,4):
                    for strided in (False,True):
                        storage=torch.randn((rows,4,4352 if strided else 4096),device='cuda',dtype=torch.bfloat16)
                        x=storage[:,:,:4096]
                        functions=dict(baseline=lambda:baseline(x,w,s),shared_rows=lambda:candidate.forward(x,w,s))
                        # Prove the graph-ready baseline executes the installed eager arithmetic.
                        if not torch.equal(functions['baseline'](),original.grouped_projection(x,w,s)):
                            raise ValueError('Probe baseline differs from serving math')
                        graphs={name:capture(fn) for name,fn in functions.items()}
                        for amplitude in (.0625,1.,8.):
                            x.copy_((torch.randn_like(x.float())*amplitude).bfloat16())
                            expected=functions['baseline']()
                            if not torch.isfinite(expected).all():raise ValueError('Nonfinite reference')
                            for name,(graph,value) in graphs.items():
                                value.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
                                if not torch.equal(value,expected):
                                    difference=(value.float()-expected.float())
                                    report['failed_case']=dict(layer=layer,rows=rows,strided=strided,amplitude=amplitude,
                                        variant=name,max_absolute=float(difference.abs().max()),different=int((value!=expected).sum()))
                                    raise ValueError('Changed-input graph differs from serving output')
                        timing={}
                        for cold in (False,True):
                            samples={name:[] for name in graphs}
                            for iteration in range(31):
                                order=('baseline','shared_rows') if iteration%2==0 else ('shared_rows','baseline')
                                for name in order:
                                    if cold:flush.add_(1)
                                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                                    begin.record();graphs[name][0].replay();end.record();end.synchronize()
                                    if iteration>=5:samples[name].append(begin.elapsed_time(end))
                            timing['flushed' if cold else 'warm']={name:statistics.median(values) for name,values in samples.items()}
                        row=dict(layer=layer,rows=rows,strided=strided,bitwise_exact=True,
                                 changed_input_graph=True,poisoned_output=True,median_ms=timing)
                        report['cases'].append(row);save();print(json.dumps(row),flush=True)
                        del graphs,functions,storage,x,expected
                del weights,w,s
            report.update(status='component_passed_not_serving_qualified',peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
