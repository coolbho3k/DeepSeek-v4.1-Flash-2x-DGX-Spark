# SPDX-License-Identifier: AGPL-3.0-only
"""Compare fused/bounded gathers to the shipped native arithmetic on idle GPUs."""
import argparse
import ctypes as C
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
    if args.output.exists():raise ValueError('Preserve existing results')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise ValueError('Idle GPU required')
    import torch
    import spark_grouped_prefill as original
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.04);torch.manual_seed(41923+args.rank)
    old=original.load_kernel('/opt/ds41-serving/ds41_miaai_fat_moe_v1.so')
    root=Path('/candidate');receipt=json.loads((root/'complete.json').read_bytes());binary=root/'dual_gather.so'
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=receipt['binary_sha256']:raise ValueError('Changed native candidate')
    native=C.CDLL(str(binary));native.ds41_dual_gather_abi.restype=C.c_int
    if native.ds41_dual_gather_abi()!=1:raise ValueError('Unexpected native ABI')
    launch=native.ds41_dual_gather;launch.restype=C.c_int
    launch.argtypes=[C.POINTER(C.c_void_p),C.c_int,C.c_int,C.c_void_p]
    native.ds41_dual_gather_info.argtypes=[C.POINTER(C.c_int)];info=(C.c_int*4)()
    if native.ds41_dual_gather_info(info):raise ValueError('Resource query failed')
    report=dict(status='running',rank=args.rank,cases=[],resources=list(info),binary_sha256=receipt['binary_sha256'],
                synthetic_activations_and_scales=True,full_model_speed_measured=False)
    def save():args.output.write_text(json.dumps(report,indent=2)+'\n')
    width,cap=5120,12288
    scales=[(torch.rand((384,width),device='cuda')*4-2).half() for _ in range(2)]
    tables=[torch.tensor([t[i].data_ptr() for i in range(384)],device='cuda',dtype=torch.int64) for t in scales]
    experts=torch.randint(0,384,(cap,),device='cuda',dtype=torch.int32)
    live=torch.zeros(1,device='cuda',dtype=torch.int32)
    flush=torch.zeros(48*2**20,device='cuda',dtype=torch.uint8)
    outputs={name:[torch.empty((cap,width),device='cuda',dtype=torch.float16) for _ in range(2)]
             for name in ('baseline','bounded_pair','dual_full','dual_bounded')}
    save()
    try:
        with torch.inference_mode():
            for rows in (1,8,24,128,512,2048):
                x=torch.randn((rows,width),device='cuda',dtype=torch.float16)*.2
                tokens=torch.arange(cap,device='cuda',dtype=torch.int64)%rows
                bound=rows*6
                def invoke(name):
                    out=outputs[name]
                    if name in ('baseline','bounded_pair'):
                        size=cap if name=='baseline' else bound
                        for i in range(2):old.gather(x,tokens,experts,tables[i],out[i][:size],live)
                    else:
                        pointers=(C.c_void_p*8)(*[t.data_ptr() for t in (x,tokens,experts,*tables,*out,live)])
                        status=launch(pointers,cap if name=='dual_full' else bound,width,C.c_void_p(torch.cuda.current_stream().cuda_stream))
                        if status:raise RuntimeError('Native dual-gather launch failed: '+str(status))
                graphs={}
                for name in outputs:
                    invoke(name);torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):invoke(name)
                    graphs[name]=graph
                for count in sorted({0,1,max(1,bound//4),bound}):
                    live.fill_(count);x.mul_(-.97);experts.add_(7).remainder_(384)
                    for out in outputs.values():
                        for tensor in out:tensor.fill_(17.)
                    for graph in graphs.values():graph.replay()
                    torch.cuda.synchronize()
                    for name,out in outputs.items():
                        for value,expected in zip(out,outputs['baseline']):
                            if not torch.equal(value.view(torch.int16),expected.view(torch.int16)):
                                raise ValueError(f'Byte/canary mismatch: {rows}/{count}/{name}')
                        if any(not torch.all(t[count:]==17.) for t in out):raise ValueError('Inactive row overwritten')
                    samples={name:[] for name in graphs};names=list(graphs)
                    for iteration in range(25):
                        order=names[iteration%4:]+names[:iteration%4]
                        for name in order:
                            flush.add_(1)
                            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            a.record();graphs[name].replay();b.record();b.synchronize()
                            if iteration>=5:samples[name].append(a.elapsed_time(b))
                    row=dict(input_rows=rows,max_routed_rows=bound,active_rows=count,bitwise_exact=True,
                             inactive_canaries_preserved=True,changed_input_graph=True,
                             median_ms={name:statistics.median(v) for name,v in samples.items()})
                    report['cases'].append(row);save();print(json.dumps(row),flush=True)
                del graphs,x,tokens
            report.update(status='component_passed_not_serving_qualified',peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:report.update(status='failed',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
