"""Exercise the fresh AOT MXFP8 library through native vLLM on one idle GPU.

Synthetic matrices only; no model/engrams/distributed groups. Auxiliary warm
caches may be reused, but FlashInfer JIT is disabled and the tested GEMM must
be mapped from the exact read-only precompiled artifact.
"""
import hashlib
import json
import os
from pathlib import Path

LIBRARY_SHA='6abdf60fb353da15d87030427e16a982e7819a6f479563b8acfde81110e78bf4'
MODULE='mxfp8_gemm_cutlass_sm120'
LIBRARY=Path('/usr/local/lib/python3.12/dist-packages/flashinfer/data/aot')/MODULE/(MODULE+'.so')
SHAPES=((256,256),(5120,256),(1024,5120),(5120,5120))
ROWS=(1,32,1056)
SCALES=(.25,1.,8.)


def main():
    assert os.environ.get('FLASHINFER_DISABLE_JIT')=='1'
    assert hashlib.sha256(LIBRARY.read_bytes()).hexdigest()==LIBRARY_SHA
    mem={p[0][:-1]:int(p[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()
         if (p:=l.split())[0] in ('MemFree:','MemAvailable:')}
    assert mem['MemAvailable']>=96*2**30 and mem['MemFree']>=32*2**30,mem
    limits={p:(Path('/sys/fs/cgroup')/p).read_text().strip() for p in ('memory.max','memory.swap.max','cpu.max')}
    assert 0<int(limits['memory.max'])<=8*2**30 and limits['memory.swap.max']=='0'
    quota,period=map(int,limits['cpu.max'].split());assert 0<quota<=4*period
    import torch
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability(0)==(12,1)
    assert torch.cuda.memory_allocated()==0
    torch.cuda.set_per_process_memory_fraction(.0075,0)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_mxfp8
    spec=gen_gemm_sm120_module_cutlass_mxfp8()
    assert spec.is_aot and spec.aot_path==LIBRARY
    from vllm.model_executor.kernels.linear import init_mxfp8_linear_kernel
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
    kernel=init_mxfp8_linear_kernel()
    assert type(kernel).__name__=='FlashInferCutlassMxfp8LinearKernel'
    cases=[]
    for n,k in SHAPES:
        for scale in SCALES:
            w=(((torch.arange(n,device='cuda',dtype=torch.int32)[:,None]*3
                +torch.arange(k,device='cuda',dtype=torch.int32)[None,:])%5)-2).to(torch.bfloat16)*scale
            qw,sw=mxfp8_e4m3_quantize(w,is_sf_swizzled_layout=False)
            layer=torch.nn.Module()
            layer.weight=torch.nn.Parameter(qw,requires_grad=False)
            layer.weight_scale=torch.nn.Parameter(sw,requires_grad=False)
            kernel.process_weights_after_loading(layer)
            for m in ROWS:
                x=(((torch.arange(m,device='cuda',dtype=torch.int32)[:,None]
                    +torch.arange(k,device='cuda',dtype=torch.int32)[None,:]*3)%5)-2).to(torch.bfloat16)*scale
                expected=(x.float()@w.float().t()).to(torch.bfloat16)
                actual=kernel.apply_weights(layer,x)
                torch.cuda.synchronize()
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                row=dict(shape=[m,n,k],scale=scale,max_absolute_error=0,dtype=str(actual.dtype),
                         output_sha256=hashlib.sha256(actual.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
                cases.append(row);print(json.dumps(dict(stage='aot_mxfp8_case_pass',**row)),flush=True)
                del x,expected,actual
            del w,qw,sw,layer
    mapped={l.split()[-1] for l in Path('/proc/self/maps').read_text().splitlines() if MODULE+'.so' in l}
    assert mapped=={str(LIBRARY)},mapped
    assert hashlib.sha256(LIBRARY.read_bytes()).hexdigest()==LIBRARY_SHA
    assert not spec.jit_library_path.exists(), 'Unexpected JIT replacement for the AOT library'
    assert torch.cuda.max_memory_allocated()<=512*2**20
    result=dict(status='clean_mxfp8_aot_gpu_exact_pass',cases=cases,
                library_sha256=LIBRARY_SHA,mapped_libraries=sorted(mapped),
                native_kernel=type(kernel).__name__,flashinfer_jit_disabled=True,
                jit_replacement_created=False,limits=limits,initial_memory=mem,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),allocator_fraction=.0075,
                probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                model_weights_loaded=False,full_serving_validated=False)
    with Path('/results/complete.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(status=result['status'],cases=len(cases),peak_allocated_bytes=result['peak_allocated_bytes'])),flush=True)


if __name__=='__main__':main()
