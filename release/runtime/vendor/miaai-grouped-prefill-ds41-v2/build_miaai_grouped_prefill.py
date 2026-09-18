# SPDX-License-Identifier: AGPL-3.0-only
# Additive build of the MiaAI-derived DS41 grouped kernel; retained licenses
# and exact corresponding sources are copied into the build artifact.
"""Offline CPU-only build, requiring an idle host and a bounded build container."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT=Path('/work')
BUILD=Path('/build')


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(name,value):
    with (BUILD/name).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    args=parser.parse_args()
    source=args.source
    assert source.resolve()==source and source.parent==ROOT/'artifacts'
    assert source.name.startswith('miaai-grouped-prefill-source-v')
    assert BUILD.resolve()==BUILD and BUILD.is_dir() and not any(BUILD.iterdir())
    memory={line.split(':')[0]:int(line.split()[1])*1024
        for line in Path('/proc/meminfo').read_text().splitlines()
        if line.split(':')[0] in ('MemFree','MemAvailable')}
    assert memory['MemAvailable']>=96*2**30 and memory['MemFree']>=32*2**30
    assert shutil.disk_usage(BUILD).free>=32*2**30
    limits={name:(Path('/sys/fs/cgroup')/name).read_text().strip()
        for name in ('memory.max','memory.swap.max','cpu.max')}
    assert limits['memory.max']==str(16*2**30) and limits['memory.swap.max']=='0'
    quota,period=map(int,limits['cpu.max'].split());assert 0<quota<=4*period
    assert os.environ['MAX_JOBS']=='1' and os.environ['TORCH_CUDA_ARCH_LIST']=='12.1a'
    assert os.environ.get('NVIDIA_VISIBLE_DEVICES') in ('void','none','')
    receipt=source/'prepared.json'
    assert receipt.resolve()==receipt and receipt.stat().st_size<256*2**10
    prepared=json.loads(receipt.read_bytes())
    assert prepared['status']=='grouped_prefill_sources_prepared_not_compiled'
    assert prepared['abi']==1003 and prepared['bits']==3 and prepared['codebook']=='MUL1'
    assert prepared['routing_before_down_fp16'] and prepared['fp16_gemm_and_output_boundaries']
    assert prepared['required_cuda_flags']==['-O3','-lineinfo','--fmad=false']
    pins=prepared['generated_sha256']
    total=0
    for name,pin in pins.items():
        path=source/name
        assert path.resolve()==path and path.is_relative_to(source) and path.is_file()
        assert path.stat().st_size<4*2**20 and digest(path)==pin
        total+=path.stat().st_size
    assert total<32*2**20
    for name in pins:
        target=BUILD/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/name,target)
    shutil.copyfile(receipt,BUILD/'prepared.json')
    shutil.copyfile(Path(__file__),BUILD/Path(__file__).name)
    save('attempt.json',dict(status='additive_grouped_prefill_build_started',memory=memory,
        cgroup_limits=limits,prepared_sha256=digest(receipt),source_sha256=pins,
        builder_sha256=digest(Path(__file__)),serving_modified=False,cuda_visible=False))
    start=time.monotonic()
    try:
        import torch
        assert not torch.cuda.is_initialized() and not torch.cuda.is_available()
        from torch.utils.cpp_extension import load
        module=load(name='ds41_miaai_fat_moe_v1',
            sources=[str(BUILD/'bindings.cpp'),str(BUILD/'include/quant/exl3_fat_moe.cu')],
            extra_include_paths=[str(BUILD/'include'),str(BUILD)],extra_cflags=['-O3'],
            extra_cuda_cflags=prepared['required_cuda_flags'],build_directory=str(BUILD),
            with_cuda=True,verbose=True)
        assert module.abi()==1003
        assert module.tile_rows_gateup()==module.tile_rows_down()==64
        assert not torch.cuda.is_initialized()
        assert all(digest(source/name)==pin and digest(BUILD/name)==pin for name,pin in pins.items())
        binary=Path(module.__file__)
        result=dict(status='additive_grouped_prefill_built_cpu_only',abi=1003,
            elapsed_seconds=time.monotonic()-start,binary_name=binary.name,
            binary_sha256=digest(binary),binary_bytes=binary.stat().st_size,
            prepared_sha256=digest(receipt),source_sha256=pins,torch_version=torch.__version__,
            cuda_version=torch.version.cuda,cuda_initialized=False,gpu_qualified=False,
            serving_modified=False)
        save('complete.json',result)
        print(json.dumps({k:result[k] for k in ('status','abi','elapsed_seconds','binary_name','binary_sha256')}),flush=True)
    except BaseException as error:
        save('failed.json',dict(status='failed',error=repr(error),elapsed_seconds=time.monotonic()-start))
        raise


if __name__=='__main__':main()
