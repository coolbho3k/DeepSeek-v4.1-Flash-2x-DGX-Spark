"""Strip only debug symbols from a completed seeded build into a NEW artifact.

CPU-only host step. Preserve the raw build and require every allocated ELF
section and program header to remain byte-identical. No kernel loading.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import struct
import subprocess

ROOT=Path('/home/emi/code/ds41')
MODULE='mxfp8_gemm_cutlass_sm120'
REL=Path('flashinfer/.cache/flashinfer/0.6.18.dev20260819/121a/cached_ops')/MODULE
GRAPH_SHA='de1e515a89b8942440260a70afeaab3a0205b12a9d0c93a77cd933bd50f2173c'


def sha(data):return hashlib.sha256(data).hexdigest()


def runtime_identity(data):
    """Parse only bounded ELF64 little-endian AArch64 shared objects."""
    if not 2**20<len(data)<4*2**20:raise ValueError('Unexpected library size')
    h=struct.unpack_from('<16sHHIQQQIHHHHHH',data)
    if (h[0][:6]!=b'\x7fELF\x02\x01' or h[1:4]!=(3,183,1)
            or h[8]!=64 or h[9]!=56 or not 0<h[10]<=32
            or h[11]!=64 or not 0<h[12]<=128 or not 0<h[13]<h[12]):
        raise ValueError('Unexpected ELF format')
    def span(offset,size):
        if offset<0 or size<0 or offset+size>len(data):raise ValueError('Truncated ELF')
        return data[offset:offset+size]
    sections=[struct.unpack('<IIQQQQIIQQ',span(h[6]+i*h[11],h[11])) for i in range(h[12])]
    names=span(sections[h[13]][4],sections[h[13]][5]);allocated={}
    for s in sections:
        if s[0]>=len(names):raise ValueError('Invalid section name')
        name=names[s[0]:].split(b'\0',1)[0].decode('ascii')
        if s[2]&2:
            if name in allocated:raise ValueError('Duplicate allocated section')
            body=b'' if s[1]==8 else span(s[4],s[5])
            allocated[name]=dict(type=s[1],flags=s[2],address=s[3],size=s[5],
                                 alignment=s[8],entry_size=s[9],sha256=sha(body))
    if not {'.text','.nv_fatbin','.dynsym','.dynstr','.dynamic','.bss'}<=set(allocated):
        raise ValueError('Missing runtime sections')
    return dict(elf_identity=h[0].hex(),type=h[1],machine=h[2],version=h[3],
                entry=h[4],flags=h[7],program_headers_sha256=sha(span(h[5],h[9]*h[10])),
                allocated_sections=allocated)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',required=True);p.add_argument('--output-run',required=True)
    p.add_argument('--expected-sha256',required=True);p.add_argument('--compile-container',required=True)
    args=p.parse_args()
    if (not re.fullmatch(r'mxfp8-clean-bootstrap-v[1-9][0-9]*',args.source_run)
            or not re.fullmatch(r'ds41-mxfp8-aot-v[1-9][0-9]*',args.output_run)
            or not re.fullmatch('[0-9a-f]{64}',args.expected_sha256)
            or not re.fullmatch('[0-9a-f]{64}',args.compile_container)):
        raise ValueError('Explicit versioned source/output and exact identities required')
    source=ROOT/'artifacts'/('ds41-'+args.source_run+'-cache')
    output=ROOT/'artifacts'/args.output_run
    if source.resolve()!=source or output.resolve()!=output or output.exists() or output.is_symlink():
        raise ValueError('Preserve existing artifacts and use canonical paths')
    def command(argv):return subprocess.check_output(argv,text=True,timeout=20)
    if command(['docker','ps','-q']).strip() or command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip():
        raise ValueError('Wait for idle containers and GPUs')
    mem={x[0][:-1]:int(x[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()
         if (x:=l.split())[0]=='MemAvailable:'}
    if mem['MemAvailable']<48*2**30:raise ValueError('At least48GiB available required')
    node=json.loads(command(['docker','inspect',args.compile_container]))[0]
    state,limits=node['State'],node['HostConfig']
    env=dict(x.split('=',1) for x in node['Config']['Env'])
    mounts=[x for x in node['Mounts'] if x['Destination']=='/cache']
    if (state['Running'] or state['ExitCode']!=0 or state['OOMKilled']
            or limits['Memory']!=32*2**30 or limits['MemorySwap']!=32*2**30
            or limits['NanoCpus']!=4*10**9 or limits.get('DeviceRequests')
            or env.get('NVIDIA_VISIBLE_DEVICES')!='void' or env.get('CUDA_VISIBLE_DEVICES')!=''
            or len(mounts)!=1 or mounts[0]['Source']!=str(source)):
        raise ValueError('Source must be owned by the exact completed CPU-only build')
    if sha((source/REL/'build.ninja').read_bytes())!=GRAPH_SHA:raise ValueError('Wrong seeded graph')
    before=(source/REL/(MODULE+'.so')).read_bytes()
    if sha(before)!=args.expected_sha256:raise ValueError('Source binary changed')
    identity=runtime_identity(before)
    # Apply only after Docker's Go client has finished; objcopy needs little RAM.
    resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20))
    resource.setrlimit(resource.RLIMIT_CPU,(30,30))
    output.mkdir();target=output/(MODULE+'.so')
    argv=['objcopy','--strip-debug',str(source/REL/(MODULE+'.so')),str(target)]
    version=command(['objcopy','--version']).splitlines()[0]
    subprocess.run(argv,check=True,timeout=20)
    after=target.read_bytes()
    if runtime_identity(after)!=identity:raise ValueError('Debug stripping changed runtime content')
    if (source/REL/(MODULE+'.so')).read_bytes()!=before:raise ValueError('Raw source changed')
    receipt=dict(status='mxfp8_debug_stripped_runtime_unchanged',source_run=args.source_run,
                 compile_container=args.compile_container,image_id=node['Image'],
                 source_sha256=sha(before),library_sha256=sha(after),library_bytes=len(after),
                 graph_sha256=GRAPH_SHA,command=argv,objcopy_version=version,
                 runtime_identity=identity,probe_sha256=sha(Path(__file__).read_bytes()),
                 source_preserved=True,gpu_kernel_execution_tested=False,serving_reuse_tested=False)
    with (output/'receipt.json').open('x') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps({k:receipt[k] for k in ('status','library_sha256','library_bytes','gpu_kernel_execution_tested')}))


if __name__=='__main__':main()
