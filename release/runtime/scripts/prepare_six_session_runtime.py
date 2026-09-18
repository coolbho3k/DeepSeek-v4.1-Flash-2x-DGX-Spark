# SPDX-License-Identifier: AGPL-3.0-only
"""Freeze C6/24-row cooperative serving over the qualified additive KV runtime.

No edits to the running/rollback kit, model weights, vendor sources or drivers.
All transformed source and native binary identities remain independently pinned.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'probes'))
from verify_runtime_bundle import verify
PARENT=ROOT/'artifacts/ds41-runtime-recipe-v13'
PARENT_SHA='4a15576d7ccb238c9473977c1791cc612f180c5e01eb905687ea93aaad1075cc'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,indent=2,sort_keys=True)+'\n').encode()

ROUTE_KERNEL='''@tr.jit
def _prepare(X, Ids, Weights, Mapping, HalfX, Local, HalfWeights, Counters,
             ROWS: tl.constexpr, XR: tl.constexpr, XC: tl.constexpr,
             IR: tl.constexpr, IC: tl.constexpr, WR: tl.constexpr, WC: tl.constexpr,
             B: tl.constexpr):
    # One bounded vector per row instead of a single 131072-element program.
    # Routing and counters have one writer; later kernels on the same stream
    # observe completion of the complete preparation grid.
    row = tl.program_id(0)
    pos = tl.arange(0, B)
    x = tl.load(X+row*XR+pos*XC, pos < 5120, other=0.)
    tl.store(HalfX+row*5120+pos, x.to(tl.float16), pos < 5120)
    if row == 0:
        slot = tl.arange(0, 256)
        raw = tl.load(Ids+(slot//6)*IR+(slot%6)*IC, slot < ROWS*6, other=-1).to(tl.int64)
        mapped = tl.load(Mapping+tl.where((raw >= 0)&(raw < 384), raw, 384))
        tl.store(Local+slot, mapped, slot < ROWS*6)
        rw = tl.load(Weights+(slot//6)*WR+(slot%6)*WC, slot < ROWS*6, other=0.)
        tl.store(HalfWeights+slot, rw.to(tl.float16), slot < ROWS*6)
        ctr = tl.arange(0, 4096)
        tl.store(Counters+ctr, 0, ctr < 2547)
'''

PREFILL_RECLAIM='''def execute_with_prefill_reclaim(worker, scheduler_output, execute):
    """Return unused CPU/CUDA blocks at sampled boundaries, never live state.

    Long prefills may never enter decode for many minutes. Reclaim after16
    successful prefill steps as well as at the original transition to decode.
    Model/KV/graph owners remain referenced; no thresholds or budgets change.
    """
    tokens = scheduler_output.total_num_scheduled_tokens
    steps = getattr(worker, '_ds41_prefill_steps_since_reclaim', 0)
    decode = 1 <= tokens <= 4*config.PROFILE['max_num_seqs']
    periodic = tokens > 32 and steps >= 16
    if (decode or periodic) and getattr(worker, '_ds41_prefill_reclaim_pending', False):
        cuda = baseline.torch.cuda
        if (not worker.use_v2_model_runner
                or worker.model_runner.execute_model_state is not None
                or cuda.is_current_stream_capturing()):
            raise RuntimeError('Prefill reclaim requires a sampled V2 state outside capture')
        cuda.synchronize()
        stage = 'ds41_prefill_interval_reclaim' if periodic else 'ds41_after_prefill_reclaim'
        worker._log_load_memory(stage+'_before')
        baseline.trim_process_heap()
        cuda.empty_cache()
        worker._log_load_memory(stage+'_after')
        worker._ds41_prefill_reclaim_pending = False
        worker._ds41_prefill_steps_since_reclaim = 0
    result = execute(scheduler_output)
    if tokens > 32:
        worker._ds41_prefill_reclaim_pending = True
        worker._ds41_prefill_steps_since_reclaim = getattr(worker, '_ds41_prefill_steps_since_reclaim', 0)+1
    return result
'''

DISPLAY_ALLOCATION_PHASE='''
from contextlib import contextmanager

_real_allocation_size = None

@contextmanager
def real_allocation(kv_cache_config):
    """Distinguish final KV from temporary profiling by lifecycle, not size."""
    global _real_allocation_size
    if _real_allocation_size is not None or _owners:
        raise RuntimeError('Final display KV initialization must occur exactly once')
    sizes = {tensor.size for tensor in kv_cache_config.kv_cache_tensors}
    if len(sizes) != 1:
        raise ValueError('Expected one shared final KV backing size')
    size = sizes.pop()
    if type(size) is not int or not 0 < size <= DISPLAY_BYTES:
        raise ValueError('Final KV descriptors exceed the display-only budget')
    _real_allocation_size = size
    try:
        yield
        if len(_owners) != 1 or _owners[0].ordinary_bytes != 0:
            raise RuntimeError('Final KV did not use exactly one display-only owner')
    finally:
        # Reset CPU state even on failure; never query CUDA or free live owners.
        _real_allocation_size = None
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version',type=int,default=14)
    p.add_argument('--release',type=int,default=96)
    p.add_argument('--display-only-kv',action='store_true',
        help='Approved 3Mi-token candidate: zero ordinary KV and512MiB watchdog')
    a=p.parse_args()
    target=ROOT/f'artifacts/ds41-runtime-recipe-v{a.version}'
    deployment=ROOT/f'reports/ds41-release-v{a.release}-deployment.json'
    if target.exists() or deployment.exists():raise ValueError('Fresh immutable candidate required')
    verify(PARENT,PARENT_SHA)
    previous=json.loads((PARENT/'bundle-manifest.json').read_bytes())
    payload={name:(PARENT/name).read_bytes() for name in previous['files']}
    histories={name:{sha(raw)} for name,raw in payload.items() if name.endswith('.py')}
    def edit(name,before,after):
        source=payload[name].decode()
        if source.count(before)!=1:raise ValueError('Changed anchor: '+name+': '+before[:100])
        payload[name]=source.replace(before,after).encode()
    # Both the external launcher and worker validate the exact same profile.
    for name in ('tools/launch_profile.py','serving/ds41/launch_profile.py'):
        edit(name,"values['max_num_seqs'] not in (1,2)","not 1 <= values['max_num_seqs'] <= 6")
        edit(name,'supports one or two simultaneous sequences','supports one through six simultaneous sequences')
    edit('serving/ds41/combined_dspark.py',"'not 1 <= scheduler.max_num_seqs <= 2'",
         "'not 1 <= scheduler.max_num_seqs <= 6'")
    # Native FP4 speculative metadata flattens C*4 queries to [C*4,1,...].
    # Preserve the total24-row bound; both [6,4] and [24,1] must be legal.
    edit('serving/ds41/dcp_indexer_graph.py','1 <= values.shape[0] <= 8',
         '1 <= values.shape[0] <= 24')
    edit('serving/ds41/dcp_indexer_graph.py',
         "raise ValueError('Expected bounded MXFP4 next_n1..4 TP2 indexer queries')",
         "raise ValueError(f'Expected at most24 MXFP4 query rows: values={values.shape}, scale={None if q_scale is None else q_scale.shape}, lengths={lengths.shape}, cap={max_model_len}')")
    old_sizes='(1, 2, 3, 4, 6, 8, 12, 18, 24)'
    new_sizes='(1, 2, 3, 4, 6, 8, 9, 12, 15, 16, 18, 20, 24)'
    edit('serving/ds41/combined_config.py',old_sizes,new_sizes)
    edit('tools/portable_node.py',str(list(ast.literal_eval(old_sizes))),str(list(ast.literal_eval(new_sizes))))
    edit('serving/conservative.yaml',str(list(ast.literal_eval(old_sizes))),str(list(ast.literal_eval(new_sizes))))
    edit('serving/spark_combined_ready.py',
        "    if PROFILE['max_num_seqs'] == 2:\n        if (not any(r['tokens']==8 and r['requests']==2 for r in target_graphs)\n                or not any(r['tokens']==6 and r['requests']==2 for r in draft_graphs)):\n            raise ValueError('Missing two-request target/draft graphs')",
        "    if enabled:\n        for requests in range(1, PROFILE['max_num_seqs']+1):\n            if (not any(r['tokens']==4*requests and r['requests']==requests for r in target_graphs)\n                    or not any(r['tokens']==3*requests and r['requests']==requests for r in draft_graphs)):\n                raise ValueError(f'Missing target/draft graphs for {requests} requests')")
    # Actual host-memory checks and GPU utilization stay unchanged. The cgroup
    # allowance is a limit, not an allocation or credit to native KV profiling.
    edit('tools/portable_node.py',"'--memory=8g','--memory-swap=8g'","'--memory=9g','--memory-swap=9g'")
    edit('tools/portable_node.py',"actual['Memory'] != 8*GIB or actual['MemorySwap'] != 8*GIB",
         "actual['Memory'] != 9*GIB or actual['MemorySwap'] != 9*GIB")
    edit('serving/guarded_worker.py',"limits['memory.max']!=str(8*GIB)","limits['memory.max']!=str(9*GIB)")
    edit('serving/guarded_worker.py','actual8GiB/no-swap','actual9GiB/no-swap')
    edit('serving/combined_worker.py','1 <= tokens <= 8 and',
         "1 <= tokens <= 4*config.PROFILE['max_num_seqs'] and")
    worker_source=payload['serving/combined_worker.py'].decode()
    old_reclaim=next(n for n in ast.parse(worker_source).body
        if isinstance(n,ast.FunctionDef) and n.name=='execute_with_prefill_reclaim')
    edit('serving/combined_worker.py',ast.get_source_segment(worker_source,old_reclaim),PREFILL_RECLAIM.rstrip())
    # C6 short requests passed with1GiB ordinary KV, but4.2M prompt admission
    # crossed the head-node RAM watchdog. Trade512MiB of surplus KV for host
    # headroom and return only unused large-tokenization heap. Native profiling,
    # the1.75GiB display suffix and768MiB watchdog remain unchanged.
    edit('serving/ds41/display_kv.py',
         'cap!=ORDINARY_LIMIT or len(available)!=2',
         'type(cap) is not int or not 512*2**20<=cap<=ORDINARY_LIMIT or len(available)!=2')
    edit('serving/ds41/display_kv.py',
         'positive native budgets and a1GiB ordinary cap',
         'positive native budgets and a512..1024MiB ordinary cap')
    edit('serving/spark_kv_cap.py','ordinary_cap_bytes=1073741824,',
         "ordinary_cap_bytes=__import__('ds41.combined_config',fromlist=['KV_CAP_BYTES']).KV_CAP_BYTES,")
    edit('tools/portable_node.py',"env['NVIDIA_DRIVER_CAPABILITIES']='compute,utility,graphics,display'",
         "env.update(MALLOC_ARENA_MAX='2',MALLOC_TRIM_THRESHOLD_='131072',VLLM_SPARSE_INDEXER_MAX_LOGITS_MB='128')\n    env['NVIDIA_DRIVER_CAPABILITIES']='compute,utility,graphics,display'")
    edit('serving/ds41/combined_config.py','    hf = model.hf_config',
         "    if os.environ.get('VLLM_SPARSE_INDEXER_MAX_LOGITS_MB') != '128':\n        raise ValueError('C6 long contexts require bounded128MiB native indexer logits')\n    hf = model.hf_config")
    payload['serving/ds41/tokenizer_heap.py']=(ROOT/'ds41/tokenizer_heap.py').read_bytes()
    edit('serving/serve.py','register_parent_heap()\n',
         'register_parent_heap()\nfrom ds41.tokenizer_heap import register as register_tokenizer_heap\nregister_tokenizer_heap()\n')
    if a.display_only_kv:
        for name in ('tools/launch_profile.py','serving/ds41/launch_profile.py'):
            edit(name,"if not 512 <= values['kv_cap_mib'] <= 1536:",
                 "if values['kv_cap_mib'] != 0:")
            edit(name,'KV downward cap must be512..1536MiB per rank; native admission still applies',
                 'Display-only candidate requires zero ordinary KV; native admission still applies')
        edit('tools/portable_node.py','STOP_AVAILABLE = 768*2**20','STOP_AVAILABLE = 512*2**20')
        name='serving/ds41/display_kv.py'
        edit(name,'not 512*2**20<=cap<=ORDINARY_LIMIT','cap != 0')
        edit(name,'positive native budgets and a512..1024MiB ordinary cap',
             'positive native budgets and zero ordinary KV')
        edit(name,"if min(ordinary)<512*2**20:raise ValueError('Insufficient native ordinary-RAM KV budget')",
             "if any(ordinary):raise ValueError('Display-only KV cannot consume ordinary RAM')")
        edit(name,'    if size<=DISPLAY_BYTES:\n        return torch.zeros(size,dtype=dtype,device=device)',
             "    if _real_allocation_size is None:\n        if _owners or size > DISPLAY_BYTES:\n            raise RuntimeError('Unexpected profiling allocation outside initialization')\n        return torch.zeros(size,dtype=dtype,device=device)\n    if size != _real_allocation_size:\n        raise ValueError('Final KV allocation differs from admitted descriptors')")
        edit(name,'ordinary=(size-DISPLAY_BYTES+QUANTUM-1)//QUANTUM*QUANTUM',
             'ordinary=0')
        edit(name,"if before['MemAvailable']<ordinary+768*2**20:",
             "if before['MemAvailable']<ordinary+512*2**20:")
        edit(name,"if after['MemAvailable']<768*2**20:",
             "if after['MemAvailable']<512*2**20:")
        edit(name,'contains <=1GiB ordinary registered RAM followed by1.75GiB display reserve.',
             'contains only1.75GiB display reserve; final KV uses zero ordinary RAM.')
        payload[name]+=DISPLAY_ALLOCATION_PHASE.encode()
        edit('serving/combined_worker.py','    init_device = _init_device\n',
             '    init_device = _init_device\n\n    def initialize_from_config(self, kv_cache_config):\n        from ds41.display_kv import real_allocation\n        with real_allocation(kv_cache_config):\n            return super().initialize_from_config(kv_cache_config)\n')
        edit('serving/combined_worker.py','native admission and768MiB steady guard remain.',
             'native admission retained; this candidate uses the approved512MiB steady guard.')
    # Expanded scratch aliases remain strictly inside the same four buffers.
    name='serving/ds41/cooperative_contract.py'
    text=payload[name].decode().replace('(48, 5120)','(144, 5120)').replace('(48, 1152)','(144, 1152)')
    text=text.replace('48*5120*2','144*5120*2').replace('48*1152*2','144*1152*2').replace('(851,)','(2547,)')
    text=text.replace('x_shape[0] <= 8','x_shape[0] <= 24')
    payload[name]=text.encode()
    name='serving/ds41/cooperative_routes.py'
    source=payload[name].decode();start=source.index('@tr.jit');end=source.index('\n\ndef prepare(',start)
    payload[name]=(source[:start]+ROUTE_KERNEL+source[end:]).encode()
    edit(name,'_prepare[(1,)]','_prepare[(len(x),)]')
    edit(name,'tr.next_power_of_2(x.numel())','8192')
    name='serving/ds41/cooperative_moe.py'
    edit(name,'goal50_coop_abi() != 1','goal50_coop_abi() != 2')
    edit(name,'(48, 851, 344)','(48, 2547, 344)')
    edit(name,"receipt.get('abi') != 1","receipt.get('abi') != 2")
    # 24*6=144 assignments must reach the cooperative path, never the old
    # 128-assignment small kernel. Disabling cooperative still uses the proper
    # grouped fallback. Keep the dispatch lock, poison guard and stream fences.
    edit('serving/spark_fused_moe_async.py','or ids.numel() > 128:',
         "or (ids.numel() > 128 and not globals().get('_ds41_coop_eligible', lambda *_: False)(x.shape, ids.shape)):")
    edit('serving/spark_grouped_prefill.py','or ids.numel()<=128:',
         "or ids.numel()<=128 or __import__('spark_fused_moe_async').__dict__.get('_ds41_coop_eligible', lambda *_: False)(x.shape,ids.shape):")
    edit('serving/spark_combined_miaai.py','cooperative_rows=list(range(1,9))','cooperative_rows=list(range(1,25))')
    # No need for the grouped-prefill workspace merely because C6 routes 144
    # assignments: cooperative itself uses only existing small-dispatch temps.
    edit('serving/spark_combined_miaai.py','needs_fat=ids.numel() > 128)',
         "needs_fat=ids.numel() > 128 and not (cooperative_enabled and modules['ds41.cooperative_contract'].selected_shape((ids.shape[0],5120),ids.shape)))")
    build=ROOT/'artifacts/cooperative-six-session-build-v1'
    native=json.loads((build/'complete.json').read_bytes());binary=(build/'cooperative_moe.so').read_bytes()
    if native['abi']!=2 or sha(binary)!=native['binary_sha256']:raise ValueError('Unexpected 24-row binary')
    payload['serving/cooperative_moe.so']=binary
    payload['serving/cooperative-native.json']=encoded(native)
    payload['sources/cooperative24.cu']=(build/'goal50_fixed_coop.cu').read_bytes()
    payload['sources/cooperative24.cuh']=(build/'upstream/exllamav3/exllamav3_ext/quant/goal50_fixed_coop_kernel.cuh').read_bytes()
    # Refresh binary/JSON attestation explicitly (Python pins propagate below).
    name='serving/spark_backend_attestation.py';source=payload[name].decode()
    assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='PRIVATE_SOURCES' for t in n.targets))
    pins=ast.literal_eval(assignment.value)
    for n in ('cooperative_moe.so','cooperative-native.json'):pins[n]=sha(payload['serving/'+n])
    pins['ds41/tokenizer_heap.py']=sha(payload['serving/ds41/tokenizer_heap.py'])
    edit(name,ast.get_source_segment(source,assignment),'PRIVATE_SOURCES = '+repr(pins))
    # Retain the attested size bound, updated only for this rebuilt binary.
    # PRIVATE_SOURCES is also a dict with that key; select the numeric size map.
    for node in ast.walk(ast.parse(payload[name])):
        if isinstance(node,ast.Dict):
            for key,value in zip(node.keys,node.values):
                if isinstance(key,ast.Constant) and key.value=='cooperative_moe.so' and isinstance(value,ast.Constant) and isinstance(value.value,int):
                    edit(name,"'cooperative_moe.so': "+str(value.value),"'cooperative_moe.so': "+str(len(binary)))
    for _ in range(32):
        for n,values in histories.items():values.add(sha(payload[n]))
        updates={before:sha(payload[n]) for n,values in histories.items() for before in values if before!=sha(payload[n])}
        changed=False
        for n,raw in list(payload.items()):
            if not n.endswith('.py') or not n.startswith(('serving/','tools/')):continue
            for before,after in updates.items():raw=raw.replace(before.encode(),after.encode())
            if raw!=payload[n]:payload[n]=raw;changed=True
        if not changed:break
    else:raise RuntimeError('Source pins failed to converge')
    requirements=json.loads(payload['runtime-requirements.json'])
    # Historical component proofs stay historical; the selected launch's
    # limits are recorded separately from old qualified profiles.
    requirements['six_session_candidate']=dict(max_num_seqs=6,max_model_len=1048576,
        cpu_memory_gib=9,gpu_memory_utilization=.92,
        minimum_aggregate_tokens=(3 if a.display_only_kv else 4)*2**20,
        cooperative_rows=24,additional_persistent_moe_scratch_bytes=0,
        ordinary_kv_cap_mib=0 if a.display_only_kv else 512,
        host_watchdog_mib=512 if a.display_only_kv else 768,
        display_only_kv=a.display_only_kv,large_prompt_cpu_heap_trim=True,
        indexer_max_logits_mib=128,prefill_reclaim_interval_steps=16,
        full_model_qualified=False)
    requirements['loaded_backend_verification']['sha256']=sha(payload['serving/spark_backend_attestation.py'])
    payload['runtime-requirements.json']=encoded(requirements)
    payload['serving/overlay-manifest.json']=encoded({n.removeprefix('serving/'):sha(raw)
        for n,raw in payload.items() if n.startswith('serving/') and n!='serving/overlay-manifest.json'})
    for n in ('scripts/prepare_six_session_runtime.py','scripts/build_six_session_cooperative.py'):
        payload[n]=(ROOT/n).read_bytes()
    for n,raw in payload.items():
        if n.endswith('.py'):compile(raw,n,'exec')
    manifest=dict(format=previous['format'],standalone_runtime=False,clean_rebuild_qualified=False,
        publication_approved=False,variant='six_sessions_cooperative24_'+('display_only_kv' if a.display_only_kv else 'additive_display_kv'),
        parent_manifest_sha256=PARENT_SHA,serving_qualified=False,
        files={n:dict(bytes=len(raw),sha256=sha(raw)) for n,raw in sorted(payload.items())})
    for n,raw in payload.items():
        path=target/n;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    raw=encoded(manifest);(target/'bundle-manifest.json').write_bytes(raw);verify(target,sha(raw))
    config=json.loads((ROOT/'reports/ds41-release-v95-deployment.json').read_bytes())
    config.update(run_id=f'ds41-release-v{a.release}',kit_manifest_sha256=sha(raw))
    config['serving']['max_num_seqs']=6
    config['serving']['kv_cap_mib']=0 if a.display_only_kv else 512
    for node in config['nodes']:node['kit']=str(target)
    deployment.write_bytes(encoded(config))
    print(json.dumps(dict(kit=str(target),deployment=str(deployment),manifest_sha256=sha(raw))),flush=True)

if __name__=='__main__':main()
