# SPDX-License-Identifier: AGPL-3.0-only
"""Freeze full cooperative decoding and configurable direct-LAN serving.

Reuse the proven native binaries. Large MoE prefills are bounded subcalls,
not larger native launches. Never mutate the running/rollback runtime.
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
PARENT=ROOT/'artifacts/ds41-runtime-draft-combined-v6'
PARENT_SHA='91fa2f50782ac3b88e94d1d097ad5469d98951099e6a68ffe35ae618a0b2d862'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
def once(source,old,new):
    if source.count(old)!=1:raise ValueError('Rewrite anchor changed: '+old[:120])
    return source.replace(old,new)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version',type=int,default=1)
    p.add_argument('--release',type=int,default=83)
    a=p.parse_args()
    target=ROOT/f'artifacts/ds41-runtime-recipe-v{a.version}'
    deployment=ROOT/f'reports/ds41-release-v{a.release}-deployment.json'
    if target.exists() or deployment.exists():raise ValueError('Fresh candidate required')
    verify(PARENT,PARENT_SHA)
    old=json.loads((PARENT/'bundle-manifest.json').read_bytes())
    payload={n:(PARENT/n).read_bytes() for n in old['files']}
    def edit(n,before,after):payload[n]=once(payload[n].decode(),before,after).encode()
    profile=(ROOT/'ds41/launch_profile.py').read_bytes()
    payload['serving/ds41/launch_profile.py']=profile
    payload['tools/launch_profile.py']=profile
    override=(ROOT/'ds41/startup_headroom_override.py').read_bytes()
    payload['serving/ds41/startup_headroom_override.py']=override
    native_utils=(ROOT/'vendor/vllm-v41/vllm/v1/worker/utils.py').read_text()
    native_request=next(n for n in ast.parse(native_utils).body if isinstance(n,ast.FunctionDef) and n.name=='request_memory')
    request_sha=sha(ast.get_source_segment(native_utils,native_request).strip().encode())
    # Keep legacy local receipts intact while admitting the published HF
    # manifest via a separate equally strict, public-only verifier.
    public=payload['tools/verify_downloaded_release.py'].decode()
    public=once(public,"FORMAT = 'ds41_release_metadata_preview_v1'","FORMAT = 'ds41_hf_weights_release_v1'")
    public=once(public,"or manifest.get('publication_ready') is not False",
        "or manifest.get('publication_authorized') is not True\n            or manifest.get('repo_id') != 'coolbho3k/DeepSeek-V4.1-Flash-EXL3-3bpw'")
    public=once(public,'            digest.update(chunk)',
        '            digest.update(chunk)\n            os.posix_fadvise(stream.fileno(), stream.tell()-len(chunk), len(chunk), os.POSIX_FADV_DONTNEED)')
    payload['tools/verify_public_download.py']=public.encode()
    edit('serving/ds41/cooperative_contract.py',
        '    # Both-rank measurements favor our tuned staged kernels at1..4 rows.\n    return eligible_shape(x_shape, ids_shape) and x_shape[0] >= 5',
        '    # Full MiaAI cooperative dispatch, including C1 speculative verification.\n    return eligible_shape(x_shape, ids_shape)')
    n='serving/ds41/combined_config.py'
    edit(n,'MAX_TOKENS = 2048',
        "from .launch_profile import from_environment\nPROFILE = from_environment()\nMAX_TOKENS = PROFILE['max_num_batched_tokens']")
    edit(n,'INITIAL_UTILIZATION = 0.92',"INITIAL_UTILIZATION = PROFILE['gpu_memory_utilization']")
    edit(n,'KV_CAP_BYTES = 1610612736',"KV_CAP_BYTES = PROFILE['kv_cap_mib'] * 2**20")
    edit(n,'scheduler.max_num_seqs != 1',"scheduler.max_num_seqs != PROFILE['max_num_seqs']")
    edit(n,"length = model.max_model_len","length = model.max_model_len\n    if (length != PROFILE['max_model_len'] or scheduler.long_prefill_token_threshold != PROFILE['long_prefill_token_threshold']):\n        raise ValueError('CLI/config differ from the explicit serving profile')")
    edit(n,'one request and2048-token prefill','profile-matched request and prefill budgets')
    edit(n,'    # Still one whole uncompressed request, not divided twice by DCP or\n    # compression. Native MXFP4 workspace layout and consumers are unchanged.',
        '    # Native split_indexer_prefill_chunks packs/splits requests to this\n    # shared bound. Never multiply by concurrency or divide again by DCP.\n    # Each individual request still fits before compression.')
    n='serving/ds41/combined_dspark.py'
    edit(n,'image_config = _compile(vision.validate_config, [',
        "image_config = _compile(vision.validate_config, [\n        ('scheduler.max_num_seqs != 1', 'not 1 <= scheduler.max_num_seqs <= 2'),")
    # Existing native expert kernels have a reviewed2048-row ABI. Keep their
    # scratch/segment limits unchanged and split only token-independent MoE.
    n='serving/spark_combined_miaai.py'
    edit(n,'maximum_prefill_tokens=2048','maximum_prefill_tokens=3072')
    edit(n,'cooperative_rows=list(range(5,9))','cooperative_rows=list(range(1,9))')
    edit(n,'cooperative_small_rows_keep_staged=True','cooperative_small_rows_keep_staged=False')
    for expression in ("(fp4_main_kv, 'MAX_WRITE_ROWS', 1056, 2048)",
        "(fp4_rope_store, 'MAX_WRITE_ROWS', 1056, 2048)",
        "(engram, 'MAX_TOKENS', 1056, 2048)",
        "(modules['spark_indexer_k_math'], 'MAX_ROWS', 1056, 2048)",
        "(modules['spark_indexer_k_math'], 'MAX_TEMPORARY_BYTES', 1056 * 256, 2048 * 256)"):
        edit(n,expression,expression.replace(', 2048',', config.MAX_TOKENS'))
    edit(n,"('indices.shape[0] <= 1056', 'indices.shape[0] <= 2048')",
        "('indices.shape[0] <= 1056', f'indices.shape[0] <= {config.MAX_TOKENS}')")
    edit(n,"('len(x) > 1056', 'len(x) > 2048')",
        "('len(x) > 1056', f'len(x) > {config.MAX_TOKENS}')")
    # Add source rewrite before the unchanged per-subcall validation/lock.
    chunk='''if len(x) > 2048:
        if len(x) > _ds41_prefill_limit or torch.cuda.is_current_stream_capturing():
            raise ValueError('Large MoE must be bounded eager prefill')
        if ids.shape[0] != len(x) or weights.shape != ids.shape:
            raise ValueError('MoE batch routing shape mismatch')
        return torch.cat([self(experts,x[start:start+2048],ids[start:start+2048],
            weights[start:start+2048],chunk_tokens) for start in range(0,len(x),2048)],dim=0)
    base.validate_inputs(experts,x,ids,weights,chunk_tokens)'''
    edit(n,"compile_shared(grouped.GroupedDispatcher, '__call__', [",
        "compile_shared(grouped.GroupedDispatcher, '__call__', [\n            "+repr(('base.validate_inputs(experts,x,ids,weights,chunk_tokens)',chunk))+',')
    edit(n,"'fat.seg_rows,fat.seg_lengths,MAX_SEGMENTS,256)'),\n        ])",
        "'fat.seg_rows,fat.seg_lengths,MAX_SEGMENTS,256)'),\n        ], {'_ds41_prefill_limit':config.MAX_TOKENS})")
    edit(n,'dspark_enabled=dspark_enabled, dspark_full_model_qualified=False,',
        'dspark_enabled=dspark_enabled, dspark_full_model_qualified=False,\n            serving_profile=dict(config.PROFILE), maximum_prefill_tokens=config.MAX_TOKENS,\n            native_moe_subcall_maximum_tokens=2048,')
    edit('serving/ds41/native_vocab_stage.py','MAX_TOKENS = 2048',
        'from .combined_config import MAX_TOKENS')
    edit('serving/ds41/combined_vocab.py','from .combined_config import validate_config',
        'from .combined_config import validate_config, MAX_TOKENS')
    edit('serving/ds41/combined_vocab.py','input_.numel() > 2048',
        'input_.numel() > MAX_TOKENS')
    # SM121's native MXFP4 metadata flattens each speculative token to an
    # independent query/table row. Two requests with three drafts therefore
    # arrive as [8,1,32,64], not [2,4,32,64]. Preserve the24-row total bound.
    edit('serving/ds41/dcp_indexer_graph.py',
        'not 1 <= values.shape[0] <= 6',
        'not 1 <= values.shape[0] <= 8\n            or values.shape[0] * values.shape[1] > 24')
    edit('serving/combined_worker.py','if 1 <= tokens <= 4 and', 'if 1 <= tokens <= 8 and')
    # API/head placement adds CPU processes on dgx0. Its startup-only extra
    # margin is not a GPU budget: retain native requested-memory admission,
    # the allocator ceiling, actual profiling, and768MiB continuous watchdog.
    n='serving/combined_worker.py'
    edit(n,"(ast.get_source_segment(source, first), '_ds41_validate_worker(worker)'),",
        "(ast.get_source_segment(source, first), '_ds41_validate_worker(worker)'),\n"
        "        (\"memory['MemFree']<required_free or memory['MemAvailable']<required_available\",\n"
        "         'not _ds41_host_admitted(worker, memory, required_free, required_available)'),")
    edit(n,"], {'_ds41_validate_worker': config.validate_worker,",
        "], {'_ds41_validate_worker': config.validate_worker, '_ds41_host_admitted':host_admitted,")
    edit(n,'validate_initial = prepare_validator()',
        "from ds41.startup_headroom_override import host_admitted, install_native\n"
        "validate_initial = prepare_validator()\n"
        f"install_native(baseline.gpu_worker, {request_sha!r})")
    edit(n,"'required_available=required+768*2**20'", "'required_available=required+256*2**20'")
    edit(n,"'required_initial_host_available_bytes=required+768*2**20'", "'required_initial_host_available_bytes=required+256*2**20'")
    edit(n,'explicit high-utilization768MiB startup reserve','explicit256MiB additional startup reserve; native admission and768MiB steady guard remain')
    # MemAvailable is a reclaimability estimate after kernel reserves, not
    # an upper bound on MemFree. Check both counters independently.
    edit('serving/guarded_worker.py',
        "not memory['MemFree']<=memory['MemAvailable']<=memory['MemTotal']",
        "not (memory['MemFree']<=memory['MemTotal'] and memory['MemAvailable']<=memory['MemTotal'])")
    n='serving/spark_combined_ready.py'
    edit(n,'INITIAL_UTILIZATION = 0.92',
        'import importlib.util\n_profile_spec = importlib.util.spec_from_file_location("ds41_inspection_profile", Path(__file__).parent/"ds41/launch_profile.py")\n_profile = importlib.util.module_from_spec(_profile_spec)\n_profile_spec.loader.exec_module(_profile)\nPROFILE = _profile.from_environment()\nINITIAL_UTILIZATION = PROFILE["gpu_memory_utilization"]')
    edit(n,'    vocabulary = vocabulary_inventory(runner, descriptor, worker.rank)',
        "    if PROFILE['max_num_seqs'] == 2:\n        if (not any(r['tokens']==8 and r['requests']==2 for r in target_graphs)\n                or not any(r['tokens']==6 and r['requests']==2 for r in draft_graphs)):\n            raise ValueError('Missing two-request target/draft graphs')\n    vocabulary = vocabulary_inventory(runner, descriptor, worker.rank)")
    # Frozen YAML defaults; explicit validated CLI profile may override them.
    edit('serving/conservative.yaml','max-num-seqs: 1','max-num-seqs: 2')
    edit('serving/conservative.yaml','max-num-batched-tokens: 2048',
        'max-num-batched-tokens: 3072\nlong-prefill-token-threshold: 2816')
    n='tools/portable_node.py'
    edit(n,"not 0 <= mem['MemFree'] <= mem['MemAvailable'] <= mem['MemTotal']",
        "not (0 <= mem['MemFree'] <= mem['MemTotal'] and 0 <= mem['MemAvailable'] <= mem['MemTotal'])")
    edit(n,"weights = module(kit,'portable_weights_check','tools/verify_downloaded_release.py',manifest)",
        "verifier = ('tools/verify_public_download.py' if config['model_manifest_sha256'] == '043785e20066f6212d30b3451a956802596a18d0b542b104ce2dd24bba900bc3' else 'tools/verify_downloaded_release.py')\n    weights = module(kit,'portable_weights_check',verifier,manifest)")
    edit(n,'NODE_RANKS = (1, 0)','NODE_RANKS = (0, 1)')
    edit(n,"'max-num-seqs': '1'","'max-num-seqs': '2'")
    edit(n,"'max-num-batched-tokens': '2048'","'max-num-batched-tokens': '3072', 'long-prefill-token-threshold': '2816'")
    edit(n,"'cache_manifest_sha256','fabric_network','nodes'","'cache_manifest_sha256','fabric_network','nodes','serving','api','startup_memory_override'")
    edit(n,"    network = ipaddress.IPv4Network(config['fabric_network'], strict=True)",
        "    from launch_profile import validate as validate_profile\n    validate_profile(config['serving'])\n    if type(config['startup_memory_override']) is not bool or (config['startup_memory_override'] and config['serving']['gpu_memory_utilization']!=.92):\n        raise ValueError('Explicit boolean startup override is scoped to0.92')\n    api = config['api']\n    if set(api) != {'host','port','master_port','model_name'}:\n        raise ValueError('Unknown API settings')\n    ipaddress.IPv4Address(api['host'])\n    if any(type(api[k]) is not int or not 1024 <= api[k] <= 65535 for k in ('port','master_port')) or api['port']==api['master_port']:\n        raise ValueError('Invalid serving ports')\n    if not re.fullmatch('[a-zA-Z0-9_.-]+',api['model_name']):\n        raise ValueError('Invalid served model name')\n    network = ipaddress.IPv4Network(config['fabric_network'], strict=True)")
    edit(n,"'fabric_ip','ifname','hca','gid_index','uid','gid'", "'fabric_ip','ifname','hca','gid_index','uid','gid','draft'")
    edit(n,"('kit','model','model_receipt','cache','runs')", "('kit','model','model_receipt','cache','runs','draft')")
    edit(n,"mounts = [('/home/emi/code/ds41/artifacts/ds41-draft-exl3-3bpw-sparse-v3','/draft-exl3',True)",
        "from launch_profile import environment as profile_environment\n    env.update(profile_environment(config['serving']))\n    env['DS41_ALLOW_STARTUP_MEMORY_SHORTFALL']='1' if config['startup_memory_override'] else '0'\n    mounts = [(node['draft'],'/draft-exl3',True)")
    edit(n,"'--master-addr',config['nodes'][1]['fabric_ip'],'--master-port','29541',\n        '--host','10.88.88.15','--port','8041','--served-model-name','deepseek-v41-flash-exl3',",
        "'--master-addr',config['nodes'][0]['fabric_ip'],'--master-port',str(config['api']['master_port']),\n        '--host',config['api']['host'],'--port',str(config['api']['port']),'--served-model-name',config['api']['model_name'],")
    edit(n,'    if rank == 1:\n        tail',
        "    for key,value in config['serving'].items():\n        if key != 'kv_cap_mib': tail += ['--'+key.replace('_','-'),str(value)]\n    if rank == 1:\n        tail")
    edit(n,"if index != 1 or not observed['state']['Running']:", "if index != 0 or not observed['state']['Running']:")
    # Loopback health on head: no DNS, SSH forwards, LAN address hardcoding.
    payload[n]=payload[n].replace(b"'http://10.88.88.15:8041/health'",b"f\"http://127.0.0.1:{config['api']['port']}/health\"")
    payload[n]=payload[n].replace(b"'http://10.88.88.15:8041/v1/models'",b"f\"http://127.0.0.1:{config['api']['port']}/v1/models\"")
    edit(n,"row.get('id') == 'deepseek-v41-flash-exl3'","row.get('id') == config['api']['model_name']")
    payload[n]=payload[n].replace(b'127.0.0.1:{config[', b"{('127.0.0.1' if config['api']['host']=='0.0.0.0' else config['api']['host'])}:{config[")
    # Each host samples both configured ports; do not use historical constants.
    edit(n,"validate_start_sample(node, sample)  # Before touching model/cache inputs.",
        "validate_start_sample(node, sample)  # Before touching model/cache inputs.\n    ports={str(config['api']['port']),str(config['api']['master_port'])}\n    if any(line.split()[3].rsplit(':',1)[-1] in ports for line in sample['sockets'].splitlines()):\n        raise ValueError('Configured serving/master port already has a listener')")
    # The public wrapper validates configured sockets before calling create;
    # keep the original historical checks as additional exclusions for rollback.
    n='tools/portable_pair.py'
    source=payload[n].decode();start=source.index('\ndef tunnel_command(');end=source.index('\ndef watch(',start)
    source=source[:start]+source[end:]
    source=once(source,"caller(config,1,'health')","caller(config,0,'health')")
    source=once(source,"float(node.PROFILE['gpu-memory-utilization']),kv_cap_bytes_per_rank=1610612736",
        "config['serving']['gpu_memory_utilization'],kv_cap_bytes_per_rank=config['serving']['kv_cap_mib']*2**20")
    source=once(source,"parser.add_argument('--no-tunnel',action='store_true',default=True,help='Deprecated compatibility option; LAN API uses no tunnel')", "parser.add_argument('--no-tunnel',action='store_true',default=True,help='Compatibility flag; no tunnels are implemented')") if "default=True,help='Deprecated" in source else source
    start=source.index('                if not args.no_tunnel:');end=source.index('            def interrupted(',start)
    source=source[:start]+source[end:]
    payload[n]=source.encode()
    # Include all new sources in attestation, then propagate source hashes.
    n='serving/spark_backend_attestation.py';source=payload[n].decode();tree=ast.parse(source)
    assignment=next(x for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PRIVATE_SOURCES' for t in x.targets))
    pins=ast.literal_eval(assignment.value);pins['ds41/launch_profile.py']=sha(profile)
    pins['ds41/startup_headroom_override.py']=sha(override)
    payload[n]=once(source,ast.get_source_segment(source,assignment),'PRIVATE_SOURCES = '+repr(pins)).encode()
    histories={n:{info['sha256']} for n,info in old['files'].items() if n.endswith('.py')}
    for _ in range(24):
        for n,h in histories.items():h.add(sha(payload[n]))
        replacements={before:sha(payload[n]) for n,h in histories.items() for before in h if before!=sha(payload[n])}
        changed=False
        for n,raw in list(payload.items()):
            if not n.endswith('.py') or not n.startswith(('serving/','tools/')):continue
            for before,after in replacements.items():raw=raw.replace(before.encode(),after.encode())
            if raw!=payload[n]:payload[n]=raw;changed=True
        if not changed:break
    else:raise RuntimeError('Source pin graph failed to settle')
    requirements=json.loads(payload['runtime-requirements.json'])
    requirements['loaded_backend_verification']['sha256']=sha(payload['serving/spark_backend_attestation.py'])
    payload['runtime-requirements.json']=encoded(requirements)
    payload['serving/overlay-manifest.json']=encoded({n.removeprefix('serving/'):sha(raw) for n,raw in payload.items() if n.startswith('serving/') and n!='serving/overlay-manifest.json'})
    payload['scripts/prepare_recipe_runtime.py']=Path(__file__).read_bytes()
    for n,raw in payload.items():
        if n.endswith('.py'):compile(raw,n,'exec')
    manifest=dict(format=old['format'],standalone_runtime=False,clean_rebuild_qualified=False,
        publication_approved=False,variant='full_cooperative_public_recipe',parent_manifest_sha256=PARENT_SHA,
        serving_qualified=False,files={n:dict(bytes=len(raw),sha256=sha(raw)) for n,raw in sorted(payload.items())})
    for n,raw in payload.items():
        path=target/n;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    raw=encoded(manifest);(target/'bundle-manifest.json').write_bytes(raw);verify(target,sha(raw))
    config=json.loads((ROOT/'reports/ds41-release-v82-deployment.json').read_bytes())
    sys.path.insert(0,str(ROOT));from ds41.launch_profile import DEFAULTS
    config.update(run_id=f'ds41-release-v{a.release}',kit_manifest_sha256=sha(raw),serving=DEFAULTS,
        api=dict(host='0.0.0.0',port=8888,master_port=29541,model_name='deepseek-v41-flash-exl3'),
        startup_memory_override=False)
    for node in config['nodes']:
        node['kit']=str(target);node['draft']=str(ROOT/'artifacts/ds41-draft-exl3-3bpw-sparse-v3')
    deployment.write_bytes(encoded(config))
    print(json.dumps(dict(kit=str(target),manifest_sha256=sha(raw),deployment=str(deployment),gpu_qualified=False)),flush=True)

if __name__=='__main__':main()
