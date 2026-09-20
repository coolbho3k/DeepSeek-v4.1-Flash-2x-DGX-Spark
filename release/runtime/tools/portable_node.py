"""One host of the release-only two-Spark deployment, with no workspace imports.

Default action is a command plan. Create/start are explicit, one-shot actions.
Never pulls/builds images, hashes weight payloads, raises limits, or deletes.
The pair controller owns simultaneous start and continuous RAM supervision.
"""
import argparse
import hashlib
import importlib.util
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

sys.dont_write_bytecode = True  # Never add __pycache__ to a verified public kit.

GIB = 2**30
STOP_AVAILABLE = 512*2**20
NODE_RANKS = (0, 1)
AOT_NAME = 'mxfp8_gemm_cutlass_sm120'
AOT_TARGET = '/usr/local/lib/python3.12/dist-packages/flashinfer/data/aot/'+AOT_NAME
AOT_SHA = '6abdf60fb353da15d87030427e16a982e7819a6f479563b8acfde81110e78bf4'
IMAGE_SHA = '7c467b82c1e1f9022a2683cffc27a387bb155115dbd08781bdc3ddd1806017fb'
CACHE_ROOTS = {'flashinfer', 'tilelang', 'triton', 'torch-extensions', 'cuda', 'vllm',
               'exllamav3', 'numba', 'torch'}
BOUND_SHARDS = {f'model-{i:05d}-of-00051.safetensors' for i in range(1,52)}
BOUND_SHARDS |= {f'engrams/engram-layer-{i:02d}.safetensors' for i in (1,14)}
BOUND_SHARDS |= {f'draft/model-{i:05d}-of-00002.safetensors' for i in (1,2)}
PROFILE = {'tensor-parallel-size': '2', 'decode-context-parallel-size': '2', 'gpu-memory-utilization': '0.92', 'max-model-len': '1048576', 'max-num-seqs': '2', 'max-num-batched-tokens': '3072', 'long-prefill-token-threshold': '2816', 'enable-chunked-prefill': 'true', 'disable-chunked-mm-input': 'true', 'enforce-eager': 'false', 'load-format': 'safetensors', 'safetensors-load-strategy': 'lazy', 'kv-cache-dtype': 'fp8_ds_mla', 'compilation-config': '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 3, 4, 6, 8, 9, 12, 15, 16, 18, 20, 24], "max_cudagraph_capture_size": 24}', 'speculative-config': '{"method": "dspark", "model": "/model/draft", "num_speculative_tokens": 3, "quantization": "fp8", "enforce_eager": false, "enable_adaptive_verification": false}', 'profiler-config': '{"profiler": "torch", "torch_profiler_dir": "/cache/ds41-native-profile-v51", "torch_profiler_with_stack": false, "torch_profiler_record_shapes": false, "torch_profiler_with_memory": false, "torch_profiler_with_flops": false, "torch_profiler_use_gzip": true, "torch_profiler_dump_cuda_time_total": true, "capture_torch_profiler": false, "detailed_trace_annotation": false, "ignore_frontend": true, "delay_iterations": 2, "max_iterations": 1, "warmup_iterations": 0}'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def json_bytes(raw, limit=4*2**20):
    if len(raw) > limit:
        raise ValueError('Oversized configuration/receipt')
    def reject(value):
        raise ValueError('Nonfinite JSON value')
    return json.loads(raw, object_pairs_hook=unique, parse_constant=reject)


def read_small(path, limit=4*2**20):
    if path.resolve() != path or not path.is_file() or path.stat().st_size > limit:
        raise ValueError('Expected small unredirected input: '+str(path))
    return path.read_bytes()


def absolute(name):
    if (not isinstance(name, str) or not name.startswith('/')
            or any(c in name for c in ('\0','\n','\r',','))
            or '..' in PurePosixPath(name).parts or str(PurePosixPath(name)) != name
            or len(PurePosixPath(name).parts) < 3):
        raise ValueError('Use an explicit normalized task path, not a root directory')
    return Path(name)


def validate_config(config):
    keys = {'format','run_id','kit_manifest_sha256','model_manifest_sha256',
            'cache_manifest_sha256','fabric_network','nodes','serving','api','startup_memory_override'}
    if not isinstance(config, dict) or set(config) != keys or config['format'] != 'ds41_two_spark_deployment_v1':
        raise ValueError('Unsupported deployment configuration')
    if not re.fullmatch(r'ds41-release-v[1-9][0-9]*', config['run_id']):
        raise ValueError('Use a fresh ds41-release-vN run ID')
    for key in ('kit_manifest_sha256','model_manifest_sha256','cache_manifest_sha256'):
        if not isinstance(config[key], str) or not re.fullmatch('[0-9a-f]{64}', config[key]):
            raise ValueError('Independent public manifest digests are required')
    from launch_profile import validate as validate_profile
    validate_profile(config['serving'])
    if type(config['startup_memory_override']) is not bool or (config['startup_memory_override'] and config['serving']['gpu_memory_utilization']!=.92):
        raise ValueError('Explicit boolean startup override is scoped to0.92')
    api = config['api']
    if set(api) != {'host','port','master_port','model_name'}:
        raise ValueError('Unknown API settings')
    ipaddress.IPv4Address(api['host'])
    if any(type(api[k]) is not int or not 1024 <= api[k] <= 65535 for k in ('port','master_port')) or api['port']==api['master_port']:
        raise ValueError('Invalid serving ports')
    if not re.fullmatch('[a-zA-Z0-9_.-]+',api['model_name']):
        raise ValueError('Invalid served model name')
    network = ipaddress.IPv4Network(config['fabric_network'], strict=True)
    nodes = config['nodes']
    if not isinstance(nodes, list) or len(nodes) != 2:
        raise ValueError('Exactly two physical Spark hosts are required')
    node_keys = {'ssh','kit','model','model_receipt','image','cache','runs',
                 'fabric_ip','ifname','hca','gid_index','uid','gid','draft','rails','drm_card','drm_gid','engram'}
    for index, node in enumerate(nodes):
        if (not isinstance(node, dict) or not node_keys <= set(node)
                or set(node)-node_keys not in (set(),{'model_bindings'})):
            raise ValueError('Unsupported host configuration')
        if ((index == 0 and node['ssh'] is not None) or (index == 1 and
                (not isinstance(node['ssh'], str) or not re.fullmatch(
                 r'(?:[a-zA-Z0-9_][a-zA-Z0-9_.-]*@)?[a-zA-Z0-9][a-zA-Z0-9.-]*', node['ssh'])))):
            raise ValueError('Host0 must be local; host1 must be a plain SSH endpoint')
        if not isinstance(node['image'], str) or not re.fullmatch('sha256:[0-9a-f]{64}', node['image']):
            raise ValueError('Use an immutable installed image ID on each host')
        paths = {key:absolute(node[key]) for key in ('kit','model','model_receipt','cache','runs','draft')}
        from engram_assets import check_reference
        packed_root = check_reference(node['engram'], index)
        for value in paths.values():
            if packed_root.is_relative_to(value) or value.is_relative_to(packed_root):
                raise ValueError('Packed Engrams must remain separate from other input/output trees')
        for key in ('kit','model','cache'):
            if paths['runs'].is_relative_to(paths[key]) or paths[key].is_relative_to(paths['runs']):
                raise ValueError('Run outputs must be separate from immutable input trees')
        if paths['model_receipt'].is_relative_to(paths['model']):
            raise ValueError('The local weight-verification receipt must remain private')
        if 'model_bindings' in node:
            bindings = node['model_bindings']
            if (not isinstance(bindings,dict) or set(bindings) != BOUND_SHARDS
                    or len(set(bindings.values())) != len(bindings)):
                raise ValueError('Bound views require all55 distinct public shard sources')
            for source in bindings.values():
                source = absolute(source)
                if len(source.parts) < 4 or any(source.is_relative_to(paths[key])
                        for key in ('model','runs','kit','cache')):
                    raise ValueError('Original bound shards must be separate from mutable/staged trees')
        address = ipaddress.IPv4Address(node['fabric_ip'])
        if address not in network or address in (network.network_address, network.broadcast_address):
            raise ValueError('Each fabric IP must be a host address in the configured subnet')
        for key in ('ifname','hca'):
            if not isinstance(node[key], str) or not re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_.-]{0,31}', node[key]):
                raise ValueError('Invalid network device name')
        if type(node['gid_index']) is not int or not 0 <= node['gid_index'] <= 255:
            raise ValueError('Invalid RoCE GID index')
        if any(type(node[k]) is not int or not 0 < node[k] < 2**31 for k in ('uid','gid')):
            raise ValueError('Use explicit non-root host UID/GID')
    if nodes[0]['fabric_ip'] == nodes[1]['fabric_ip']:
        raise ValueError('The two Spark fabric addresses must differ')
    return config


def run_dir(config, index):
    return absolute(config['nodes'][index]['runs'])/config['run_id']/f'node{index}'


def docker_command(config, index, owner=None):
    validate_config(config)
    node = config['nodes'][index]
    rank = NODE_RANKS[index]
    directory = run_dir(config, index)
    env = dict(DS41_ENABLE_COOPERATIVE_MOE='1', DS41_DRAFT_EXL3_PATH='/draft-exl3', CUDA_VISIBLE_DEVICES='0', VLLM_PLUGINS='ds41', DS41_ENABLE_DCP2='1',
        DS41_SSD_ENGRAM_SOURCE='/model/engrams', DS41_ENGRAM_CACHE_MIB='64',
        PYTHONPATH='/opt/ds41-serving:/opt/ds41-dcp-v3:/opt/exllamav3',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HOME='/cache/hf',
        VLLM_NO_USAGE_STATS='1', VLLM_USE_V2_MODEL_RUNNER='1', DS41_ENABLE_DSPARK='1', DS41_ENABLE_SSD_VOCAB='1', DSV41_IO_THREADS='96', VLLM_USE_BREAKABLE_CUDAGRAPH='0',
        VLLM_CACHE_ROOT='/cache/vllm', TILELANG_CACHE_DIR='/cache/tilelang',
        TRITON_CACHE_DIR='/cache/triton', TORCH_EXTENSIONS_DIR='/cache/torch-extensions',
        CUDA_CACHE_PATH='/cache/cuda', XDG_CACHE_HOME='/cache',
        FLASHINFER_WORKSPACE_BASE='/cache/flashinfer', TMPDIR='/cache/tmp',
        VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_ENGINE_READY_TIMEOUT_S='3600',
        MAX_JOBS='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        TORCH_CUDA_ARCH_LIST='12.1a', FLASHINFER_CUDA_ARCH_LIST='12.1a',
        NCCL_NET='IB', NCCL_IB_DISABLE='0', NCCL_IB_HCA=node['hca'],
        NCCL_IB_GID_INDEX=str(node['gid_index']), NCCL_IB_ROCE_VERSION_NUM='2',
        NCCL_IB_ADDR_FAMILY='AF_INET', NCCL_IB_ADDR_RANGE=config['fabric_network'],
        NCCL_SOCKET_IFNAME=node['ifname'], GLOO_SOCKET_IFNAME=node['ifname'],
        VLLM_HOST_IP=node['fabric_ip'],
        TP_SOCKET_IFNAME=node['ifname'], NCCL_NVLS_ENABLE='0', NCCL_CROSS_NIC='0',
        NCCL_IB_MERGE_NICS='0', NCCL_CUMEM_ENABLE='0', NCCL_IGNORE_CPU_AFFINITY='1',
        NCCL_DEBUG='INFO', NCCL_MAX_CTAS='8', NCCL_BUFFSIZE='1048576', NCCL_LL128_BUFFSIZE='262144', NCCL_PROTO='^LL128', NCCL_MAX_NCHANNELS='8', TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    from launch_profile import environment as profile_environment
    env.update(profile_environment(config['serving']))
    env.update(MALLOC_ARENA_MAX='2',MALLOC_TRIM_THRESHOLD_='131072',VLLM_SPARSE_INDEXER_MAX_LOGITS_MB='128')
    env['NVIDIA_DRIVER_CAPABILITIES']='compute,utility,graphics,display'
    env['DS41_ALLOW_STARTUP_MEMORY_SHORTFALL']='1' if config['startup_memory_override'] else '0'
    mounts = [(node['draft'],'/draft-exl3',True), (node['model'],'/model',True), (str(directory/'overlay'),'/opt/ds41-serving',True),
        (str(directory/'cache'),'/cache',False),
        (str(absolute(node['kit'])/'aot'/AOT_NAME),AOT_TARGET,True)]
    mounts += [(source,'/model/'+name,True) for name,source
               in sorted(node.get('model_bindings',{}).items())]
    from engram_assets import mounts as packed_mounts
    mounts += packed_mounts(node['engram'], index)
    args = ['docker','create','--name',f"{config['run_id']}-rank{rank}",
        '--restart=no','--pull=never','--runtime=runc','--gpus=all','--network=host',
        '--ipc=private','--cgroupns=private','--memory=9g','--memory-swap=9g','--cpus=6',
        '--shm-size=2g','--pids-limit=512','--read-only','--cap-drop=ALL','--cap-add=IPC_LOCK',
        '--security-opt=no-new-privileges','--device=/dev/infiniband','--ulimit=memlock=-1:-1',
        '--user',f"{node['uid']}:{node['gid']}"]
    # Expose only the DRM card and its device group, never elevate the worker.
    args += ['--device='+node['drm_card']+':/dev/dri/card0','--group-add',str(node['drm_gid'])]
    if owner is not None:
        if not isinstance(owner,str) or not re.fullmatch('[0-9a-f]{32}',owner):
            raise ValueError('A random recorded owner token is required')
        args += ['--label','ds41.release-owner='+owner]
    for source, target, readonly in mounts:
        args += ['--mount',f'type=bind,src={source},dst={target}'+(',readonly' if readonly else '')]
    args += ['--tmpfs=/tmp:rw,nosuid,nodev,size=512m','--workdir=/cache',
             '--entrypoint=/opt/ds41-venv/bin/python']
    for key, value in env.items():
        args += ['--env',key+'='+value]
    tail = ['/opt/ds41-serving/serve.py','serve','/model','--config','/opt/ds41-serving/conservative.yaml',
        '--worker-cls','combined_worker.CombinedWorker','--distributed-executor-backend','mp',
        '--disable-custom-all-reduce','--nnodes','2','--node-rank',str(rank),
        '--master-addr',config['nodes'][0]['fabric_ip'],'--master-port',str(config['api']['master_port']),
        '--host',config['api']['host'],'--port',str(config['api']['port']),'--served-model-name',config['api']['model_name'],
        '--enable-auto-tool-choice','--tool-call-parser','deepseek_v41',
        '--reasoning-parser','deepseek_v41','--enable-prompt-tokens-details']
    for key,value in config['serving'].items():
        if key != 'kv_cap_mib': tail += ['--'+key.replace('_','-'),str(value)]
    if rank == 1:
        tail += ['--headless']
    return dict(command=args+[node['image']]+tail, cmd=tail, env=env, mounts=mounts)


def command(args, timeout=30):
    return subprocess.check_output(args, text=True, stderr=subprocess.PIPE, timeout=timeout)


def memory():
    return {line.split(':')[0]:int(line.split()[1])*1024
        for line in Path('/proc/meminfo').read_text().splitlines()
        if line.split(':')[0] in ('MemTotal','MemFree','MemAvailable')}


def sample_start(node):
    base = Path('/sys/class/infiniband')/node['hca']/'ports/1'
    return dict(memory=memory(),
        gpu_processes=command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip(),
        ib_state=(base/'state').read_text().strip(), gid=(base/'gids'/str(node['gid_index'])).read_text().strip(),
        sockets=command(['ss','-ltnH']),
        addresses=json.loads(command(['ip','-j','address','show','dev',node['ifname']])),
        uid=os.getuid(), gid_number=os.getgid())


def validate_start_sample(node, sample):
    if sample['gpu_processes']:
        raise ValueError('GPU is busy; refuse another deployment without changing the existing job')
    mem = sample['memory']
    if (set(mem) != {'MemTotal','MemFree','MemAvailable'}
            or any(type(value) is not int for value in mem.values())
            or not 120*GIB <= mem['MemTotal'] <= 128*GIB
            or not (0 <= mem['MemFree'] <= mem['MemTotal'] and 0 <= mem['MemAvailable'] <= mem['MemTotal'])):
        raise ValueError('Expected complete Spark unified-memory accounting')
    required = int(.89*mem['MemTotal'])
    if (sample['gpu_processes'] or mem['MemFree'] < required-GIB
            or mem['MemAvailable'] < required+2*GIB):
        raise ValueError('GPU must be idle with the 0.89 ceiling with unchanged startup RAM reserves')
    addresses = {row.get('local') for interface in sample['addresses'] for row in interface.get('addr_info',[])
                 if row.get('family') == 'inet'}
    if ('ACTIVE' not in sample['ib_state'] or node['fabric_ip'] not in addresses
            or ipaddress.IPv6Address(sample['gid']).ipv4_mapped != ipaddress.IPv4Address(node['fabric_ip'])
            or (sample['uid'],sample['gid_number']) != (node['uid'],node['gid'])):
        raise ValueError('Actual host user/interface/RoCE GID differs from deployment')
    # preflight checks configured API/master ports, not campaign-specific ports.


def module(kit, name, filename, manifest):
    raw = read_small(kit/filename)
    if sha(raw) != manifest['files'][filename]['sha256']:
        raise ValueError('Public runtime helper changed')
    spec = importlib.util.spec_from_file_location(name, kit/filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def cache_inputs(source, expected_sha):
    raw = read_small(source/'cache-manifest.json')
    manifest = json_bytes(raw)
    if sha(raw) != expected_sha or manifest.get('format') != 'ds41_auxiliary_runtime_cache_v1':
        raise ValueError('Expected an independently pinned public auxiliary cache')
    files = manifest.get('files')
    if not isinstance(files, dict) or not 1 <= len(files) <= 100000:
        raise ValueError('Invalid cache inventory')
    total = 0
    for name, row in files.items():
        p = PurePosixPath(name)
        if (not isinstance(name,str) or str(p) != name or p.is_absolute() or '..' in p.parts
                or not p.parts or p.parts[0] not in CACHE_ROOTS or '\\' in name or '\0' in name
                or name.endswith('.lock') or p.name == '.ninja_lock'
                or type(row.get('bytes')) is not int or not 0 <= row['bytes'] <= 512*2**20
                or not re.fullmatch('[0-9a-f]{64}',row.get('sha256',''))):
            raise ValueError('Unsafe or unsupported public cache entry')
        path = source/name
        if path.resolve() != path or not path.is_file() or path.stat().st_size != row['bytes']:
            raise ValueError('Public cache file missing/redirected/resized')
        if 'mtime_ns' in row and (type(row['mtime_ns']) is not int
                or not 0 <= row['mtime_ns'] < 2**63 or path.stat().st_mtime_ns != row['mtime_ns']):
            raise ValueError('Public cache mtime differs from the qualified compiler inputs')
        total += row['bytes']
    if total > 8*GIB:
        raise ValueError('Public auxiliary cache exceeds the bounded staging budget')
    return manifest


def check_model_receipt(config,index,manifest):
    node = config['nodes'][index]
    kit = absolute(node['kit'])
    verifier = ('tools/verify_public_download.py' if config['model_manifest_sha256'] in {'043785e20066f6212d30b3451a956802596a18d0b542b104ce2dd24bba900bc3', '6d79a9ae5cfd121df7c559b76cde87b50551adbe68ae0dc94c97973e85e8e1d1'} else 'tools/verify_downloaded_release.py')
    weights = module(kit,'portable_weights_check',verifier,manifest)
    model = absolute(node['model'])
    public,summary = weights.load_manifest(model/'release-manifest.json',config['model_manifest_sha256'])
    receipt,_ = weights.small_json(absolute(node['model_receipt']))
    if 'model_bindings' in node:
        bound = module(kit,'portable_bound_weights_check','tools/verify_mapped_release.py',manifest)
        return bound.check_receipt(model,public,summary,node['model_bindings'],receipt)
    return weights.check_receipt(model,public,summary,receipt)


def preflight(config, index):
    node = config['nodes'][index]
    sample = sample_start(node)
    validate_start_sample(node, sample)  # Before touching model/cache inputs.
    ports={str(config['api']['port']),str(config['api']['master_port'])}
    if any(line.split()[3].rsplit(':',1)[-1] in ports for line in sample['sockets'].splitlines()):
        raise ValueError('Configured serving/master port already has a listener')
    for key in ('kit','model','cache','runs'):
        path = absolute(node[key])
        if path.resolve() != path or not path.is_dir():
            raise ValueError('Missing or redirected host input/run root: '+key)
    if run_dir(config,index).exists() or run_dir(config,index).is_symlink():
        raise ValueError('Preserve the existing run; use inspect, never create/start again')
    kit = absolute(node['kit'])
    manifest_raw = read_small(kit/'bundle-manifest.json')
    if sha(manifest_raw) != config['kit_manifest_sha256']:
        raise ValueError('Runtime kit differs from the independent public digest')
    manifest = json_bytes(manifest_raw)
    checker = module(kit,'portable_bundle_check','verify.py',manifest)
    checked = checker.verify(kit,config['kit_manifest_sha256'])
    if sha(Path(__file__).read_bytes()) != manifest['files']['tools/portable_node.py']['sha256']:
        raise ValueError('Node launcher differs from the verified public kit')
    images = module(kit,'portable_image_check','tools/verify_runtime_image.py',manifest)
    image = images.verify(images.inspect(node['image']), images.read_identity(kit/'runtime-image-identity.json',IMAGE_SHA))
    weight_check = check_model_receipt(config,index,manifest)
    packed_check = module(kit,'public_packed_engram_check','tools/engram_assets.py',manifest)
    packed_check.validate(node['engram'],index)
    profile = read_small(kit/'serving/conservative.yaml').decode()
    pairs = [line.split(':',1) for line in profile.splitlines() if line.strip() and not line.lstrip().startswith('#')]
    if len(pairs) != len(PROFILE) or {k.strip():v.strip() for k,v in pairs} != PROFILE:
        raise ValueError('Public kit changed the qualified serving limits')
    if sha(read_small(kit/'aot'/AOT_NAME/(AOT_NAME+'.so'))) != AOT_SHA:
        raise ValueError('Unqualified MXFP8 AOT payload')
    cache = cache_inputs(absolute(node['cache']),config['cache_manifest_sha256'])
    cache_bytes = sum(row['bytes'] for row in cache['files'].values())
    if shutil.disk_usage(node['runs']).free < 32*GIB + cache_bytes + 64*2**20:
        raise ValueError('Staging must preserve the32GiB disk reserve')
    return dict(status='portable_node_preflight_pass', sample=sample, kit=checked, image=image,
                weights=weight_check, cache_files=len(cache['files']), cache_bytes=cache_bytes)


def staged_files(directory):
    result = {}
    for part in ('overlay','cache'):
        for base, directories, names in os.walk(directory/part,followlinks=False):
            for name in directories:
                p = Path(base)/name
                if p.resolve() != p:
                    raise ValueError('Redirected staged directory')
            for name in names:
                p = Path(base)/name
                info = p.lstat()
                if p.resolve() != p or not p.is_file():
                    raise ValueError('Nonregular staged file')
                result[p.relative_to(directory).as_posix()] = [info.st_dev,info.st_ino,
                    info.st_mode,info.st_size,info.st_mtime_ns,info.st_ctime_ns,info.st_nlink]
    return result


def exclusive(path, value):
    with path.open('xb') as stream:
        stream.write(encoded(value))


def create(config, index):
    checks = preflight(config,index)
    node = config['nodes'][index]
    directory = run_dir(config,index)
    directory.mkdir(parents=True,exist_ok=False)
    owner = secrets.token_hex(16)
    contract = docker_command(config,index,owner)
    record = dict(config_sha256=sha(encoded(config)), node=index, contract=contract,owner=owner,
        launcher_sha256=sha(Path(__file__).read_bytes()))
    exclusive(directory/'create-request.json',record)
    exclusive(directory/'preflight.json',checks)
    kit = absolute(node['kit'])
    shutil.copytree(kit/'serving',directory/'overlay',copy_function=shutil.copy2)
    kit_manifest = json_bytes(read_small(kit/'bundle-manifest.json'))
    for name,row in kit_manifest['files'].items():
        if name.startswith('serving/') and sha(read_small(directory/'overlay'/name.removeprefix('serving/'))) != row['sha256']:
            raise ValueError('Copied serving snapshot differs from verified public kit')
    cache = directory/'cache'
    cache.mkdir()
    for part in sorted(CACHE_ROOTS | {'tmp','hf'}):
        (cache/part).mkdir()
    source = absolute(node['cache'])
    manifest = cache_inputs(source,config['cache_manifest_sha256'])
    for name, row in manifest['files'].items():
        target = cache/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/name,target)
        if 'mtime_ns' in row and target.stat().st_mtime_ns != row['mtime_ns']:
            raise ValueError('Staged compiler cache mtime changed')
        with target.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest() != row['sha256']:
                raise ValueError('Staged cache hash mismatch; preserve failed run')
    exclusive(directory/'staged-inputs.json',staged_files(directory))
    # Staging can populate page cache. Do not flush it globally or relax RAM
    # checks; start rechecks the original free/available reserves on both hosts.
    exclusive(directory/'docker-create-attempt.json',dict(owner=owner,automatic_retries=False))
    cid = command(contract['command']).strip()
    if not re.fullmatch('[0-9a-f]{64}',cid):
        raise ValueError('Docker create returned an invalid container ID')
    exclusive(directory/'created.json',dict(container=cid,config_sha256=record['config_sha256']))
    return dict(status='portable_node_created_not_started',container=cid,node=index)


def validate_owned(config,index,record,created,node):
    cid = created['container']
    if (not re.fullmatch('[0-9a-f]{64}',cid) or node['Id'] != cid
            or record['config_sha256'] != sha(encoded(config))
            or created['config_sha256'] != record['config_sha256'] or record['node'] != index
            or record['launcher_sha256'] != sha(Path(__file__).read_bytes())):
        raise ValueError('Container ownership or immutable controller inputs changed')
    expected = docker_command(config,index,record['owner'])
    # JSON records tuples as lists; compare canonical serialization.
    if encoded(record['contract']) != encoded(expected):
        raise ValueError('Recorded Docker contract changed')
    actual = node['HostConfig']
    env = dict(value.split('=',1) for value in node['Config']['Env'])
    binds = sorted((m['Source'],m['Destination'],not m['RW']) for m in node['Mounts'] if m['Type']=='bind')
    if (node['Name'] != f"/{config['run_id']}-rank{NODE_RANKS[index]}"
            or node['Config']['Image'] != config['nodes'][index]['image']
            or node['Config'].get('Labels',{}).get('ds41.release-owner') != record['owner']
            or node['Config']['Cmd'] != expected['cmd']
            or node['Config']['Entrypoint'] != ['/opt/ds41-venv/bin/python']
            or node['Config']['User'] != f"{config['nodes'][index]['uid']}:{config['nodes'][index]['gid']}"
            or any(env.get(k) != value for k,value in expected['env'].items())
            or binds != sorted(expected['mounts'])
            or actual['Memory'] != 9*GIB or actual['MemorySwap'] != 9*GIB
            or actual['NanoCpus'] != 6*10**9 or not actual['ReadonlyRootfs']
            or actual['ShmSize'] != 2*GIB or actual['PidsLimit'] != 512
            or actual['NetworkMode'] != 'host' or actual['IpcMode'] != 'private'
            or actual['CgroupnsMode'] != 'private' or actual['Privileged']
            or actual['CapAdd'] != ['CAP_IPC_LOCK'] or actual['CapDrop'] != ['ALL']
            or actual['SecurityOpt'] != ['no-new-privileges']
            or actual['Runtime'] != 'runc' or actual['RestartPolicy']['Name'] != 'no'):
        raise ValueError('Actual container differs from the qualified Docker/resource contract')


def inspect_owned(config,index):
    directory = run_dir(config,index)
    record = json_bytes(read_small(directory/'create-request.json'))
    created = json_bytes(read_small(directory/'created.json'))
    cid = created['container']
    if not re.fullmatch('[0-9a-f]{64}',cid):
        raise ValueError('Invalid exact container identity')
    nodes = json.loads(command(['docker','inspect',cid]))
    if len(nodes) != 1:
        raise ValueError('Expected one exact container observation')
    validate_owned(config,index,record,created,nodes[0])
    return dict(status='portable_owned_container_observed',container=cid,node=index,
                state=nodes[0]['State'],memory=memory())


def startup_log_summary(text):
    """Return fixed hints only, never prompt text or arbitrary log contents."""
    if 'register display IO:' in text and 'CUDA_ERROR_INVALID_VALUE' in text:
        return dict(last_observed_stage='display_io_registration_failed', hint=
            'CUDA rejected registration of the DRM mapping. Check the loaded driver '
            'against qualified 580.173.02; see docs/display-memory.md. '
            'Do not increase GPU utilization to address this error.')
    if any(marker in text for marker in ('Starting to load model',
            'Loading safetensors checkpoint shards', 'Model loading took')):
        return dict(last_observed_stage='model_load_or_later', hint=
            'Model loading was observed; initial distributed setup progressed. '
            'Check both inference logs for load/profile/KV/warmup progress.')
    if "Using ['PYNCCL'] all-reduce backends" in text:
        return dict(last_observed_stage='tp_communicator_selected', hint=
            'If both ranks remain here before model loading, inspect the vLLM '
            'ZeroMQ control handshake and TCP reachability between their primary '
            'fabric IPs, including dynamic ports. This is not proof of an NCCL '
            'failure; an RDMA bandwidth test does not test this TCP path.')
    return dict(last_observed_stage='unknown', hint=
        'No recognized startup marker in the bounded log tail. '
        'Inspect both inference logs; absence of a marker does not identify a hang.')


def startup_diagnostics(config,index):
    observed = inspect_owned(config,index)
    # Python/vLLM and NCCL can log on different streams; inspect both, but
    # never copy arbitrary log text into the controller's diagnostic journal.
    tail = subprocess.check_output(
        ['docker','logs','--tail','120',observed['container']],
        text=True, stderr=subprocess.STDOUT, timeout=5)
    return dict(node=index,container=observed['container'],
                control_ip=config['nodes'][index]['fabric_ip'],
                **startup_log_summary(tail))


def recover_created(config,index):
    """Resolve only an already-attempted create; never creates/starts/restarts."""
    directory = run_dir(config,index)
    if (directory/'created.json').exists():
        return inspect_owned(config,index)
    record = json_bytes(read_small(directory/'create-request.json'))
    attempted = json_bytes(read_small(directory/'docker-create-attempt.json'))
    if attempted != dict(owner=record['owner'],automatic_retries=False):
        raise ValueError('No matching original Docker-create attempt')
    nodes = json.loads(command(['docker','inspect',f"{config['run_id']}-rank{NODE_RANKS[index]}"]))
    if len(nodes) != 1:
        raise ValueError('Expected the originally named container')
    created = dict(container=nodes[0]['Id'],config_sha256=record['config_sha256'])
    validate_owned(config,index,record,created,nodes[0])
    exclusive(directory/'created.json',created)
    return inspect_owned(config,index)


def start(config,index):
    observed = inspect_owned(config,index)
    directory = run_dir(config,index)
    if observed['state']['Status'] != 'created' or (directory/'start-attempt.json').exists():
        raise ValueError('Start is one-shot: inspect the same container after an ambiguous result')
    sample = sample_start(config['nodes'][index])
    validate_start_sample(config['nodes'][index],sample)
    # The real warm cache has12,957 files. Its explicit fingerprint inventory
    # is larger than small configuration files; retain a separate bounded limit.
    if staged_files(directory) != json_bytes(read_small(directory/'staged-inputs.json',16*2**20),16*2**20):
        raise ValueError('Staged runtime/cache changed before start')
    # Recheck public model fingerprints and AOT after staging, without another
    # weight payload pass. No original author/workspace receipt is consulted.
    node = config['nodes'][index]
    kit = absolute(node['kit'])
    raw = read_small(kit/'bundle-manifest.json')
    if sha(raw) != config['kit_manifest_sha256']:
        raise ValueError('Public runtime kit changed before start')
    manifest = json_bytes(raw)
    check_model_receipt(config,index,manifest)
    packed_check = module(kit,'public_start_packed_engram_check','tools/engram_assets.py',manifest)
    packed_check.validate(node['engram'],index)
    if sha(read_small(kit/'aot'/AOT_NAME/(AOT_NAME+'.so'))) != AOT_SHA:
        raise ValueError('Public AOT library changed before start')
    exclusive(directory/'start-attempt.json',dict(container=observed['container'],sample=sample))
    result = command(['docker','start',observed['container']],timeout=45).strip()
    if result != observed['container']:
        raise ValueError('Unexpected start acknowledgement; inspect the same container')
    return inspect_owned(config,index)


def stop(config,index):
    observed = inspect_owned(config,index)
    if observed['state']['Running']:
        command(['docker','stop','--time','20',observed['container']],timeout=45)
    # The acknowledgement alone is not terminal evidence.
    result = inspect_owned(config,index)
    if result['state']['Running']:
        raise ValueError('Stop not yet observed terminal; inspect the same container')
    return result


def health(config,index):
    observed = inspect_owned(config,index)
    if index != 0 or not observed['state']['Running']:
        return dict(healthy=False)
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(f"http://{('127.0.0.1' if config['api']['host']=='0.0.0.0' else config['api']['host'])}:{config['api']['port']}/health",timeout=2) as response:
            healthy = response.status == 200
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(f"http://{('127.0.0.1' if config['api']['host']=='0.0.0.0' else config['api']['host'])}:{config['api']['port']}/v1/models",timeout=2) as response:
            models = json.loads(response.read(1024*1024))
        return dict(healthy=healthy and any(row.get('id') == config['api']['model_name']
            for row in models.get('data',[])))
    except (urllib.error.URLError,TimeoutError,ConnectionError):
        return dict(healthy=False)


def aot_maps(config,index):
    observed = inspect_owned(config,index)
    kit = absolute(config['nodes'][index]['kit'])
    manifest_raw = read_small(kit/'bundle-manifest.json')
    if sha(manifest_raw) != config['kit_manifest_sha256']:
        raise ValueError('Backend verification requires the owned kit manifest')
    manifest = json_bytes(manifest_raw)
    requirements_raw = read_small(kit/'runtime-requirements.json')
    if sha(requirements_raw) != manifest['files']['runtime-requirements.json']['sha256']:
        raise ValueError('Runtime backend requirements changed')
    policy = json_bytes(requirements_raw).get('loaded_backend_verification')
    if policy is not None:
        name = 'serving/spark_backend_attestation.py'
        digest = manifest['files'][name]['sha256']
        if policy != dict(format='ds41_loaded_combined_miaai_v9', inspector=name, sha256=digest):
            raise ValueError('Unreviewed loaded-backend verification policy')
        if sha(read_small(kit/name)) != digest:
            raise ValueError('Backend inspector differs from owned kit')
        result = json_bytes(command(['docker','exec',observed['container'],
            '/opt/ds41-venv/bin/python','-I','-S','/opt/ds41-serving/spark_backend_attestation.py',
            '--rank',str(NODE_RANKS[index]),'--source-sha',digest]).encode())
        if (result.get('status') != 'qualified_loaded_combined_miaai_with_native_draft_graphs'
                or result.get('attention') != {'implementation': 'image_safe_packed_sparse_attention_v2', 'kernel_sha256': '22839ccef76d501a9191b193a522429dcfe98ae9977f31f51be04455ea51d9e9', 'registration_sha256': 'e61a6b984fc351cf5147603035913b17fa4c99d66e654861a124118a04320a36', 'query_dtype': 'bfloat16', 'probability_bf16_terms': 2, 'partial_dtype': 'float32', 'lse_base': 2, 'main_cache_format': 'fp4', 'swa_cache_format': 'fp8', 'image_visibility_unchanged': True}
                or result.get('image_prefix') != {'implementation': 'whole_image_prefix_v1', 'bootstrap_sha256': '4f3049b68b0fbdec933c281d0303f2702af729818a94b1af84dbaf0807642ab5', 'override_sha256': 'dd2b90570f6f42a70ce3b2b997e0027d98a6baa75889b441ae5c60c6443173aa', 'partial_image_prefix_hits': False, 'complete_image_prefix_hits_preserved': True, 'original_image_pixels_preserved': True}
                or result.get('sparse_mapping') != {'implementation': 'stable_fused_dcp2_sparse_slots_v1', 'kernel_sha256': 'acbd5dce12e3a988697268c946f7c1a178cc38a3dc738dbb5a94287b7cc43edb', 'stable_candidate_order': True, 'duplicate_candidates_preserved': True, 'synchronous_bounds_checks': True, 'image_key_membership_unchanged': True, 'maximum_rows': 512, 'maximum_width': 8192, 'persistent_gpu_workspace_bytes': 0}
                or result.get('native_engram') != {'implementation': 'miaai_parallel_native_engram_v1', 'license': 'AGPL-3.0-only', 'core_sha256': '79e771e79820c439478ccb51187b329639eb88e2555d31eed4ade9987cd324e7', 'maximum_chunk_tokens': 256, 'maximum_local_heads': 144, 'staging_bytes_per_layer_ceiling': 19759104, 'cache_bytes_per_layer_ceiling': 67108864, 'io_threads': 96, 'resident_tables': False, 'resident_scales': False, 'native_image_hasher_unchanged': True, 'unowned_and_dead_ids_zero': True, 'callback_stream_ordering': True, 'full_model_graph_capture_enabled': True, 'live_tables': 2, 'staging_bounds_verified': True, 'layout': 'page15', 'gpu_readable_host': True, 'deferred_retrieval': True, 'row_bytes_unchanged': True, 'reader_abi': 2}
                or result.get('grouped_prefill') != {'implementation': 'miaai_grouped_prefill_ds41_v1', 'license': 'AGPL-3.0-only', 'kernel_sha256': '35f11df05fc5b870128db1d521513618a9f7a2ca05953b19a6c1d1194230c097', 'dispatcher_sha256': 'e09cde421ec879a9e6e34aca2cdd6219f437d78d9cf6512e6dccf4a4f57ae839', 'abi': 1003, 'minimum_expert_rows': 16, 'maximum_tokens': 2048, 'maximum_assignments': 12288, 'shared_workspace_bytes': 280173312, 'workspace_allocated_on_first_large_forward': True, 'small_decode_math_unchanged': True, 'device_only_routing': True, 'pre_down_fp32_routing': True, 'fp16_rounding_boundaries_preserved': True}
                or result.get('dcp_communication') != {'implementation': 'fused_dcp2_communication_v1', 'license': 'AGPL-3.0-only', 'kernel_sha256': '7aa4a5e6d978f4be72db26e65619e75c7c09e75a218426afbf4a8aab1d56ce20', 'maximum_rows': 512, 'maximum_sparse_width': 8192, 'stable_sparse_partition': True, 'duplicate_entries_preserved': True, 'packed_output_lse_collective': True, 'per_chunk_collectives': 2, 'sink_exchange_unchanged': True, 'query_exchange_unchanged': True, 'partial_dtype': 'float32', 'lse_base': 2, 'original_image_visibility': True, 'synchronous_cache_bounds_checks': True, 'persistent_gpu_workspace_bytes': 0, 'maximum_packed_send_bytes': 33619968, 'full_model_graph_capture_enabled': False}
                or result.get('combined_ready', {}).get('format') != 'ds41_combined_ready_v1'
                or result.get('rank') != NODE_RANKS[index]
                or result.get('counts') != dict(packed_wo_a_layers=40, b12x_layers=210)):
            raise ValueError('Loaded backend verification failed')
        return dict(result,node=index)
    code = '''import json,pathlib,resource
resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20))
resource.setrlimit(resource.RLIMIT_CPU,(15,15))
rows=[]
for path in pathlib.Path('/proc').glob('[0-9]*'):
 try: lines=(path/'maps').read_text().splitlines()
 except (FileNotFoundError,ProcessLookupError,PermissionError): continue
 libraries=sorted({line.split()[-1] for line in lines if 'mxfp8_gemm_cutlass_sm120.so' in line})
 if libraries: rows.append(dict(pid=int(path.name),libraries=libraries))
print(json.dumps(rows))'''
    rows = json.loads(command(['docker','exec',observed['container'],
        '/opt/ds41-venv/bin/python','-I','-S','-c',code]))
    target = AOT_TARGET+'/'+AOT_NAME+'.so'
    if not rows or any(row['libraries'] != [target] for row in rows):
        raise ValueError('Worker did not exclusively map the qualified AOT library')
    return dict(status='qualified_aot_exclusively_mapped',node=index,processes=rows,library_sha256=AOT_SHA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--config',type=Path)
    source.add_argument('--config-stdin',action='store_true')
    parser.add_argument('--node',type=int,choices=(0,1),required=True)
    parser.add_argument('--action',choices=('plan','preflight','create','recover-created','start','inspect','stop','health','aot','startup-diagnostics'),default='plan')
    args = parser.parse_args()
    raw = sys.stdin.buffer.read(65537) if args.config_stdin else read_small(args.config.absolute())
    if len(raw) > 65536:
        raise ValueError('Deployment configuration must be at most64KiB')
    config = validate_config(json_bytes(raw))
    if args.action == 'plan':
        result = dict(status='portable_node_command_plan',node=args.node,
            resources_checked=False,assets_checked=False,created=False,**docker_command(config,args.node))
    else:
        functions = dict(preflight=preflight,create=create,start=start,inspect=inspect_owned,
                         stop=stop,health=health,aot=aot_maps)
        functions['recover-created'] = recover_created
        functions['startup-diagnostics'] = startup_diagnostics
        result = functions[args.action](config,args.node)
    print(json.dumps(result,sort_keys=True),flush=True)



# Public transport configuration. No network or driver changes are made.
_public_validate_config = validate_config
_public_sample_start = sample_start
_public_validate_start_sample = validate_start_sample
_public_docker_command = docker_command

def validate_config(config):
    _public_validate_config(config)
    network=ipaddress.IPv4Network(config['fabric_network'])
    addresses=[]
    for node in config['nodes']:
        if not isinstance(node['rails'],list) or len(node['rails']) not in (1,2):
            raise ValueError('Configure one or two RoCE rails on each host')
        if node['rails'][0]!={k:node[k] for k in ('fabric_ip','ifname','hca','gid_index')}:
            raise ValueError('First rail must match primary fabric settings')
        if not re.fullmatch(r'/dev/dri/card[0-9]+',node['drm_card']) or type(node['drm_gid']) is not int or node['drm_gid']<0:
            raise ValueError('Invalid DRM device/group')
        for rail in node['rails']:
            if set(rail)!={'fabric_ip','ifname','hca','gid_index'}:raise ValueError('Unknown rail configuration')
            address=ipaddress.IPv4Address(rail['fabric_ip'])
            if address not in network or address in (network.network_address,network.broadcast_address):
                raise ValueError('Rail IP outside fabric subnet')
            addresses.append(address)
            if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,31}',rail[k]) for k in ('hca','ifname')):
                raise ValueError('Invalid rail interface/HCA')
            if type(rail['gid_index']) is not int or not 0<=rail['gid_index']<=255:raise ValueError('Invalid GID index')
    if len(set(addresses))!=len(addresses) or len(config['nodes'][0]['rails'])!=len(config['nodes'][1]['rails']):
        raise ValueError('Rails require unique addresses and matching counts')
    return config

def display_flags(node):
    names = ('modeset', 'fbdev')
    paths = ['/sys/module/nvidia_drm/parameters/'+name for name in names]
    try:
        values = [Path(path).read_text().strip() for path in paths]
    except PermissionError:
        # Some NVIDIA drivers expose these harmless status flags as root-only.
        # Use the already-installed image, with no host mounts, GPU, network,
        # capabilities or writable rootfs. Never prompt for sudo or alter flags.
        if not re.fullmatch('sha256:[0-9a-f]{64}', node['image']):
            raise ValueError('Display inspection requires an immutable installed image')
        values = command(['docker', 'run', '--rm', '--pull=never', '--runtime=runc',
            '--network=none', '--read-only', '--cap-drop=ALL',
            '--security-opt=no-new-privileges', '--memory=32m', '--memory-swap=32m',
            '--pids-limit=16', '--user=0:0', '--entrypoint=/bin/cat',
            node['image'], *paths], timeout=30).splitlines()
    if len(values) != 2 or any(value not in ('Y', 'N') for value in values):
        raise ValueError('Invalid NVIDIA display-mode flag response')
    return dict(zip(names, values))


def sample_start(node):
    from display_driver import observe
    driver = observe()
    sample=_public_sample_start(node)
    sample['driver']=driver
    sample['rails']=[]
    for rail in node['rails']:
        base=Path('/sys/class/infiniband')/rail['hca']/'ports/1'
        sample['rails'].append(dict(ib_state=(base/'state').read_text().strip(),
            gid=(base/'gids'/str(rail['gid_index'])).read_text().strip(),
            gid_type=(base/'gid_attrs/types'/str(rail['gid_index'])).read_text().strip(),
            addresses=json.loads(command(['ip','-j','address','show','dev',rail['ifname']]))))
    sample['display']=dict(**display_flags(node),
        card_exists=Path(node['drm_card']).is_char_device(),card_gid=os.stat(node['drm_card']).st_gid)
    return sample

def validate_start_sample(node,sample):
    from display_driver import validate
    validate(sample.get('driver'), node['ssh'] or 'head')
    _public_validate_start_sample(node,sample)
    if sample['display']!=dict(modeset='Y',fbdev='N',card_exists=True,card_gid=node['drm_gid']):
        raise ValueError('Display KV needs nvidia_drm modeset=1 fbdev=0 and the configured DRM card; see docs/display-memory.md. No settings were changed.')
    if len(sample['rails'])!=len(node['rails']):raise ValueError('Incomplete rail observations')
    for expected,rail in zip(node['rails'],sample['rails']):
        address=ipaddress.IPv4Address(expected['fabric_ip'])
        addresses={r.get('local') for interface in rail['addresses']
            for r in interface.get('addr_info',[]) if r.get('family')=='inet'}
        if ('ACTIVE' not in rail['ib_state'] or rail['gid_type']!='RoCE v2'
                or str(address) not in addresses or ipaddress.IPv6Address(rail['gid']).ipv4_mapped!=address):
            raise ValueError('RoCE link/address/GID mismatch for '+expected['ifname'])

def docker_command(config,index,owner=None):
    result=_public_docker_command(config,index,owner)
    rails=config['nodes'][index]['rails']
    updates=dict(NCCL_IB_HCA='='+','.join(r['hca']+':1' for r in rails),
        NCCL_IB_MERGE_NICS='1' if len(rails)==2 else '0')
    for key,value in updates.items():
        old=key+'='+result['env'][key]
        positions=[i for i,item in enumerate(result['command'])
            if i and result['command'][i-1]=='--env' and item==old]
        if len(positions)!=1:raise ValueError('Ambiguous network environment')
        result['command'][positions[0]]=key+'='+value
        result['env'][key]=value
    return result

if __name__ == '__main__':
    main()
