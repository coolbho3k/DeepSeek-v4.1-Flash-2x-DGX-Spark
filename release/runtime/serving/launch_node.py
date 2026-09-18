"""Create (but do not start) one node of the first TP2/DCP2 serving pair.

Default mode prints the exact command. --create requires completed source
verification, and physical host0 additionally requires its hashed mirror.
Start both returned container IDs together after fresh host-memory checks.
Containers have no restart policy and are retained for diagnosis.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path('/home/emi/code/ds41')
MODEL = ROOT / 'artifacts/ds41-exl3-3bpw-candidate-v1'
OVERLAY = ROOT / 'artifacts/serving-overlay-v12'
NODE_RANKS = (1, 0)  # Physical host0=local, host1=dgx1; API/coordinator on dgx1.
MASTER_ADDR = '10.100.32.2'
MANIFEST_SHA = '7f7cfee4a7dc618196b0699a57034dce3316b83257a7320687d079d72d13ee73'
IMAGES = (
    'sha256:a5ef1cecb16259d16e49578c334b05f60dc58873eeac94cc4a29c5c246d0bcbf',
    'sha256:314893911de4009c4e989408d17d48ec0f050bcd972e9683adf02e6cc38a8e58',
)
KERNEL_NAME = 'ds41_moe_mul1_v1.so'
KERNEL_SOURCE = ROOT/'artifacts/exl3-moe-mul1-build-v1'/KERNEL_NAME
KERNEL_SHA256 = '66de4aa31e49462fd00fb4b26ebd4e16dfd995b631157e3b0194c27fc1239342'
MXFP8_AOT_SHA256 = '6abdf60fb353da15d87030427e16a982e7819a6f479563b8acfde81110e78bf4'
MXFP8_AOT_PROBE_SHA256 = 'd9e6f8c37ae68be7200c3b692f71a6f3f1d246b225ca722a445c1c9eb08b7af1'
MXFP8_AOT_MODULE = 'mxfp8_gemm_cutlass_sm120'
MXFP8_AOT_TARGET = '/usr/local/lib/python3.12/dist-packages/flashinfer/data/aot/'+MXFP8_AOT_MODULE
OVERLAY_FILES = frozenset(('conservative.yaml', 'guarded_worker.py', 'streaming_loader.py',
                           'spark_topk.py', 'spark_moe.py', 'spark_fused_moe.py',
                           KERNEL_NAME, 'spark_kv_cap.py', 'serve.py', 'launch_node.py'))


def validate_overlay_path(overlay):
    if (overlay.resolve() != overlay or overlay.parent != ROOT/'artifacts'
            or not re.fullmatch(r'(?:serving-overlay-|ds41-serving-)[a-z0-9-]+', overlay.name)):
        raise ValueError('Use a canonical task-specific serving snapshot under artifacts')
    return overlay


def validate_aot_path(path):
    if (path.resolve()!=path or path.parent!=ROOT/'artifacts'
            or not re.fullmatch(r'ds41-mxfp8-aot-v[1-9][0-9]*',path.name)):
        raise ValueError('Use a canonical versioned MXFP8 AOT artifact')
    return path


def validate_aot_evidence(host,receipt,proof):
    if (receipt.get('status')!='mxfp8_debug_stripped_runtime_unchanged'
            or receipt.get('image_id')!=IMAGES[host] or receipt.get('library_sha256')!=MXFP8_AOT_SHA256
            or receipt.get('library_bytes')!=2279592 or receipt.get('source_preserved') is not True
            or proof.get('status')!='clean_mxfp8_aot_gpu_exact_pass'
            or proof.get('probe_sha256')!=MXFP8_AOT_PROBE_SHA256
            or proof.get('library_sha256')!=MXFP8_AOT_SHA256
            or proof.get('native_kernel')!='FlashInferCutlassMxfp8LinearKernel'
            or proof.get('flashinfer_jit_disabled') is not True
            or proof.get('jit_replacement_created') is not False
            or proof.get('allocator_fraction')!=.0075
            or not 0<proof.get('peak_allocated_bytes',0)<=512*2**20
            or proof.get('mapped_libraries')!=[MXFP8_AOT_TARGET+'/'+MXFP8_AOT_MODULE+'.so']):
        raise ValueError('Missing exact clean-build/native-loader GPU qualification')
    cases=proof.get('cases',[])
    expected={(m,n,k,s) for n,k in ((256,256),(5120,256),(1024,5120),(5120,5120))
              for m in (1,32,1056) for s in (.25,1.,8.)}
    if (len(cases)!=36 or {(c['shape'][0],c['shape'][1],c['shape'][2],c['scale']) for c in cases}!=expected
            or any(c['max_absolute_error']!=0 or c['dtype']!='torch.bfloat16' for c in cases)):
        raise ValueError('Incomplete fresh AOT numerical coverage')


def validate_mxfp8_aot(host,path):
    validate_aot_path(path)
    library=path/(MXFP8_AOT_MODULE+'.so')
    if library.resolve()!=library or hashlib.sha256(library.read_bytes()).hexdigest()!=MXFP8_AOT_SHA256:
        raise ValueError('MXFP8 AOT library changed')
    proof_path=ROOT/f'reports/mxfp8-clean-gpu-v1/host{host}/complete.json'
    receipt=json.loads((path/'receipt.json').read_text())
    proof=json.loads(proof_path.read_text());validate_aot_evidence(host,receipt,proof)
    return dict(path=str(path),library_sha256=MXFP8_AOT_SHA256,
                qualification_sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest())


def command(host_index, run_id, overlay=None, mxfp8_aot=None):
    overlay = validate_overlay_path(OVERLAY if overlay is None else overlay)
    rank = NODE_RANKS[host_index]
    cache = ROOT / 'artifacts' / (run_id + '-cache')
    env = dict(
        CUDA_VISIBLE_DEVICES='0', VLLM_PLUGINS='ds41', DS41_ENABLE_DCP2='1',
        DS41_SSD_ENGRAM_SOURCE='/model/engrams', DS41_ENGRAM_CACHE_MIB='64',
        PYTHONPATH='/opt/ds41-serving:/opt/ds41-dcp-v3:/opt/exllamav3',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HOME='/cache/hf',
        VLLM_NO_USAGE_STATS='1', VLLM_USE_V2_MODEL_RUNNER='0',
        VLLM_CACHE_ROOT='/cache/vllm', TILELANG_CACHE_DIR='/cache/tilelang',
        TRITON_CACHE_DIR='/cache/triton', TORCH_EXTENSIONS_DIR='/cache/torch-extensions',
        CUDA_CACHE_PATH='/cache/cuda', XDG_CACHE_HOME='/cache',
        FLASHINFER_WORKSPACE_BASE='/cache/flashinfer', TMPDIR='/cache/tmp',
        VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_ENGINE_READY_TIMEOUT_S='3600',
        MAX_JOBS='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        TORCH_CUDA_ARCH_LIST='12.1a', FLASHINFER_CUDA_ARCH_LIST='12.1a',
        NCCL_NET='IB', NCCL_IB_DISABLE='0', NCCL_IB_HCA='rocep1s0f1',
        NCCL_IB_GID_INDEX='3', NCCL_IB_ROCE_VERSION_NUM='2',
        NCCL_IB_ADDR_FAMILY='AF_INET', NCCL_IB_ADDR_RANGE='10.100.32.0/24',
        NCCL_SOCKET_IFNAME='enp1s0f1np1', GLOO_SOCKET_IFNAME='enp1s0f1np1',
        TP_SOCKET_IFNAME='enp1s0f1np1', NCCL_NVLS_ENABLE='0', NCCL_CROSS_NIC='0',
        NCCL_IB_MERGE_NICS='0', NCCL_CUMEM_ENABLE='0', NCCL_IGNORE_CPU_AFFINITY='1',
        NCCL_DEBUG='WARN', NCCL_MAX_CTAS='8', TORCH_NCCL_ASYNC_ERROR_HANDLING='1',
    )
    argv = ['docker', 'create', '--name', f'{run_id}-rank{rank}', '--restart=no', '--pull=never',
            '--runtime=runc', '--gpus=all', '--network=host', '--ipc=private',
            '--cgroupns=private', '--memory=8g', '--memory-swap=8g', '--cpus=6',
            '--shm-size=2g', '--pids-limit=512', '--read-only',
            '--cap-drop=ALL', '--cap-add=IPC_LOCK', '--security-opt=no-new-privileges',
            '--device=/dev/infiniband', '--ulimit=memlock=-1:-1',
            '--user', f'{os.getuid()}:{os.getgid()}',
            '--mount', f'type=bind,src={MODEL},dst=/model,readonly',
            '--mount', f'type=bind,src={overlay},dst=/opt/ds41-serving,readonly',
            '--mount', f'type=bind,src={cache},dst=/cache',
            '--tmpfs=/tmp:rw,nosuid,nodev,size=512m', '--workdir=/cache',
            '--entrypoint=/opt/ds41-venv/bin/python']
    if mxfp8_aot is not None:
        validate_aot_path(mxfp8_aot)
        argv+=['--mount',f'type=bind,src={mxfp8_aot},dst={MXFP8_AOT_TARGET},readonly']
    for key, value in env.items():
        argv += ['--env', f'{key}={value}']
    argv += [IMAGES[host_index], '/opt/ds41-serving/serve.py', 'serve', '/model',
             '--config', '/opt/ds41-serving/conservative.yaml',
             '--worker-cls', 'guarded_worker.InitialWorker',
             '--distributed-executor-backend', 'mp', '--disable-custom-all-reduce',
             '--nnodes', '2', '--node-rank', str(rank),
             '--master-addr', MASTER_ADDR, '--master-port', '29541',
             '--host', '127.0.0.1', '--port', '8041',
             '--served-model-name', 'deepseek-v41-flash-exl3']
    if rank == 1:
        argv += ['--headless']
    return argv, cache


def validate_inputs(host_index, overlay=None, mxfp8_aot=None):
    overlay = validate_overlay_path(OVERLAY if overlay is None else overlay)
    if mxfp8_aot is not None:validate_mxfp8_aot(host_index,mxfp8_aot)
    if MODEL.resolve() != MODEL:
        raise ValueError('Canonical model and serving overlay directories required')
    if hashlib.sha256((MODEL / 'package-manifest.json').read_bytes()).hexdigest() != MANIFEST_SHA:
        raise ValueError('Wrong candidate package')
    proof = json.loads((ROOT / 'reports/worker-hf-verify-v1/complete.json').read_text())
    if (proof.get('status') != 'worker_hf_candidate_independently_verified'
            or proof.get('package_manifest_sha256') != MANIFEST_SHA
            or proof.get('verification', {}).get('status') != 'source_anchored_inventory_verified'
            or proof.get('verification', {}).get('structure_only') is not False):
        raise ValueError('Full source-anchored package verification has not completed')
    if host_index == 0:
        mirror = json.loads((ROOT / 'reports/candidate-mirror-v1/verified.json').read_text())
        if mirror['status'] != 'complete_candidate_mirror_hash_verified' or mirror['manifest_sha256'] != MANIFEST_SHA:
            raise ValueError('Local package mirror is not hash verified')
        for name, row in mirror['files'].items():
            p = MODEL / name
            s = p.stat()
            stamp = [s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink]
            if p.resolve() != p or stamp != row['stamp']:
                raise ValueError(f'Local verified package changed: {name}')
    # The staged overlay is a snapshot, not a bind of the editable worktree.
    manifest = json.loads((overlay / 'overlay-manifest.json').read_text())
    if set(manifest) != OVERLAY_FILES:
        raise ValueError('Incomplete or unexpected serving overlay inventory')
    if manifest[KERNEL_NAME] != KERNEL_SHA256:
        raise ValueError('Unqualified native MUL1 kernel in serving snapshot')
    for name, digest in manifest.items():
        p = overlay / name
        if p.resolve() != p or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Staged serving overlay changed: {name}')
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host-index', type=int, choices=(0, 1), required=True,
                        help='Physical host: 0=local Spark, 1=dgx1; not the TP/node rank')
    parser.add_argument('--run-id', default='ds41-serving-initial-v1')
    parser.add_argument('--overlay', type=Path, default=OVERLAY)
    parser.add_argument('--mxfp8-aot',type=Path,help='Optional GPU-qualified read-only clean-build MXFP8 library')
    parser.add_argument('--create', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'ds41-serving-[a-z0-9-]+', args.run_id):
        raise ValueError('Expected a task-specific serving run ID')
    rank = NODE_RANKS[args.host_index]
    argv, cache = command(args.host_index, args.run_id, args.overlay,args.mxfp8_aot)
    if not args.create:
        print(shlex.join(argv))
        return
    validate_inputs(args.host_index, args.overlay,args.mxfp8_aot)
    journal = ROOT / 'reports' / args.run_id
    journal.mkdir(exist_ok=False)
    cache.mkdir(exist_ok=False)
    for directory in ('tmp', 'hf', 'vllm', 'tilelang', 'triton', 'torch-extensions', 'cuda', 'flashinfer'):
        (cache / directory).mkdir()
    with (journal / 'create-command.json').open('x') as f:
        json.dump(dict(host_index=args.host_index, node_rank=rank, command=argv, started=False), f, indent=2)
    cid = subprocess.check_output(argv, text=True).strip()
    if not re.fullmatch('[0-9a-f]{64}', cid):
        raise RuntimeError(f'Unexpected Docker create result: {cid}')
    inspection = json.loads(subprocess.check_output(['docker', 'inspect', cid], text=True))[0]
    with (journal / 'created.json').open('x') as f:
        json.dump(inspection, f, indent=2)
    print(json.dumps(dict(container=cid, host_index=args.host_index, node_rank=rank, started=False)), flush=True)


if __name__ == '__main__':
    main()
