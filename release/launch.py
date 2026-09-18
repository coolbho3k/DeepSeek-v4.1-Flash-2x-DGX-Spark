#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Portable two-Spark launcher: Docker + SSH, no system services or tunnels."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import urllib.request

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'.state/public'
from config import load


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True)+'\n').encode()


def run(argv, host=None, **kwargs):
    if host:
        argv = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                host, shlex.join([str(x) for x in argv])]
    return subprocess.run(argv, check=True, **kwargs)


def json_run(argv, host=None, **kwargs):
    return json.loads(run(argv, host, stdout=subprocess.PIPE, **kwargs).stdout)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def process_identity(pid):
    """Never signal a recycled PID. Linux /proc checks are CPU-only."""
    try:
        base = Path('/proc')/str(pid)
        stat = (base/'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return dict(pid=pid, start_ticks=stat[19], uid=base.stat().st_uid,
                    argv=(base/'cmdline').read_bytes().decode().split('\0')[:-1])
    except (FileNotFoundError, ProcessLookupError):
        return None


def read_state():
    path = STATE/'current.json'
    if not path.exists():
        return None
    state = json.loads(path.read_bytes())
    config_path = Path(state['deployment'])
    raw = config_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != state['deployment_sha256']:
        raise ValueError('Saved deployment changed; refusing an ambiguous lifecycle operation')
    state['config'] = json.loads(raw)
    return state


def node_action(state, index, action):
    config = state['config']
    n = config['nodes'][index]
    return json_run(['python3', '-B', str(Path(n['kit'])/'tools/portable_node.py'),
                    '--config-stdin', '--node', str(index), '--action', action], n['ssh'],
                    input=encoded(config), timeout=660)


def controller_alive(state):
    return bool(state.get('controller') and process_identity(state['controller']['pid']) == state['controller'])


def stop(state):
    if not state:
        print('No server owned by this public launcher. Other servers are not touched.')
        return
    if controller_alive(state):
        os.kill(state['controller']['pid'], signal.SIGTERM)
        deadline = time.monotonic()+120
        while controller_alive(state) and time.monotonic() < deadline:
            time.sleep(1)
    # Resolve exact random-label-owned containers, never stop by a broad name.
    # The controller may have failed before creating either worker.
    for index, n in enumerate(state['config']['nodes']):
        directory = str(Path(n['runs'])/state['config']['run_id']/f'node{index}')
        created = json_run(['python3', '-c',
            'import json,pathlib,sys; print(json.dumps((pathlib.Path(sys.argv[1])/"created.json").is_file()))',
            directory], n['ssh'])
        if created:
            observed = node_action(state, index, 'stop')
            if observed['state']['Running']:
                raise RuntimeError('Owned worker did not stop; inspect the same container')
    if controller_alive(state):
        raise RuntimeError('Workers stopped but controller is still exiting; inspect controller logs')
    print('Stopped only this launcher\'s recorded pair. Models, caches and logs were preserved.')


def remote_home(worker):
    return json_run(['python3', '-c', 'import json,pathlib; print(json.dumps(str(pathlib.Path.home())))'], worker)


def host_info(worker, card):
    return json_run(['python3', '-c',
        'import json,os,sys; print(json.dumps(dict(uid=os.getuid(),gid=os.getgid(),drm_gid=os.stat(sys.argv[1]).st_gid)))',
        card], worker)


def prepare_existing(settings):
    """Reuse explicitly pinned local assets; never adopt or restart an old run."""
    values = settings['values']
    source = Path(values['EXISTING_DEPLOYMENT'])
    if source.resolve() != source or not source.is_file() or source.stat().st_size > 65536:
        raise ValueError('Existing deployment must be a small unredirected JSON file')
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != values['EXISTING_DEPLOYMENT_SHA256']:
        raise ValueError('Existing deployment differs from its configured SHA256')
    config = json.loads(raw)
    kit = Path(config['nodes'][0]['kit'])
    # Verify the entire explicit kit with the checkout's independent checker
    # BEFORE importing a helper from it. Asset/runtime pins come from this
    # opt-in deployment, not from the fresh-download recipe lock.
    checker = module(ROOT/'release/runtime/verify.py', 'existing_kit_check')
    checker.verify(kit, config['kit_manifest_sha256'])
    sys.path.insert(0, str(kit/'tools'))
    node = module(kit/'tools/portable_node.py', 'existing_node')
    node.validate_config(config)
    if config['nodes'][1]['ssh'] != settings['worker']:
        raise ValueError('Existing assets belong to a different worker; refusing to retarget them')
    config.update(run_id='ds41-release-v'+str(time.time_ns()),
                  api=settings['api'], serving=settings['serving'],
                  fabric_network=settings['fabric_network'],
                  startup_memory_override=settings['startup_memory_override'])
    for i, n in enumerate(config['nodes']):
        n.update(settings['rails'][i][0])
        n['rails'] = settings['rails'][i]
        n['drm_card'] = values[('HEAD' if i == 0 else 'WORKER')+'_DRM_CARD']
        n.update(host_info(n['ssh'], n['drm_card']))
    node.validate_config(config)
    # Read-only checks on BOTH hosts, including exact image identities,
    # weight receipts, runtime/cache pins, fabric, free RAM and idle GPUs.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda i: node_action({'config': config}, i, 'preflight'), (0, 1)))
    if any(row.get('status') != 'portable_node_preflight_pass' for row in results):
        raise ValueError('Existing asset preflight did not pass on both hosts')
    path = STATE/config['run_id']/'deployment.json'
    path.parent.mkdir(parents=True, exist_ok=False)
    path.write_bytes(encoded(config))
    (path.parent/'reuse-preflight.json').write_bytes(encoded(dict(
        source=str(source), source_sha256=values['EXISTING_DEPLOYMENT_SHA256'],
        nodes=results, downloaded=False, containers_started=False)))
    print('Reusing verified existing assets on both hosts; no downloads or model copies.', flush=True)
    return path, config


def prepare(settings, lock):
    if lock.get('runtime') is None:
        raise ValueError('Prebuilt runtime publication is pending. No local build fallback is allowed.')
    worker = settings['worker']
    values = settings['values']
    # Check both hosts before downloading/hashing hundreds of GiB or copying anything.
    for host in (None, worker):
        jobs = run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                   host, stdout=subprocess.PIPE, timeout=30).stdout.strip()
        if jobs:
            raise ValueError(f'GPU busy on {host or "head"}; preparation/start refused. Existing server was not touched.')
    if values['EXISTING_DEPLOYMENT']:
        return prepare_existing(settings)
    home = remote_home(worker)
    remote_cache = values['REMOTE_CACHE_DIR'] or home+'/.cache/ds41'
    remote_dir = values['REMOTE_DIR'] or home+'/.cache/ds41/recipe'
    # Version the remote recipe so a future update cannot mutate a running kit.
    code_digest = hashlib.sha256((ROOT/'recipe-lock.json').read_bytes()+(ROOT/'release/bootstrap.py').read_bytes()+(ROOT/'release/registry.py').read_bytes()).hexdigest()[:16]
    remote_dir = str(Path(remote_dir)/code_digest)
    run(['mkdir', '-p', remote_dir+'/release'], worker)
    run(['rsync', '-a', '--protect-args', str(ROOT/'release/runtime')+'/',
         worker+':'+remote_dir+'/release/runtime/'])
    for source, target in ((ROOT/'recipe-lock.json', remote_dir+'/recipe-lock.json'),
                           (ROOT/'release/bootstrap.py', remote_dir+'/release/bootstrap.py'),
                           (ROOT/'release/registry.py', remote_dir+'/release/registry.py')):
        # argv is shell-quoted and contents travel over stdin, not a shell heredoc.
        run(['python3', '-c',
             'import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.write_bytes(sys.stdin.buffer.read())', target],
            worker, input=source.read_bytes())
    bootstrap = module(ROOT/'release/bootstrap.py', 'public_bootstrap')
    def peer_prepare():
        run(['python3', '-B', remote_dir+'/release/bootstrap.py', '--lock', remote_dir+'/recipe-lock.json',
             '--cache-dir', remote_cache, '--kit', remote_dir+'/release/runtime'], worker)
        return json_run(['python3', '-c', 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).read_text())',
                         remote_cache+'/prepared.json'], worker)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(peer_prepare)
        head = bootstrap.bootstrap(lock, Path(settings['cache']), ROOT/'release/runtime')
        peer = future.result()
    for i, n in enumerate((head, peer)):
        n.pop('lock_sha256', None)
        n['ssh'] = None if i == 0 else worker
        n.update(settings['rails'][i][0])
        n['rails'] = settings['rails'][i]
        n['drm_card'] = values[('HEAD' if i == 0 else 'WORKER')+'_DRM_CARD']
        n.update(host_info(n['ssh'], n['drm_card']))
    config = dict(format='ds41_two_spark_deployment_v1', run_id='ds41-release-v'+str(time.time_ns()),
        kit_manifest_sha256=lock['kit_manifest_sha256'], model_manifest_sha256=lock['model']['manifest_sha256'],
        cache_manifest_sha256=lock['runtime']['cache_manifest_sha256'], fabric_network=settings['fabric_network'],
        nodes=[head, peer], serving=settings['serving'], api=settings['api'],
        startup_memory_override=settings['startup_memory_override'])
    sys.path.insert(0, str(Path(head['kit'])/'tools'))
    node = module(Path(head['kit'])/'tools/portable_node.py', 'portable_node')
    node.validate_config(config)
    path = STATE/config['run_id']/'deployment.json'
    path.parent.mkdir(parents=True)
    path.write_bytes(encoded(config))
    return path, config


def start(path, config, no_wait=False):
    kit = Path(config['nodes'][0]['kit'])
    log = path.parent/'controller.log'
    argv = [sys.executable, '-B', str(kit/'tools/portable_pair.py'), '--config', str(path), '--execute']
    with log.open('ab', buffering=0) as output:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                                 start_new_session=True, env={**{k:v for k,v in os.environ.items() if k not in ('HF_TOKEN_WRITE','HF_TOKEN','HUGGING_FACE_HUB_TOKEN')}, 'PYTHONUNBUFFERED':'1'})
    identity = process_identity(child.pid)
    if identity is None:
        raise RuntimeError('Controller exited immediately; see '+str(log))
    state = dict(deployment=str(path), deployment_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 controller=identity, log=str(log))
    (STATE/'current.json').write_bytes(encoded(state))
    print('Starting the pair. Controller log: '+str(log), flush=True)
    print('No system service was installed. Use ./start-server.sh logs or status.', flush=True)
    if no_wait:
        return
    ready = Path(config['nodes'][0]['runs'])/config['run_id']/'pair/health-ready.json'
    try:
        for tick in range(720):
            if child.poll() is not None:
                raise RuntimeError('Startup ended; inspect ./start-server.sh logs --controller')
            if ready.exists():
                host = config['api']['host']
                print(f"Ready: http://{host if host!='0.0.0.0' else '<head-LAN-IP>'}:{config['api']['port']}/v1", flush=True)
                return
            if tick % 6 == 0:
                print('Waiting for model load, kernel-cache checks and CUDA graph capture…', flush=True)
            time.sleep(5)
        print('Startup observation timed out; the same controller remains running. Inspect logs; do not launch another copy.')
    except KeyboardInterrupt:
        print('\nStopped waiting only. The server/controller remain running; use ./start-server.sh stop to stop them.')


def status(state):
    if not state:
        print('No deployment owned by this launcher. Historical/other servers are not adopted.')
        return
    print('Controller: '+('running' if controller_alive(state) else 'not running'))
    for i in (0, 1):
        try:
            row = node_action(state, i, 'inspect')
            print(json.dumps(dict(node=i, container=row['container'], status=row['state']['Status'],
                mem_available_gib=round(row['memory']['MemAvailable']/2**30, 2))))
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            print(f'Node {i}: not yet inspectable ({type(e).__name__}); see controller log.')


def logs(state, args):
    if not state:
        raise ValueError('No deployment owned by this launcher')
    if args.controller:
        command = ['tail', '-n', '80']+(['-f'] if not args.no_follow else [])+[state['log']]
        run(command)
    else:
        row = node_action(state, args.node, 'inspect')
        run(['docker', 'logs', '--tail', '80']+(['-f'] if not args.no_follow else [])+[row['container']],
            state['config']['nodes'][args.node]['ssh'])


def doctor(settings):
    """Read-only host inspection. Does not run containers or allocate GPU memory."""
    for i, host in enumerate((None, settings['worker'])):
        prefix = 'HEAD' if i == 0 else 'WORKER'
        print('\n'+prefix+' (read-only)', flush=True)
        run(['python3', '-c', '''import json,os,pathlib,platform,shutil,sys
p=pathlib.Path
def read(x):
 try:return p(x).read_text().strip()
 except OSError:return 'unavailable'
card=p(sys.argv[1])
print(json.dumps(dict(architecture=platform.machine(),python=platform.python_version(),uid=os.getuid(),
 tools={x:bool(shutil.which(x)) for x in ('docker','nvidia-smi','rsync','ip','ss','ssh')},
 modeset=read('/sys/module/nvidia_drm/parameters/modeset'),fbdev=read('/sys/module/nvidia_drm/parameters/fbdev'),
 drm_card=str(card),drm_present=card.is_char_device(),
 memory=[x for x in read('/proc/meminfo').splitlines() if x.startswith(('MemTotal:','MemAvailable:'))]),indent=2))''',
            settings['values'][prefix+'_DRM_CARD']], host)
        run(['docker', 'version', '--format', '{{.Server.Version}}'], host)
        run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], host)
        for rail in settings['rails'][i]:
            run(['ip', '-brief', 'address', 'show', rail['ifname']], host)
    print('\nDoctor made no changes. Startup separately requires idle GPUs, matching RoCE GIDs and display settings.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='start', choices=('start','prepare','status','logs','stop','doctor'))
    parser.add_argument('--dry-run', action='store_true', help='Print resolved settings only; no SSH, downloads, or GPU work')
    parser.add_argument('--restart', action='store_true', help='Stop only this public launcher\'s own pair, then start')
    parser.add_argument('--no-wait', action='store_true', help='Return after spawning the background controller')
    parser.add_argument('--controller', action='store_true', help='Show controller/watchdog log instead of inference log')
    parser.add_argument('--node', type=int, choices=(0,1), default=0)
    parser.add_argument('--no-follow', action='store_true')
    for flag, typ in (('port',int),('host',str),('worker',str),('gpu-memory-utilization',float),
                      ('max-model-len',int),('max-num-seqs',int),('max-num-batched-tokens',int),
                      ('long-prefill-token-threshold',int)):
        parser.add_argument('--'+flag, type=typ)
    args = parser.parse_args()
    if args.restart and args.action != 'start':
        parser.error('--restart is only valid for start')
    overrides = {k.upper():v for k,v in vars(args).items() if k in
                 ('gpu_memory_utilization','max_model_len','max_num_seqs','max_num_batched_tokens','long_prefill_token_threshold')}
    overrides.update(API_PORT=args.port, API_HOST=args.host, WORKER_HOST=args.worker)
    if args.action in ('status','logs','stop'):
        if args.dry_run:
            print(json.dumps(dict(action=args.action, changed=False))); return
        state = read_state()
        if args.action == 'status':status(state)
        elif args.action == 'logs':logs(state,args)
        else:
            STATE.mkdir(parents=True,exist_ok=True)
            with (STATE/'operation.lock').open('a') as lockfile:
                fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
                stop(read_state())
        return
    settings = load(ROOT, overrides)
    lock = json.loads((ROOT/'recipe-lock.json').read_bytes())
    if args.dry_run:
        assets = (dict(mode='reuse_existing', deployment=settings['values']['EXISTING_DEPLOYMENT'],
                       sha256=settings['values']['EXISTING_DEPLOYMENT_SHA256'])
                  if settings['values']['EXISTING_DEPLOYMENT'] else lock)
        print(json.dumps(dict(settings=settings, assets=assets, gpu_started=False, changed=False),indent=2)); return
    if args.action == 'doctor':doctor(settings); return
    STATE.mkdir(parents=True,exist_ok=True)
    with (STATE/'operation.lock').open('a') as lockfile:
        fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state = read_state()
        if args.restart:stop(state)
        elif state and controller_alive(state):
            raise ValueError('An owned server is running. Use status/logs; changing settings requires an explicit restart.')
        path, config = prepare(settings, lock)
        if args.action == 'prepare':
            print('Assets prepared; no server started. Run ./start-server.sh to serve.')
        else:start(path, config, args.no_wait)


if __name__ == '__main__':
    try:main()
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
        print('Error: '+str(error),file=sys.stderr);sys.exit(1)
