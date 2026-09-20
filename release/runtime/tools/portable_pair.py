"""Release-only pair launcher with continuous exact-container RAM protection.

Default prints plans. --execute stages and starts one fresh pair, then remains
its foreground RAM watcher. Ctrl-C stops only that recorded pair. A transient
observation failure never triggers another create/start. --watch-existing
reattaches observation without starting anything. No image pulls/builds here.
"""
import argparse
import concurrent.futures
import datetime
import fcntl
import json
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import portable_node as node


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def node_command(config,index,action):
    host = config['nodes'][index]
    args = ['python3','-B',str(node.absolute(host['kit'])/'tools/portable_node.py'),
            '--config-stdin','--node',str(index),'--action',action]
    if index == 1:
        args = ['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',host['ssh'],shlex.join(args)]
    return args


def call(config,index,action):
    result = subprocess.run(node_command(config,index,action),input=node.encoded(config),
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        timeout=600 if action=='create' else (15 if action=='startup-diagnostics' else 65))
    if result.returncode:
        # Bounded diagnostic text; do not dump image inspect/environment data.
        raise RuntimeError(f'node{index} {action} failed: '+result.stderr.decode(errors='replace')[-3000:])
    return node.json_bytes(result.stdout)


def both(config,action,caller=call):
    def invoke(index):
        try:
            return dict(ok=True,result=caller(config,index,action))
        except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
            return dict(ok=False,error=f'{type(error).__name__}: {error}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        return list(pool.map(invoke,(0,1)))


def results(rows):
    if len(rows) != 2 or any(row.get('ok') is not True for row in rows):
        raise RuntimeError('Both node operations must succeed: '+json.dumps(rows))
    return [row['result'] for row in rows]


def stop_reason(observations,cids,started_at=None):
    if len(observations) != 2 or len(cids) != 2:
        raise ValueError('Two exact worker observations required')
    for index,(row,cid) in enumerate(zip(observations,cids)):
        if row['container'] != cid or row['node'] != index:
            raise ValueError('Observation does not match the owned container pair')
        state = row['state']
        if (not state['Running'] or state['OOMKilled'] or state.get('Paused')
                or state.get('Restarting') or state.get('Dead')):
            return f'node{index}_worker_stopped_or_unhealthy'
        if started_at and state['StartedAt'] != started_at[index]:
            return f'node{index}_unexpected_restart'
        available = row['memory']['MemAvailable']
        if type(available) is not int or available < 0:
            raise ValueError('Invalid host-memory observation')
        if available < node.STOP_AVAILABLE:
            return f'node{index}_available_below_{node.STOP_AVAILABLE}'
    return None


def watch(config,cids,record,caller=call,sleep=time.sleep,clock=time.monotonic,on_ready=lambda proof:None):
    beginning = clock()
    ready = False
    started_at = None
    pending_reason = None
    next_diagnostic = 180
    record(dict(time=now(),stage='startup_control_plane',
        nodes=[dict(node=i,control_ip=n['fabric_ip'],interface=n['ifname'])
               for i,n in enumerate(config['nodes'])],
        hint='VLLM_HOST_IP is pinned per rank. ZeroMQ also needs bidirectional TCP '
             'on dynamic ports over the primary fabric, not only NCCL/RoCE. '
             'No host networking settings were changed.'))
    while True:
        observed = both(config,'inspect',caller)
        if any(not row['ok'] for row in observed):
            record(dict(time=now(),stage='observation_retry_same_pair',observations=observed))
            sleep(10)
            continue
        rows = results(observed)
        reason = pending_reason or stop_reason(rows,cids,started_at)
        record(dict(time=now(),stage='continuous_ram_watch',observations=rows))
        if reason:
            pending_reason = reason
            stops = both(config,'stop',caller)
            terminal = all(row['ok'] and not row['result']['state']['Running'] for row in stops)
            if terminal:
                return dict(status='portable_pair_safety_stop',reason=reason,stops=stops,
                            all_workers_observed_terminal=True)
            record(dict(time=now(),stage='stop_retry_same_pair',reason=reason,stops=stops))
            sleep(10)
            continue
        if started_at is None:
            started_at = [row['state']['StartedAt'] for row in rows]
        if not ready:
            elapsed = clock()-beginning
            if elapsed > 3600:
                pending_reason = 'startup_deadline'
                continue
            try:
                healthy = caller(config,0,'health')['healthy']
            except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
                record(dict(time=now(),stage='health_observation_retry',error=repr(error)))
                healthy = False
            if healthy:
                maps = both(config,'aot',caller)
                if any(not row['ok'] for row in maps):
                    pending_reason = 'aot_check_failed'
                    record(dict(time=now(),stage='aot_check_failed',maps=maps))
                    continue
                proof = dict(time=now(),status='portable_pair_api_ready_ram_watch_continues',
                    containers=cids,started_at=started_at,aot=results(maps),
                    text_vision_generation_tested=False,utilization=config['serving']['gpu_memory_utilization'],ordinary_kv_cap_bytes_per_rank=config['serving']['kv_cap_mib']*2**20, external_display_bytes_per_rank=1879048192, kv_cap_bytes_per_rank=config['serving']['kv_cap_mib']*2**20+1879048192)
                record(proof)
                on_ready(proof)
                ready = True
            elif elapsed >= next_diagnostic:
                # Diagnostics are hints, never a new safety-stop condition.
                # Keep the existing one-hour readiness deadline and RAM floor.
                record(dict(time=now(),stage='startup_diagnostics',
                    elapsed_seconds=int(elapsed),
                    results=both(config,'startup-diagnostics',caller)))
                next_diagnostic = elapsed+300
        sleep(10)


def stop_until_terminal(config,record,caller=call,sleep=time.sleep):
    while True:
        stopped = both(config,'stop',caller)
        terminal = all(row['ok'] and not row['result']['state']['Running'] for row in stopped)
        record(dict(time=now(),stage='exact_pair_stop_observation',stops=stopped,terminal=terminal))
        if terminal:
            return stopped
        sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--execute',action='store_true')
    modes.add_argument('--watch-existing',action='store_true')
    parser.add_argument('--no-tunnel',action='store_true',default=True,help='Direct dgx1 LAN API; SSH forwarding is disabled')
    args = parser.parse_args()
    config = node.validate_config(node.json_bytes(node.read_small(args.config.absolute())))
    if not (args.execute or args.watch_existing):
        print(json.dumps(dict(status='portable_pair_plan_only',created=False,
            commands=[node.docker_command(config,index)['command'] for index in (0,1)],
            prerequisite_checks_performed=False,continuous_stop_available_bytes=node.STOP_AVAILABLE),indent=2))
        return
    kit = node.absolute(config['nodes'][0]['kit'])
    raw = node.read_small(kit/'bundle-manifest.json')
    if node.sha(raw) != config['kit_manifest_sha256']:
        raise ValueError('Public runtime kit changed')
    manifest = node.json_bytes(raw)
    for filename,path in (('tools/portable_pair.py',Path(__file__)),
                          ('tools/portable_node.py',Path(node.__file__))):
        if node.sha(path.read_bytes()) != manifest['files'][filename]['sha256']:
            raise ValueError('Executing launcher differs from verified public kit')
    directory = node.absolute(config['nodes'][0]['runs'])/config['run_id']/'pair'
    if args.execute:
        # A busy GPU or failed public prerequisite produces NO new run files.
        initial = results(both(config,'preflight'))
        directory.mkdir(parents=True,exist_ok=False)
        node.exclusive(directory/'preflight.json',initial)
        node.exclusive(directory/'deployment.json',config)
        node.exclusive(directory/'create-attempt.json',dict(time=now(),automatic_retries=False))
        created_raw = both(config,'create')
        node.exclusive(directory/'create-results.json',created_raw)
        if any(not row['ok'] for row in created_raw):
            # Resolve a lost create acknowledgement by its random owner label
            # and exact saved contract. This performs NO additional create.
            created_raw = both(config,'recover-created')
            node.exclusive(directory/'create-observation-recovery.json',created_raw)
        created = results(created_raw)
        cids = [row['container'] for row in created]
        node.exclusive(directory/'controller.json',dict(containers=cids,config_sha256=node.sha(node.encoded(config))))
    else:
        saved = node.json_bytes(node.read_small(directory/'controller.json'))
        if saved['config_sha256'] != node.sha(node.encoded(config)):
            raise ValueError('Existing deployment configuration changed')
        cids = saved['containers']
    # Hold one watcher lock for this pair. Acquiring it does not start a model.
    with (directory/'watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (directory/'events.jsonl').open('a',buffering=1) as journal:
            def record(value):
                journal.write(json.dumps(value)+'\n')
                brief = {k:v for k,v in value.items() if k not in ('observations','aot')}
                if value.get('stage') == 'continuous_ram_watch':
                    brief['available'] = [row['memory']['MemAvailable'] for row in value['observations']]
                print(json.dumps(brief),flush=True)
            tunnel = None
            def ready(proof):
                nonlocal tunnel
                if not (directory/'health-ready.json').exists():
                    node.exclusive(directory/'health-ready.json',proof)
            def interrupted(signum,frame):
                raise KeyboardInterrupt(f'signal {signum}')
            signal.signal(signal.SIGTERM,interrupted)
            try:
                if args.execute:
                    node.exclusive(directory/'start-attempt.json',dict(time=now(),containers=cids,automatic_retries=False))
                    started = both(config,'start')
                    record(dict(time=now(),stage='one_shot_start_results',results=started))
                    # An acknowledgement timeout is not proof of exit. Observe
                    # these same IDs; never call start again from the watcher.
                outcome = watch(config,cids,record,on_ready=ready)
                record(dict(time=now(),**outcome))
            except KeyboardInterrupt:
                stops = stop_until_terminal(config,record)
                record(dict(time=now(),status='portable_pair_operator_stop',stops=stops))
            except Exception as error:
                # A broken observer must not silently abandon resident models.
                record(dict(time=now(),stage='controller_exception_stopping_owned_pair',error=repr(error)))
                stop_until_terminal(config,record)
                raise
            finally:
                if tunnel is not None and tunnel.poll() is None:
                    tunnel.terminate()
                    try:
                        tunnel.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        tunnel.kill()
                        tunnel.wait(timeout=5)


if __name__ == '__main__':
    main()
