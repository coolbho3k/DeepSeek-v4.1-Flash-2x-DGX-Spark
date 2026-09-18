"""Private, unarmed CPU call profiler: no tensor copies or CUDA profiling.

Reports Python/C-extension dispatch and blocking wall time, NOT kernel time.
The observed scope is the eager backbone forward, excluding logits/sampling.
At most four operator-armed profiles, each at most eight forwards, per worker.
"""
import cProfile
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

TRACE_SHA = 'c5ad25678aaf3b13067f6cb5f2dd70ac400260c39379cbef3ea9228e04e71cc5'
MIN_AVAILABLE = 2 * 2**30
ROOT = Path('/cache/ds41-cpu-profile')
_installed = None


def available():
    return next(int(line.split()[1])*1024 for line in
                Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))


def exclusive(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()
    if len(raw)>2**20:
        raise ValueError('CPU profile exceeds one MiB')
    with path.open('xb') as stream:
        stream.write(raw)


def control(path, now):
    if not path.exists() and not path.is_symlink():
        return None
    if path.resolve()!=path or not path.is_file() or path.stat().st_size>4096:
        raise ValueError('Bounded regular CPU profile control required')
    def unique(pairs):
        value = {}
        for key,item in pairs:
            if key in value:
                raise ValueError('Duplicate CPU profile control key')
            value[key] = item
        return value
    value = json.loads(path.read_bytes(),object_pairs_hook=unique)
    if (set(value)!={'format','tag','expires_unix','max_forwards','mode'}
            or value['format']!='ds41_cpu_profile_control_v1'
            or not isinstance(value['tag'],str)
            or not re.fullmatch('[a-z0-9][a-z0-9_-]{0,39}',value['tag'])
            or type(value['max_forwards']) is not int
            or not 1<=value['max_forwards']<=8
            or value['mode'] not in ('decode','prefill')
            or type(value['expires_unix']) not in (int,float)
            or not math.isfinite(value['expires_unix'])
            or value['expires_unix']>now+300):
        raise ValueError('Invalid bounded CPU profile control')
    return value if value['expires_unix']>now else None


class Recorder:
    def __init__(self, root):
        self.root = root
        self.active = None
        self.done = set()
        self.next_poll = 0.
        self.disabled = False

    def begin(self, args, kwargs):
        if self.disabled or len(self.done)>=4 or sys.getprofile() is not None:
            return False
        now = time.time()
        if self.active is None:
            if now<self.next_poll:
                return False
            self.next_poll = now+.5
            value = control(self.root/'control.json',now)
            if value is None or value['tag'] in self.done:
                return False
            if (self.root/(value['tag']+'.json')).exists():
                raise ValueError('CPU profile tag already exists')
            self.active = value
            self.profiler = cProfile.Profile()
            self.rows = []
            self.started = now
        if now>=self.active['expires_unix']:
            self.finish('expired')
            return False
        ids = kwargs.get('input_ids',args[0] if args else None)
        shape = getattr(ids,'shape',())  # Tensor metadata only; never read values.
        if len(shape)!=1 or not 1<=shape[0]<=1056:
            return False
        rows = int(shape[0])
        if (rows==1)!=(self.active['mode']=='decode'):
            return False
        if available()<MIN_AVAILABLE:
            self.finish('skipped_low_host_memory')
            return False
        self.current_rows = rows
        self.current_start = time.perf_counter()
        self.profiler.enable()
        return True

    def end(self, successful):
        self.profiler.disable()
        self.rows.append(dict(tokens=self.current_rows,
            wall_seconds=time.perf_counter()-self.current_start,
            successful=successful))
        if not successful or len(self.rows)>=self.active['max_forwards']:
            self.finish('complete' if successful else 'model_forward_failed')

    def finish(self, status):
        value = self.active
        self.active = None
        self.done.add(value['tag'])
        self.profiler.create_stats()
        def render(item):
            (filename,line,name),(primitive,total,self_time,cumulative,_) = item
            return dict(file=filename,line=line,function=name,primitive_calls=primitive,
                calls=total,self_seconds=self_time,cumulative_seconds=cumulative)
        entries = list(self.profiler.stats.items())
        exclusive(self.root/(value['tag']+'.json'),dict(
            format='ds41_cpu_profile_v1',status=status,control=value,
            rank=0,pid=os.getpid(),started_unix=self.started,finished_unix=time.time(),
            forwards=self.rows,profiled_functions=len(entries),
            top_self=[render(x) for x in sorted(entries,key=lambda x:x[1][2],reverse=True)[:200]],
            top_cumulative=[render(x) for x in sorted(entries,key=lambda x:x[1][3],reverse=True)[:200]],
            scope='eager_backbone_forward_excludes_logits_and_sampler',
            timer='CPU_call_wall_time_including_blocking_and_profiler_overhead',
            cuda_kernel_times=False,tensor_copies=False,unprofiled_speed_measurement=False,
            min_host_available_bytes=MIN_AVAILABLE))
        del self.profiler


def instrument(core, root):
    """Wrap one eager backbone without altering arguments, results or errors."""
    original = core.forward
    recorder = Recorder(root)
    @functools.wraps(original)
    def forward(*args, **kwargs):
        armed = False
        try:
            armed = recorder.begin(args,kwargs)
        except Exception as error:
            recorder.disabled = True
            print('DS41 CPU profile disabled: '+repr(error),flush=True)
        successful = False
        try:
            result = original(*args,**kwargs)
            successful = True
            return result
        finally:
            if armed:
                try:
                    recorder.end(successful)
                except Exception as error:
                    recorder.disabled = True
                    print('DS41 CPU profile report failed: '+repr(error),flush=True)
    core.forward = forward
    return recorder


def register():
    global _installed
    import spark_attention_trace as trace
    if hashlib.sha256(Path(trace.__file__).read_bytes()).hexdigest()!=TRACE_SHA:
        raise ValueError('Unreviewed original observer source')
    if _installed is not None:
        if trace.attach is not _installed:
            raise ValueError('CPU profiling attachment changed')
        return
    original = trace.attach
    @functools.wraps(original)
    def attach(model,rank):
        result = original(model,rank)
        if rank==0:
            if type(model).__name__!='DeepseekV41ForCausalLM':
                raise ValueError('Actual native model required')
            if ROOT.resolve()!=ROOT or ROOT.exists():
                raise ValueError('Fresh private CPU profile directory required')
            ROOT.mkdir(mode=0o700)
            model._ds41_cpu_profile = instrument(model.language_model.model,ROOT)
            exclusive(ROOT/'ready.json',dict(status='cpu_profile_attached_unarmed',
                rank=rank,pid=os.getpid(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                max_profiles=4,max_forwards=8,max_tokens=1056,min_host_available_bytes=MIN_AVAILABLE,
                gpu_profiling_allocations=False,tensor_copies=False,outputs_replaced=False))
        return result
    trace.attach = _installed = attach
