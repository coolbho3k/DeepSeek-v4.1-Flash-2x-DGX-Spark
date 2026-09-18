# SPDX-License-Identifier: AGPL-3.0-only
"""Reclaim the CPU scheduler parent's unused heap after worker spawn.

No CUDA initialization, model access, global cache advice, memory-budget
change or worker mutation. Native readiness results and exceptions propagate.
"""
import ctypes
import functools
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

_installed = None


def trim_unused():
    torch = sys.modules.get('torch')
    initialized=torch is not None and torch.cuda.is_initialized()
    # Native backend registration may create a context in the CPU scheduler.
    # It must not own tensor allocations; never initialize CUDA here ourselves.
    if initialized and (torch.cuda.memory_allocated()!=0 or torch.cuda.memory_reserved()!=0):
        raise RuntimeError('Parent heap cleanup requires no GPU tensor allocations')
    def sample():
        status=Path('/proc/self/status').read_text().splitlines()
        rss=next(int(x.split()[1])*1024 for x in status if x.startswith('VmRSS:'))
        memory={x.split(':')[0]:int(x.split()[1])*1024 for x in
                Path('/proc/meminfo').read_text().splitlines()
                if x.startswith(('MemAvailable:','MemFree:'))}
        return dict(rss_bytes=rss,**memory)
    before=sample()
    collected=gc.collect()
    trim=ctypes.CDLL('libc.so.6').malloc_trim
    trim.argtypes,trim.restype=[ctypes.c_size_t],ctypes.c_int
    result=int(trim(0))
    proof=dict(stage='ds41_parent_post_spawn_heap_trim',pid=os.getpid(),
        before=before,after=sample(),gc_collected=collected,malloc_trim_result=result,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cuda_context_already_initialized=initialized,cuda_allocations=False,
        limits_changed=False,only_current_process_unused_heap=True)
    print(json.dumps(proof),flush=True)
    return proof


def wrap(original):
    @functools.wraps(original)
    def wait(unready_proc_handles):
        trim_unused()
        return original(unready_proc_handles)
    return wait


def register():
    global _installed
    from vllm.v1.executor.multiproc_executor import WorkerProc
    if _installed is not None:
        if WorkerProc.wait_for_ready is not _installed:
            raise RuntimeError('Parent readiness binding changed')
        return
    descriptor=vars(WorkerProc)['wait_for_ready']
    if not isinstance(descriptor,staticmethod):
        raise RuntimeError('Native static readiness method required')
    import inspect
    if tuple(inspect.signature(descriptor.__func__).parameters)!=('unready_proc_handles',):
        raise RuntimeError('Native readiness signature changed')
    _installed=wrap(descriptor.__func__)
    WorkerProc.wait_for_ready=staticmethod(_installed)
