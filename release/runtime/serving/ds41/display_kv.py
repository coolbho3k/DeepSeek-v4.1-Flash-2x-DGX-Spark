# SPDX-License-Identifier: AGPL-3.0-only
"""Additive display-backed KV with unchanged native views and block indexing.

Only the KV backing allocation is replaced. A verified contiguous UVA span
contains only1.75GiB display reserve; final KV uses zero ordinary RAM.
This external allocation is explicitly budgeted, not charged to PyTorch's
allocator. Owners are retained until process exit, including graph lifetimes.
"""
import ctypes
import hashlib
import importlib
import json
import os
from pathlib import Path

DISPLAY_BYTES = 1792 * 2**20
ORDINARY_LIMIT = 1024 * 2**20
QUANTUM = 65536
_owners = []
_installed = None
NATIVE_PINS = {
    'vllm.v1.worker.utils': '5ed6caaf5797c214398be2a28fe7b6b24a4e9eb7b9b72fa6f17afec4a495875c',
    'vllm.v1.worker.gpu.attn_utils': 'f97c054920a382cbbf844b6029aa6561e7a158fce0aaf9915dcd00a6ec817aa9',
}

def memory():
    return {line.split(':')[0]:int(line.split()[1])*1024
            for line in Path('/proc/meminfo').read_text().splitlines()
            if line.split(':')[0] in ('MemFree','MemAvailable')}

class Owner:
    def __init__(self, ordinary_bytes, library):
        import torch
        if not torch.cuda.is_initialized() or torch.cuda.current_device()!=0:
            raise RuntimeError('Initialize the existing CUDA0 context first')
        if type(ordinary_bytes) is not int or not 0<=ordinary_bytes<=ORDINARY_LIMIT or ordinary_bytes%QUANTUM:
            raise ValueError('Invalid explicitly budgeted ordinary-memory size')
        self.lib=ctypes.CDLL(str(library))
        self.lib.ds41_display_create.argtypes=[ctypes.c_size_t,ctypes.c_size_t]
        self.lib.ds41_display_create.restype=ctypes.c_void_p
        self.lib.ds41_display_pointer.argtypes=[ctypes.c_void_p]
        self.lib.ds41_display_pointer.restype=ctypes.c_uint64
        self.lib.ds41_display_destroy.argtypes=[ctypes.c_void_p]
        self.lib.ds41_display_destroy.restype=None
        self.lib.ds41_display_error.restype=ctypes.c_char_p
        self.handle=self.lib.ds41_display_create(ordinary_bytes,DISPLAY_BYTES)
        if not self.handle:raise RuntimeError(self.lib.ds41_display_error().decode())
        self.pointer=self.lib.ds41_display_pointer(self.handle)
        self.ordinary_bytes=ordinary_bytes
        self.size=ordinary_bytes+DISPLAY_BYTES
        self.__cuda_array_interface__={'shape':(self.size,), 'strides':None,
            'typestr':'|i1','data':(self.pointer,False),'version':3}

    def tensor(self):
        import torch
        tensor=torch.as_tensor(self,device='cuda:0')
        if tensor.data_ptr()!=self.pointer or tensor.dtype!=torch.int8 or tensor.numel()!=self.size:
            raise RuntimeError('CUDA array interface copied or misinterpreted external storage')
        return tensor

    def close_for_probe(self):
        """Probe only: synchronize and discard all tensor views/graphs first."""
        if self.handle:
            self.lib.ds41_display_destroy(self.handle)
            self.handle=None

def credited_budgets(available, cap):
    if type(cap) is not int or cap != 0 or len(available)!=2 or any(type(v) is not int or v<=0 for v in available):
        raise ValueError('Additive display KV requires positive native budgets and zero ordinary KV')
    # Round DOWN ordinary credit so backing-page rounding never exceeds it.
    ordinary=[min(v,cap)//QUANTUM*QUANTUM for v in available]
    if any(ordinary):raise ValueError('Display-only KV cannot consume ordinary RAM')
    budgets=[v+DISPLAY_BYTES for v in ordinary]
    print(json.dumps(dict(stage='ds41_additive_display_kv_budget',
        native_profile_bytes=available,ordinary_budget_bytes=ordinary,
        external_display_bytes_per_rank=DISPLAY_BYTES,total_budget_bytes=budgets,
        ordinary_gpu_utilization_unchanged=True)),flush=True)
    return budgets

def backing(size, dtype, device):
    import torch
    # Native graph-estimation/profiling uses small, temporary ordinary caches.
    if _real_allocation_size is None:
        if _owners or size > DISPLAY_BYTES:
            raise RuntimeError('Unexpected profiling allocation outside initialization')
        return torch.zeros(size,dtype=dtype,device=device)
    if size != _real_allocation_size:
        raise ValueError('Final KV allocation differs from admitted descriptors')
    if dtype!=torch.int8 or torch.device(device)!=torch.device('cuda:0') or _owners:
        raise RuntimeError('Only one real, bounded additive KV pool per worker is allowed')
    ordinary=0
    if ordinary>ORDINARY_LIMIT:raise ValueError('Native descriptors exceed additive KV budget')
    before=memory()
    if before['MemAvailable']<ordinary+512*2**20:
        raise RuntimeError('Insufficient ordinary host RAM for the additive KV prefix')
    owner=Owner(ordinary,Path(__file__).resolve().parent.parent/'libds41_display_kv.so')
    _owners.append(owner)  # Process-lifetime ownership, even if a later check fails.
    full=owner.tensor()
    buf=full[:size]
    buf.zero_()
    torch.cuda.synchronize()
    after=memory()
    if before['MemAvailable']-after['MemAvailable']>ordinary+128*2**20:
        raise RuntimeError('External display allocation consumed unexpected ordinary RAM')
    if after['MemAvailable']<512*2**20:
        raise RuntimeError('Additive KV allocation crossed the existing host headroom floor')
    receipt=dict(stage='ds41_additive_display_kv_allocated',pid=os.getpid(),
        logical_bytes=size,ordinary_bytes=ordinary,display_bytes=DISPLAY_BYTES,
        reserved_virtual_bytes=owner.size,storage_pointer=owner.pointer,
        before=before,after=after,torch_allocator_tracks_external_pool=False,
        native_layout_unchanged=True,owner_lifetime='worker_process')
    print(json.dumps(receipt),flush=True)
    path=Path('/cache/ds41-display-kv');path.mkdir(exist_ok=True)
    (path/f'{os.getpid()}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return buf

def register():
    global _installed
    from .vllm_dcp import _compile
    modules={name:importlib.import_module(name) for name in NATIVE_PINS}
    for name,module in modules.items():
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=NATIVE_PINS[name]:
            raise RuntimeError('Unreviewed native display-KV allocation target: '+name)
    utils,attn=modules.values()
    if _installed is not None:
        if utils.allocate_kv_cache is not _installed or attn.allocate_kv_cache is not _installed:
            raise RuntimeError('Display-KV hook changed after registration')
        return
    original=utils.allocate_kv_cache
    if attn.allocate_kv_cache is not original:raise RuntimeError('Native allocator binding already changed')
    patched=_compile(original,[
        ('buf = torch.zeros(buf_size, dtype=torch.int8, device=device)',
         'buf = _ds41_display_backing(buf_size, dtype=torch.int8, device=device)'),
    ],{'_ds41_display_backing':backing})
    utils.allocate_kv_cache=attn.allocate_kv_cache=patched
    _installed=patched

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
