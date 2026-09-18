# SPDX-License-Identifier: AGPL-3.0-only
# Adapted from MiaAI Lab / Wesley Young cooperative_moe/runtime.py @
# b9c49e90bdcc6f1e0192feb57214df11b67d36aa; native kernels derive from
# Turboderp's ExLlamaV3. Original MIT notices and AGPL license are retained in
# vendor/miaai-cooperative-moe-agpl. Local changes: shared-workspace aliases,
# native sparse Bank tables, fused input preparation and graph-owner fences.
"""Opt-in cooperative MoE, integrated INSIDE our existing serialized dispatcher.

No weights, vision, prefill, draft or collective changes. Native failures
propagate through the dispatcher's poison guard, never retry another kernel.
Registration reads files only; CUDA prewarm happens on first eligible eager
forward and is forbidden during capture. Kernel is a separate quality candidate.
"""
import ctypes as C
import hashlib
import json
from pathlib import Path

from .cooperative_contract import CAPACITIES, LAYOUT, UPSTREAM_COMMIT, eligible_shape, intervals


def alias_scratch(work):
    import torch
    intervals()
    if len(work.temps) != 4:
        raise ValueError('Expected the original four shared MoE temps')
    for temp, capacity in zip(work.temps, CAPACITIES):
        if (temp.dtype != torch.float16 or temp.device != work.device or not temp.is_contiguous()
                or temp.numel()*temp.element_size() != capacity):
            raise ValueError('Unexpected shared workspace layout')
    views = []
    for row, (_, parent, start, end) in zip(LAYOUT, intervals()):
        _, _, _, shape, _, dtype = row
        view = work.temps[parent].view(torch.uint8).flatten().narrow(0, start, end-start)
        views.append(view.view(getattr(torch, dtype)).reshape(shape))
    return views


class Native:
    def __init__(self, work, binary, digest):
        import torch
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Cooperative MoE requires eager prewarm before capture')
        if hashlib.sha256(binary.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Cooperative binary changed after registration')
        self.device = work.device
        self.library = C.CDLL(str(binary))
        self.library.goal50_coop_abi.argtypes = []
        self.library.goal50_coop_abi.restype = C.c_int
        if self.library.goal50_coop_abi() != 2:
            raise RuntimeError('Unknown cooperative ABI')
        self.library.goal50_coop_info.argtypes = [C.c_int, C.c_int, C.POINTER(C.c_int)]
        self.library.goal50_coop_info.restype = C.c_int
        self.launch = self.library.goal50_coop_launch
        self.launch.argtypes = [C.POINTER(C.c_void_p), C.c_int, C.c_int, C.c_int,
            C.c_float, C.c_int, C.c_int, C.c_void_p]
        self.launch.restype = C.c_int
        info = (C.c_int*18)()
        status = self.library.goal50_coop_info(3, 1, info)
        if (status or tuple(info)[15:] != (48, 2547, 344)
                or any(info[i] != 512 or info[i+4] < 1 or info[i+1] > 101376 for i in (0, 5, 10))):
            raise RuntimeError(f'Unexpected cooperative resources: {status}, {list(info)}')
        self.resources = tuple(info)
        self.scratch = alias_scratch(work)

    def __call__(self, bank, x, ids, weights):
        import torch
        from .cooperative_routes import prepare
        if x.device != self.device or not eligible_shape(x.shape, ids.shape):
            raise ValueError('Unsupported cooperative call')
        # Bank.__init__ has already checked EVERY expert/projection is K3/MUL1
        # and retained its tensor owners. Keep its sparse sentinel mapping.
        if len(bank.ptrs) != 9 or not 1 <= len(bank.keys) <= 384:
            raise ValueError('Unexpected cooperative pointer bank')
        half_x, local, rw = prepare(x, ids, weights, bank.mapping, self.scratch[-1])
        out = torch.empty(x.shape, device=x.device, dtype=torch.float32)
        tensors = [half_x, local, rw, *bank.ptrs, *self.scratch, out]
        pointers = (C.c_void_p*20)(*[tensor.data_ptr() for tensor in tensors])
        status = self.launch(pointers, 3, len(x), len(bank.keys), 10., 1, 0,
            C.c_void_p(torch.cuda.current_stream(x.device).cuda_stream))
        if status:
            raise RuntimeError(f'Cooperative MoE CUDA launch failed: {status}; no retry')
        return out.to(x.dtype)


def configure(serving_directory):
    """Return a verified callable without creating a CUDA context/buffer."""
    root = Path(serving_directory).resolve()
    receipt = json.loads((root/'cooperative-native.json').read_bytes())
    binary = root/'cooperative_moe.so'
    digest = receipt.get('binary_sha256')
    if (receipt.get('status') != 'cooperative_moe_built_cpu_only' or receipt.get('abi') != 2
            or receipt.get('upstream_commit') != UPSTREAM_COMMIT
            or hashlib.sha256(binary.read_bytes()).hexdigest() != digest):
        raise RuntimeError('Unreviewed cooperative native source/binary contract')

    def call(work, bank, x, ids, weights):
        # Caller holds the real dispatcher lock and has waited on its ready
        # event. GraphOwner already retains this same dispatcher/work/bank.
        native = getattr(work, '_ds41_cooperative', None)
        if native is None:
            native = Native(work, binary, digest)
            work._ds41_cooperative = native
        return native(bank, x, ids, weights)

    return call
