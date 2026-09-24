# SPDX-License-Identifier: AGPL-3.0-only
"""Two-rank low-latency collectives for small decode messages (GB10 pair).

libfastcomm RDMA-writes into the peer's pinned host buffers, which the
integrated GPU reads coherently (GPUDirect RDMA is unavailable on GB10).
Only BF16 all-reduces and byte all-gathers of at most LIMIT bytes (16-byte
aligned, contiguous CUDA tensors) are routed here; everything else,
including every prefill-sized message, keeps the original NCCL path. Two-rank
BF16 sums are one commutative add, so results equal NCCL's bit for bit.

Channel 0 serves the main stream, channel 1 the DCP overlap side stream; each
channel is strictly ordered, and both ranks issue identical sequences.
"""
import ctypes as C
import hashlib
import os
from pathlib import Path
import threading

LIB_SHA256 = '23585e6fc35f88c6d43432ad0bc6b7f6e98f3a68c61f604f52c54fc1e171113a'
MAX_BYTES = 1 << 20
LIMIT = 256 * 1024
_state = None
_lock = threading.Lock()


class _FastComm:
    def __init__(self, lib, handle, rank):
        self.lib, self.handle, self.rank = lib, handle, rank
        self.registered = set()
        self.side_stream = None

    def check(self):
        error = self.lib.fc_error(self.handle)
        if error:
            raise RuntimeError(f'fastcomm proxy failed with code {error}')


def _eligible(t):
    return (t.is_cuda and t.is_contiguous() and 0 < t.numel() * t.element_size() <= LIMIT
            and (t.numel() * t.element_size()) % 16 == 0 and t.data_ptr() % 16 == 0)


def _channel(stream):
    state = _state
    return 1 if state.side_stream is not None and stream == state.side_stream else 0


def prepare():
    """Create and connect the queue pair once, after native distributed init."""
    global _state
    import torch
    import torch.distributed as dist
    from vllm.distributed import get_dcp_group, get_tp_group
    with _lock:
        if _state is not None:
            return _state
        tp, dcp = get_tp_group(), get_dcp_group()
        if tp.world_size != 2 or dcp.world_size != 2 or tp.rank_in_group != dcp.rank_in_group:
            raise RuntimeError('fastcomm requires the same two ranks for TP and DCP')
        path = Path(__file__).resolve().parents[1] / 'libfastcomm.so'
        if hashlib.sha256(path.read_bytes()).hexdigest() != LIB_SHA256:
            raise RuntimeError('Unreviewed fastcomm library')
        lib = C.CDLL(str(path))
        lib.fc_create.restype = C.c_void_p
        lib.fc_create.argtypes = [C.c_char_p, C.c_int, C.c_uint64, C.c_void_p, C.POINTER(C.c_int)]
        lib.fc_connect.argtypes = [C.c_void_p, C.c_void_p]
        lib.fc_error.argtypes = [C.c_void_p]
        lib.fc_allreduce_bf16.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_uint64, C.c_int, C.c_void_p]
        lib.fc_allgather.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_uint64, C.c_int, C.c_int, C.c_void_p]
        device = os.environ['NCCL_IB_HCA'].lstrip('=').split(',')[0].split(':')[0]
        gid = int(os.environ['NCCL_IB_GID_INDEX'])
        info, size = C.create_string_buffer(256), C.c_int(0)
        handle = lib.fc_create(device.encode(), gid, MAX_BYTES, info, C.byref(size))
        if not handle:
            raise RuntimeError('fastcomm queue pair creation failed')
        infos = [None, None]
        dist.all_gather_object(infos, info.raw[:size.value], group=tp.cpu_group)
        if lib.fc_connect(handle, infos[1 - tp.rank_in_group]) != 0:
            raise RuntimeError('fastcomm connection failed')
        dist.barrier(group=tp.cpu_group)
        state = _FastComm(lib, handle, tp.rank_in_group)
        state.registered = {id(tp.device_communicator), id(dcp.device_communicator)}
        _install_all_reduce()
        _state = state
        return state


def set_side_stream(stream):
    """The DCP overlap transport's dedicated stream uses channel 1."""
    if _state is not None:
        _state.side_stream = stream.cuda_stream


def _install_all_reduce():
    from vllm.distributed.device_communicators import cuda_communicator as module
    cls = module.CudaCommunicator
    if getattr(cls.all_reduce, '_ds41_fastcomm', False):
        return
    original = cls.all_reduce

    def all_reduce(self, input_):
        import torch
        state = _state
        if state is not None and id(self) in state.registered and input_.dtype == torch.bfloat16 and _eligible(input_):
            stream = torch.cuda.current_stream().cuda_stream
            out = torch.empty_like(input_)
            if state.lib.fc_allreduce_bf16(state.handle, input_.data_ptr(), out.data_ptr(),
                                           input_.numel(), _channel(stream), stream):
                raise RuntimeError('fastcomm all-reduce launch failed')
            return out
        return original(self, input_)

    all_reduce._ds41_fastcomm = True
    cls.all_reduce = all_reduce


def all_gather(comm, destination, value, stream):
    """Drop-in for PyNccl all_gather(destination, value, stream=...) in the DCP transport."""
    state = _state
    if (state is not None and _eligible(value) and destination.is_contiguous()
            and destination.numel() == 2 * value.numel() and destination.dtype == value.dtype):
        handle = stream.cuda_stream
        if state.lib.fc_allgather(state.handle, value.data_ptr(), destination.data_ptr(),
                                  value.numel() * value.element_size(), _channel(handle), state.rank, handle):
            raise RuntimeError('fastcomm all-gather launch failed')
        return
    comm.all_gather(destination, value, stream=stream)
