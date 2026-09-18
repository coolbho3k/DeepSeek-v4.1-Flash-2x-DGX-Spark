# SPDX-License-Identifier: AGPL-3.0-only
# Integrates with the existing vLLM communicator (Apache-2.0); no NCCL code is
# vendored. MiaAI's two-Spark foundation and notices remain in the parent kit.
"""One existing DCP communicator, one side stream, explicit CUDA fork/join.

No new process group, networking configuration, persistent tensor workspace,
stream-global monkeypatch, or CUDA work on import. Prepare before capture.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import threading

from .policy import CHANNELS, HEADS, MAX_ROWS, MAX_TRANSFERS

# Pinned serving virtualenv, not the unrelated system-Python installation.
PYNCCL_SHA256 = '579f58bfe934bda4414d7098bce95cc721ddeeff004c64300a01d34d3193e25c'
_transports = {}
_registry_lock = threading.RLock()
_failed_resources = []


class Transfer:
    def __init__(self, transport, source, destination, ready, done):
        self.transport = transport
        self.source = source
        self.destination = destination
        self.ready, self.done = ready, done
        self.joined = False
        self.query_dependency = None
        self.resources = None

    def join(self):
        """Enqueue a GPU wait, never a host/device synchronization."""
        import torch
        transport = self.transport
        transport.check_session()
        if self.joined:
            raise RuntimeError('DCP transfer joined twice')
        current = torch.cuda.current_stream(transport.device)
        if current.cuda_stream != transport.caller_stream:
            raise RuntimeError('DCP transfer must join its allocating stream')
        current.wait_event(self.done)
        self.joined = True
        transport.pending.remove(self)
        if self.query_dependency is not None:
            dependency = self.query_dependency
            if dependency.joined or dependency not in transport.pending:
                raise RuntimeError('Changed side-stream query ownership')
            dependency.joined = True
            transport.pending.remove(dependency)
        # source/destination remain referenced by this ticket through the
        # consumer enqueue. Both were allocated/produced on the joined stream.
        return self.destination


class Transport:
    def __init__(self, group):
        import torch
        from vllm.distributed.device_communicators import pynccl
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare the DCP overlap stream before capture')
        source = Path(pynccl.__file__)
        if hashlib.sha256(source.read_bytes()).hexdigest() != PYNCCL_SHA256:
            raise RuntimeError('Unreviewed vLLM PyNCCL stream interface')
        comm = getattr(getattr(group, 'device_communicator', None), 'pynccl_comm', None)
        if (type(comm) is not pynccl.PyNcclCommunicator or group.world_size != 2
                or group.rank_in_group not in (0, 1) or comm.world_size != 2
                or comm.rank != group.rank_in_group or not comm.available or comm.disabled
                or comm.device != torch.device('cuda', torch.cuda.current_device())):
            raise RuntimeError('DCP overlap requires the existing enabled two-rank PyNCCL communicator')
        self.group, self.comm, self.device = group, comm, comm.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.lock = threading.RLock()
        self.pending = []
        self.failed = False
        self.thread = self.caller_stream = None

    def check_session(self):
        if (self.failed or self.thread != threading.get_ident()
                or self.caller_stream is None or self.comm.disabled
                or not self.comm.available):
            raise RuntimeError('Inactive or poisoned DCP overlap session')

    @contextmanager
    def session(self):
        import torch
        from ds41.graph_validation import _execution_lock, require_capture_owner
        # Existing owned model replay uses this same lock. No eager overlap
        # forward can interleave communicator submissions with another graph.
        with _execution_lock, self.lock:
            if self.failed or self.thread is not None or self.pending:
                raise RuntimeError('Nested, unfinished, or poisoned DCP overlap')
            require_capture_owner()
            self.thread = threading.get_ident()
            self.caller_stream = torch.cuda.current_stream(self.device).cuda_stream
            try:
                yield self
                if self.pending:
                    raise RuntimeError('DCP side stream was not joined before returning')
            except BaseException:
                self.failed = True
                # Never free/reuse raw NCCL buffers after a partial submission
                # failure, and never issue CUDA cleanup work on that path.
                _failed_resources.append((self, tuple(self.pending)))
                raise
            finally:
                self.thread = self.caller_stream = None

    def gather(self, value, *, kind):
        import torch
        from ds41.graph_validation import current_owner, require_capture_owner
        self.check_session()
        expected = CHANNELS if kind == 'query' else CHANNELS + 1 if kind == 'result' else None
        dtype = torch.bfloat16 if kind == 'query' else torch.float32
        if (expected is None or value.ndim != 3 or not 1 <= len(value) <= MAX_ROWS
                or tuple(value.shape[1:]) != (HEADS, expected) or value.dtype != dtype
                or value.device != self.device or not value.is_contiguous()
                or value.requires_grad or len(self.pending) >= MAX_TRANSFERS):
            raise ValueError('Invalid or unbounded DCP overlap payload')
        caller = torch.cuda.current_stream(self.device)
        if caller.cuda_stream != self.caller_stream:
            raise RuntimeError('Allocate transport payloads on the caller stream')
        destination = torch.empty((2, *value.shape), device=value.device, dtype=value.dtype)
        ready, done = torch.cuda.Event(), torch.cuda.Event()
        ticket = Transfer(self, value, destination, ready, done)
        self.pending.append(ticket)  # Retain BEFORE the first CUDA submission.
        if torch.cuda.is_current_stream_capturing():
            require_capture_owner()
            owner = current_owner()
            events = getattr(owner, '_ds41_dcp_overlap_events', None)
            if events is None:
                events = owner._ds41_dcp_overlap_events = []
            if len(events) >= 4096:
                raise RuntimeError('Bounded DCP graph event inventory exceeded')
            # Retain event/stream handles, not captured temporary tensors. The
            # native graph pool retains addresses under the existing owner.
            events.append((self.stream, ready, done))
        ready.record(caller)
        self.stream.wait_event(ready)
        # Pass an explicit stream: vLLM's cached current_stream helper is NOT
        # assumed to observe a torch.cuda.stream context-manager switch.
        self.comm.all_gather(destination, value, stream=self.stream)
        done.record(self.stream)
        return ticket

    def remote_result(self, query, payload, produce):
        """Compute peer heads on the NCCL stream while own heads run on caller.

        Output/send/receive allocations all belong to caller. Retain the
        closure (all metadata/input/output views) until the transitive join.
        No extra stream, persistent workspace or new communicator is needed.
        """
        import torch
        from ds41.graph_validation import current_owner, require_capture_owner
        self.check_session()
        caller = torch.cuda.current_stream(self.device)
        if (caller.cuda_stream != self.caller_stream or query.transport is not self
                or query.joined or self.pending != [query] or not callable(produce)
                or payload.shape != (*query.source.shape[:2], CHANNELS + 1)
                or payload.dtype != torch.float32 or payload.device != self.device
                or not payload.is_contiguous() or payload.requires_grad):
            raise ValueError('Invalid concurrent DCP fork')
        destination = torch.empty((2, *payload.shape), device=self.device, dtype=payload.dtype)
        ready, done = torch.cuda.Event(), torch.cuda.Event()
        ticket = Transfer(self, payload, destination, ready, done)
        ticket.query_dependency, ticket.resources = query, produce
        self.pending.append(ticket)
        if torch.cuda.is_current_stream_capturing():
            require_capture_owner()
            events = current_owner()._ds41_dcp_overlap_events
            if len(events) >= 4096:
                raise RuntimeError('Bounded DCP graph event inventory exceeded')
            events.append((self.stream, ready, done))
        ready.record(caller)  # Metadata preparation follows the query fork.
        self.stream.wait_event(ready)
        with torch.cuda.stream(self.stream):
            produce()  # Same stream as the preceding query all-gather.
            self.comm.all_gather(destination, payload, stream=self.stream)
            done.record(self.stream)
        return ticket


def prepare(group):
    """Called once by the candidate worker after native distributed init."""
    with _registry_lock:
        key = id(group)
        if key not in _transports:
            if _transports:
                raise RuntimeError('Only one DCP overlap communicator is admitted')
            _transports[key] = Transport(group)
        result = _transports[key]
        if result.group is not group or result.failed:
            raise RuntimeError('DCP communicator identity changed or was poisoned')
        return result


def prepared(group):
    result = _transports.get(id(group))
    if result is None or result.group is not group or result.failed:
        raise RuntimeError('DCP overlap transport was not prepared before execution')
    return result
