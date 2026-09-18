# SPDX-License-Identifier: AGPL-3.0-only
# DS41 graph integration for the MiaAI-derived serving stack. See
# ../vendor/miaai-serving-stack-agpl/LICENSE and LICENSE.MIT.
"""Owned graph error reporting and native SSD callback lifetimes.

No hooks or CUDA objects are created on import. Kernels must mask invalid
addresses before reporting an error here. Graph replay is checked before its
outputs leave the execution boundary; errors poison the owner, not the GPU.
Eager calls retain their synchronous exception boundary.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import threading

MAX_ERROR_VALUES = 8192
# Whole target/draft graphs plus bounded per-layer piecewise entries. This
# upper bound costs16MiB, not a model/context-sized persistent workspace.
MAX_OWNERS = 512
_current = ContextVar('ds41_owned_model_graph', default=None)
_execution_lock = threading.RLock()
_live_owners = set()


def current_owner():
    return _current.get()


def require_capture_owner():
    import torch
    if torch.cuda.is_current_stream_capturing() and current_owner() is None:
        raise RuntimeError('Graph capture requires an owned validation/callback boundary')


def _raise_flags(flags, messages):
    known = 0
    for bit, message in messages:
        if type(bit) is not int or bit <= 0 or bit & (bit - 1):
            raise ValueError('Error messages require distinct positive bit masks')
        if known & bit:
            raise ValueError('Duplicate graph error mask')
        known |= bit
        if any(value & bit for value in flags):
            raise ValueError(message)
    if any(value & ~known for value in flags):
        raise ValueError('Unknown graph validation error')


def check_flags(errors, messages):
    """Keep the original error codes; coalesce only the graph's host transfer."""
    import torch
    messages = tuple(messages)
    _raise_flags((), messages)  # Validate even when all device flags are zero.
    if (not isinstance(errors, torch.Tensor) or errors.device.type != 'cuda'
            or errors.dtype not in (torch.int32, torch.int64)
            or not 0 < errors.numel() <= MAX_ERROR_VALUES):
        raise ValueError('Expected a bounded CUDA integer error tensor')
    if torch.cuda.is_current_stream_capturing():
        require_capture_owner()
        current_owner().record(errors, messages)
    else:
        _raise_flags(errors.reshape(-1).cpu().tolist(), messages)


class GraphOwner:
    """One native graph entry, not an independent graph or allocator policy."""
    def __init__(self, device):
        import torch
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Create graph owners before capture and KV admission')
        device = torch.device(device)
        if device.type != 'cuda' or device.index != torch.cuda.current_device():
            raise ValueError('Graph owner must use the current visible CUDA device')
        if len(_live_owners) >= MAX_OWNERS:
            raise ValueError('Combined graph owner bound exceeded')
        self.device = device
        # Outside the shared graph pool; no captured temporary tensor is retained.
        self.flags = torch.zeros(MAX_ERROR_VALUES, device=device, dtype=torch.int32)
        self.event = torch.cuda.Event()
        self.checks = []
        self.used = 0
        self.stages = []
        # Separate ownership bound: the two Engram tables remain unchanged;
        # an optional, explicitly typed input-vocabulary stage has its own slot.
        self.vocab_stages = []
        # One separately bounded raw draft expert bank; never consumes or
        # expands the original two-Engram or one-vocabulary ownership slots.
        self.draft_stages = []
        self.locked_stages = []
        self.dispatchers = []
        self.locked_dispatchers = []
        self.failed = self.closed = self.pending = self.captured = False
        self.capture_only = False
        _live_owners.add(self)

    def record(self, errors, messages):
        if (self.failed or self.closed or self.captured or not self.capture_only
                or current_owner() is not self or errors.device != self.device):
            raise RuntimeError('Graph validation owner is not capturing this operation')
        count = errors.numel()
        if self.used + count > MAX_ERROR_VALUES:
            raise ValueError('Captured graph exceeds bounded validation storage')
        self.flags[self.used:self.used + count].copy_(errors.reshape(-1))
        self.checks.append((self.used, count, messages))
        self.used += count

    def enqueue_stage(self, stage, indices, out):
        """Retain raw callback descriptors and fence shared pinned buffers."""
        if (current_owner() is not self or not self.capture_only or self.captured
                or self.failed or self.closed or stage.device != self.device):
            raise RuntimeError('SSD callback is outside its whole-model capture owner')
        if stage not in self.stages:
            if len(self.stages) >= 2:
                raise ValueError('Only the two original Engram stages may be captured')
            stage.lock.acquire()
            self.locked_stages.append(stage)
            if stage.failed or stage.closed:
                raise RuntimeError('Cannot capture a failed or closed SSD stage')
            stage._wait()
            stage.graphs += 1
            self.stages.append(stage)
        stage._validate(indices, out)
        stage._enqueue(indices, out)

    def enqueue_vocab_stage(self, stage, indices, out):
        """Retain one separately qualified, shared input-embedding callback."""
        from .native_vocab_stage import NativeVocabStage
        if (type(stage) is not NativeVocabStage or current_owner() is not self
                or not self.capture_only or self.captured or self.failed or self.closed
                or stage.device != self.device):
            raise RuntimeError('Vocabulary callback is outside its owned capture boundary')
        if stage not in self.vocab_stages:
            if self.vocab_stages:
                raise ValueError('Only one shared input vocabulary stage may be captured')
            stage.lock.acquire()
            self.locked_stages.append(stage)
            if stage.failed or stage.closed:
                raise RuntimeError('Cannot capture a failed or closed vocabulary stage')
            stage._wait()
            stage.graphs += 1
            self.vocab_stages.append(stage)
        stage._validate(indices, out)
        stage._enqueue(indices, out)

    def retain_moe(self, dispatcher, experts, *, needs_fat):
        """Only prewarmed immutable target banks may enter a model graph.

        The eager dispatcher takes its own non-reentrant lock during capture.
        Replay bypasses that Python call, so execution() takes the same lock
        and updates its event outside the graph for later eager/stream users.
        """
        if (current_owner() is not self or not self.capture_only or self.captured
                or self.failed or self.closed or dispatcher.failed):
            raise RuntimeError('EXL3 workspace is outside its capture owner')
        work = dispatcher.workspace
        bank = dispatcher.banks.get(id(experts))
        if (work is None or work.device != self.device or bank is None or bank.owner is not experts
                or (needs_fat and dispatcher.fat_workspace is None)):
            raise RuntimeError('Prewarm EXL3 workspaces and immutable banks before model capture')
        if dispatcher not in self.dispatchers:
            if self.dispatchers:
                raise RuntimeError('Only the shared target EXL3 dispatcher is admitted')
            self.dispatchers.append(dispatcher)

    def enqueue_draft_stage(self, stage, indices, layer):
        """Keep raw expert callbacks and consumer kernels in one owned fence."""
        from .native_draft_stage import NativeDraftStage
        if (type(stage) is not NativeDraftStage or current_owner() is not self
                or not self.capture_only or self.captured or self.failed or self.closed
                or stage.device != self.device):
            raise RuntimeError('Draft callback is outside its owned capture boundary')
        if stage not in self.draft_stages:
            if self.draft_stages:
                raise ValueError('Only one shared draft expert stage may be captured')
            stage.lock.acquire()
            self.locked_stages.append(stage)
            if stage.failed or stage.closed:
                raise RuntimeError('Cannot capture a failed or closed draft stage')
            stage._wait()
            stage.graphs += 1
            self.draft_stages.append(stage)
        stage._validate(indices, layer)
        return stage._enqueue(indices, layer)

    @contextmanager
    def execution(self, *, capture_only):
        import torch
        with _execution_lock, torch.cuda.device(self.device):
            if (self.failed or self.closed or current_owner() is not None
                    or type(capture_only) is not bool or self.captured == capture_only):
                raise RuntimeError('Invalid, nested or poisoned graph execution')
            self.capture_only = capture_only
            token = _current.set(self)
            try:
                stream = torch.cuda.current_stream(self.device)
                if self.pending:
                    stream.wait_event(self.event)
                for stage in (*self.stages, *self.vocab_stages, *self.draft_stages):
                    stage.lock.acquire()
                    self.locked_stages.append(stage)
                    if stage.failed or stage.closed:
                        raise RuntimeError('Captured SSD stage is unavailable')
                    stage._wait()
                for dispatcher in self.dispatchers:
                    dispatcher.lock.acquire()
                    self.locked_dispatchers.append(dispatcher)
                    if dispatcher.failed:
                        raise RuntimeError('Captured EXL3 dispatcher is unavailable')
                    work = dispatcher.workspace
                    if work.pending:
                        stream.wait_event(work.ready)
                yield self
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError('Native graph did not close its capture scope')
                if not capture_only and self.used:
                    flags = self.flags[:self.used].cpu().tolist()
                    for start, count, messages in self.checks:
                        _raise_flags(flags[start:start + count], messages)
                # Capture itself does not execute the kernels. Its output is
                # only native warmup output; real requests require a replay.
                self.captured = True
                for stage in (*self.stages, *self.vocab_stages, *self.draft_stages):
                    stage._record(stream)
                for dispatcher in self.dispatchers:
                    work = dispatcher.workspace
                    work.ready.record(stream)
                    work.pending, work.stream_id = True, stream.cuda_stream
                self.event.record(stream)
                self.pending = True
            except BaseException:
                self.failed = True
                # Keep all raw callback owners alive after any CUDA failure.
                # Do not launch cleanup CUDA work here.
                raise
            finally:
                _current.reset(token)
                for dispatcher in reversed(self.locked_dispatchers):
                    dispatcher.lock.release()
                self.locked_dispatchers.clear()
                for stage in reversed(self.locked_stages):
                    stage.lock.release()
                self.locked_stages.clear()

    def wait_before_graph_destruction(self):
        if self.failed or self.closed or current_owner() is not None:
            raise RuntimeError('Cannot clear active, poisoned or closed model graphs')
        if self.pending:
            self.event.synchronize()

    def release_after_graph_destruction(self):
        """Caller must first destroy the exact native CUDA graph entry."""
        if self.failed or self.closed or current_owner() is not None:
            raise RuntimeError('Cannot release active, poisoned or closed graph ownership')
        for stage in (*self.stages, *self.vocab_stages, *self.draft_stages):
            with stage.lock:
                if stage.graphs < 1:
                    raise RuntimeError('SSD graph ownership counter was lost')
                stage.graphs -= 1
        self.stages.clear()
        self.vocab_stages.clear()
        self.draft_stages.clear()
        self.dispatchers.clear()
        self.flags = None
        self.closed = True
        _live_owners.remove(self)
