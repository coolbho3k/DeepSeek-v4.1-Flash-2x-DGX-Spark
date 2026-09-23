# SPDX-License-Identifier: AGPL-3.0-only
# Callback/fence pattern adapted from the attributed MiaAI integration in
# serving/miaai_engram.py. Original notices remain there and under vendor/.
"""Unselected native BF16 input-embedding stage; GPU qualification pending.

No install hook. Construction belongs to model loading and native memory
profiling. The caller owns the TP collective; the output is the local shard's
exact BF16 rows, with all unowned/padding/image IDs masked to zero.
"""
import ctypes as C
import threading
import torch
from .native_vocab_rows import NativeVocabRows, Work
from .ssd_vocab_rows import ROW_BYTES, WIDTH, MAX_BATCH
from . import graph_validation as validation

BINARY_SHA = '28db58a40c22e8f1506eddbbd3ed5bf74bba5e3f6dd5b112c48408e718e8e443'
from .combined_config import MAX_TOKENS
_LIVE_STAGES = set()
ERRORS = ((1, 'Invalid native vocabulary callback descriptor'),
          (2, 'Input embedding checkpoint changed during serving'),
          (4, 'Input embedding O_DIRECT read failed or was short'),
          (8, 'Input embedding native store is poisoned'))


class NativeVocabStage:
    def __init__(self, checkpoint, library, *, rank, device=None):
        if torch.cuda.is_current_stream_capturing() or _LIVE_STAGES:
            raise RuntimeError('Create exactly one shared input vocabulary stage before capture')
        self.device = torch.device('cuda', torch.cuda.current_device()) if device is None else torch.device(device)
        if self.device.type != 'cuda' or self.device.index != torch.cuda.current_device():
            raise ValueError('Vocabulary stage requires the current visible CUDA device')
        from miaai_engram import load_cudart
        self.cuda = load_cudart()
        self.native = NativeVocabRows(checkpoint, library, BINARY_SHA, rank=rank, threads=16)
        self.lock = threading.Lock()
        self.closed = self.failed = self.pending = False
        self.graphs = 0
        self.works = {}
        try:
            self.ids = torch.empty(MAX_BATCH, dtype=torch.int64, device='cpu', pin_memory=True)
            self.host_rows = torch.empty((MAX_BATCH, WIDTH), dtype=torch.bfloat16,
                                         device='cpu', pin_memory=True)
            self.host_status = torch.zeros(1, dtype=torch.int32, device='cpu', pin_memory=True)
            self.device_errors = torch.zeros(MAX_TOKENS//MAX_BATCH, dtype=torch.int32, device=self.device)
            self.event = torch.cuda.Event(external=True)
            self.staging_bytes = MAX_BATCH*(8+ROW_BYTES)+4+self.device_errors.numel()*4
            assert self.staging_bytes <= 3*2**20
            _LIVE_STAGES.add(self)
        except BaseException:
            self.native.close()
            raise

    def _validate(self, indices, out):
        if self.failed or self.closed or not self.native.store:
            raise RuntimeError('Vocabulary stage is failed or closed')
        if (indices.device != self.device or indices.dtype not in (torch.int32, torch.int64)
                or indices.ndim != 1 or not indices.is_contiguous()
                or not 0 <= indices.numel() <= MAX_TOKENS
                or out.device != self.device or out.dtype != torch.bfloat16
                or tuple(out.shape) != (indices.numel(), WIDTH) or not out.is_contiguous()):
            raise ValueError('Invalid bounded BF16 vocabulary staging contract')

    def _wait(self):
        stream = torch.cuda.current_stream(self.device)
        if self.pending:
            stream.wait_event(self.event)
        return stream

    def _record(self, stream):
        self.event.record(stream)
        self.pending = True

    def _work(self, count):
        if type(count) is not int or not 1 <= count <= MAX_BATCH:
            raise ValueError('Native vocabulary callback chunk exceeds staging')
        if count not in self.works:
            self.works[count] = Work(self.native.store, self.ids.data_ptr(),
                self.host_rows.data_ptr(), self.host_status.data_ptr(), count)
        return self.works[count]

    def _enqueue(self, indices, out):
        for chunk, start in enumerate(range(0, indices.numel(), MAX_BATCH)):
            count = min(MAX_BATCH, indices.numel()-start)
            descriptor = self._work(count)
            # Copy int64 IDs on-device before the asynchronous D2H transfer.
            # Never perform a CPU cast/read of IDs during graph capture.
            self.ids[:count].copy_(indices[start:start+count].to(torch.int64), non_blocking=True)
            error = self.cuda.cudaLaunchHostFunc(torch.cuda.current_stream(self.device).cuda_stream,
                C.cast(self.native.lib.ds41_vocab_row_lookup, C.c_void_p), C.addressof(descriptor))
            if error:
                self.failed = True
                raise RuntimeError(f'Vocabulary host callback enqueue failed: {error}')
            out[start:start+count].copy_(self.host_rows[:count], non_blocking=True)
            flags = self.device_errors[chunk:chunk+1]
            flags.copy_(self.host_status, non_blocking=True)
            # The host callback sets status before these same-stream copies.
            # Captured flags are copied into distinct owner slots before reuse.
            validation.check_flags(flags, ERRORS)

    def lookup(self, indices, out):
        self._validate(indices, out)
        if torch.cuda.is_current_stream_capturing():
            validation.require_capture_owner()
            return validation.current_owner().enqueue_vocab_stage(self, indices, out)
        with self.lock:
            stream = self._wait()
            try:
                self._enqueue(indices, out)
                self._record(stream)
            except BaseException:
                self.failed = True
                # Retain native descriptors/buffers after queued CUDA failure.
                raise

    def close(self):
        with self.lock:
            if self.closed:
                return
            if self.failed or self.graphs:
                raise RuntimeError('Retain failed or graph-owned vocabulary callback resources')
            if self.pending:
                self.event.synchronize()
            self.native.close()
            self.closed = True
            _LIVE_STAGES.remove(self)
