# SPDX-License-Identifier: AGPL-3.0-only
# Adapted from MiaAI-Lab's Engram callback/dequant path at
# 979e68a62c90b24d928f5638596e0ceed90e9f34. Attribution: Mia's AI Lab and
# upstream contributors. See ../vendor/miaai-dsv41-agpl/LICENSE and LICENSE.MIT.
# Local changes: bounded chunk staging, original ownership masking, exact
# E8M0 edge cases, explicit stream ordering and managed graph/resource lifetime.
"""Native parallel Engram staging; no table-sized pinned or GPU allocations."""
import ctypes as C
import hashlib
import os
from pathlib import Path
import threading

import torch
import triton
import triton.language as tl

BINARY_SHA = 'b66c3eac86ed189277cb456c5d3b866ed494eed346e99449504cb2c7aa8b1f71'
MAX_HEADS = 144
MAX_CHUNK = 256
MAX_TOKENS = 1056
# Keep callback descriptors, buffers and native stores alive after any failure.
# A successful explicit close removes the owner; fatal stages die with worker.
_LIVE_STAGES = set()


class Work(C.Structure):
    _fields_ = [('store', C.c_void_p), ('ids', C.c_void_p),
               ('weights', C.c_void_p), ('scales', C.c_void_p), ('count', C.c_uint64)]


@triton.jit
def _dequant(weight, scales, out, count, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < count * 256
    value = tl.load(weight + index, mask=valid, other=0.0).to(tl.float32)
    code = tl.load(scales + index // 32, mask=valid, other=0).to(tl.int32)
    scale = (code << 23).to(tl.float32, bitcast=True)
    # E8M0 code0 is 2**-127, not zero; code255 is NaN, not infinity.
    scale = tl.where(code == 0, 5.877471754111438e-39, scale)
    scale = tl.where(code == 255, float('nan'), scale)
    tl.store(out + index, (value * scale).to(tl.bfloat16), mask=valid)


def load_native(path):
    path = Path(path).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != BINARY_SHA:
        raise ValueError('Unqualified native Engram binary')
    lib = C.CDLL(str(path))
    P, U = C.c_void_p, C.c_uint64
    lib.ds41_row_store_open.argtypes = [C.c_char_p, U, U, U, U]
    lib.ds41_row_store_open.restype = P
    lib.ds41_row_store_range.argtypes = [P, U, U]
    lib.ds41_row_store_lookup.argtypes = [P]
    lib.ds41_row_store_close.argtypes = [P]
    lib.row_store_stats.argtypes = [P, C.POINTER(U)]
    lib.ds41_row_store_abi.restype = U
    lib.ds41_row_store_attach_packed.argtypes = [P, C.c_char_p, U]
    lib.ds41_row_store_attach_packed.restype = C.c_int
    lib.ds41_row_store_profile.argtypes = [P, C.POINTER(U)]
    lib.ds41_row_store_clear_cache.argtypes = [P]
    if lib.ds41_row_store_abi() != 2:
        raise ValueError('Unexpected native Engram ABI')
    return lib


def load_cudart():
    # Resolve the runtime already mapped by PyTorch, not a different installed
    # CUDA version. No arbitrary filesystem library scan or driver calls.
    paths = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
             if 'libcudart.so' in line and line.split()[-1].startswith('/')}
    if len(paths) != 1:
        raise RuntimeError('Expected exactly one loaded CUDA runtime')
    lib = C.CDLL(paths.pop())
    lib.cudaLaunchHostFunc.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p]
    lib.cudaLaunchHostFunc.restype = C.c_int
    return lib


class NativeStage:
    """One immutable table/TP partition with fixed reusable pinned staging.

    The caller retains the stage until close. Explicit close waits for queued
    work and refuses while a managed graph still owns a callback. All eager and
    managed replay calls serialize buffer reuse across CUDA streams.
    """
    def __init__(self, embedding, library, *, device=None, packed=None, packed_layer=None, mapped=False):
        if (embedding.dim != 256 or not 1 <= embedding.part_n_hash_cols <= MAX_HEADS
                or not 1 <= embedding.chunk_tokens <= MAX_CHUNK):
            raise ValueError('Unsupported bounded native Engram partition')
        if (os.environ.get('OFFLOAD_MODE') != 'ssd'
                or os.environ.get('DSV41_RESIDENT_SCALES') != '0'
                or os.environ.get('DSV41_IO_THREADS') not in ('32', '64')):
            raise ValueError('Explicit bounded native Engram modes are required')
        if type(mapped) is not bool or ((packed is None) != (packed_layer is None)):
            raise ValueError('Explicit mapped mode and paired packed path/layer required')
        self.mapped = mapped
        self.embedding = embedding
        self.mode = tuple(os.environ[k] for k in ('OFFLOAD_MODE','DSV41_RESIDENT_SCALES','DSV41_IO_THREADS'))
        self.device = torch.device('cuda', torch.cuda.current_device()) if device is None else torch.device(device)
        if self.device.type != 'cuda' or self.device.index != torch.cuda.current_device():
            raise ValueError('Exactly the current CUDA device is supported')
        self.lib = load_native(library)
        self.cuda = load_cudart()
        self.lock = threading.Lock()
        self.works = {}
        self.graphs = 0
        self.pending = False
        self.failed = False
        self.closed = False
        self.event = torch.cuda.Event()
        self.store = None
        weight, scale = embedding.reader.weight, embedding.reader.scale
        if (weight.path != scale.path or weight.rows != scale.rows
                or weight.row_bytes != 256 or scale.row_bytes != 8):
            raise ValueError('Native rows require one official weight/scale table')
        budget = (weight.max_pages + scale.max_pages) * 4096
        if budget > 64 * 2**20:
            raise ValueError('Native row cache exceeds original table budget')
        cap = embedding.chunk_tokens * embedding.part_n_hash_cols
        try:
            self.ids = torch.empty(cap, dtype=torch.int64, device='cpu', pin_memory=True)
            self.host_w = torch.empty((cap, 256), dtype=torch.uint8, device='cpu', pin_memory=True)
            self.host_s = torch.empty((cap, 8), dtype=torch.uint8, device='cpu', pin_memory=True)
            if self.mapped:
                from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
                self.dev_w = get_accelerator_view_from_cpu_tensor(self.host_w)
                self.dev_s = get_accelerator_view_from_cpu_tensor(self.host_s)
                if self.dev_w.device != self.device or self.dev_s.device != self.device:
                    raise RuntimeError('Mapped staging is not on the current GPU')
            else:
                self.dev_w = torch.empty_like(self.host_w, device=self.device)
                self.dev_s = torch.empty_like(self.host_s, device=self.device)
            self.store = self.lib.ds41_row_store_open(str(weight.path).encode(), weight.rows,
                weight.offset, scale.offset, budget)
            if not self.store:
                raise RuntimeError('Native Engram table open failed')
            self.lib.ds41_row_store_range(self.store, embedding.vocab_start_idx, embedding.vocab_end_idx)
            if packed is not None and self.lib.ds41_row_store_attach_packed(
                    self.store, str(packed).encode(), packed_layer) != 1:
                raise ValueError('Explicit packed Engram artifact did not attach')
            self.staging_bytes = cap * (8 + 264 * (1 if self.mapped else 2))
            assert self.staging_bytes <= MAX_CHUNK * MAX_HEADS * 536
            _LIVE_STAGES.add(self)
        except BaseException:
            if self.store:
                self.lib.ds41_row_store_close(self.store)
                self.store = None
            raise

    def _validate(self, indices, out):
        e = self.embedding
        if self.closed or self.failed:
            raise RuntimeError('Native Engram stage closed or failed')
        if tuple(os.environ.get(k) for k in ('OFFLOAD_MODE','DSV41_RESIDENT_SCALES','DSV41_IO_THREADS')) != self.mode:
            raise RuntimeError('Native Engram startup mode changed')
        if (indices.device != self.device or indices.dtype not in (torch.int64, torch.int32)
                or indices.ndim != 2 or indices.shape[1] != e.n_hash_cols
                or not 0 <= len(indices) <= MAX_TOKENS
                or out.device != self.device or out.dtype != torch.bfloat16
                or tuple(out.shape) != (len(indices), e.part_n_hash_cols, 256)
                or not out.is_contiguous()):
            raise ValueError('Invalid native Engram input/output contract')

    def _wait(self):
        stream = torch.cuda.current_stream(self.device)
        if self.pending:
            stream.wait_event(self.event)
        return stream

    def _record(self, stream):
        self.event.record(stream)
        self.pending = True

    def _work(self, rows):
        if rows not in self.works:
            self.works[rows] = Work(self.store, self.ids.data_ptr(), self.host_w.data_ptr(),
                                   self.host_s.data_ptr(), rows)
        return self.works[rows]

    def _enqueue(self, indices, out):
        e = self.embedding
        for start in range(0, len(indices), e.chunk_tokens):
            stop = min(start + e.chunk_tokens, len(indices))
            local = indices[start:stop, e.head_start:e.head_end].to(torch.int64)
            if local.shape[1] != e.part_n_hash_cols:
                local = torch.nn.functional.pad(local, (0, e.part_n_hash_cols-local.shape[1]), value=-1)
            # Match the existing SSD embedding: any unowned/invalid ID is zero,
            # including >=num_embeddings. Image DEAD_ID and duplicates survive.
            local = torch.where((local >= e.vocab_start_idx) & (local < e.vocab_end_idx), local, -1)
            rows = local.numel()
            work = self._work(rows)
            self.ids[:rows].copy_(local.reshape(-1), non_blocking=True)
            error = self.cuda.cudaLaunchHostFunc(torch.cuda.current_stream(self.device).cuda_stream,
                C.cast(self.lib.ds41_row_store_lookup, C.c_void_p), C.addressof(work))
            if error:
                self.failed = True
                raise RuntimeError(f'Native Engram callback enqueue failed: {error}')
            if not self.mapped:
                self.dev_w[:rows].copy_(self.host_w[:rows], non_blocking=True)
                self.dev_s[:rows].copy_(self.host_s[:rows], non_blocking=True)
            _dequant[(triton.cdiv(rows * 256, 4096),)](
                self.dev_w.view(torch.float8_e4m3fn), self.dev_s, out[start:stop], rows,
                BLOCK=4096, enable_fp_fusion=False)

    def lookup(self, indices, out, background=False):
        self._validate(indices, out)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Use NativeStage.capture for managed callback graph lifetime')
        with self.lock:
            stream = self._wait()
            try:
                self._enqueue(indices, out)
            except BaseException:
                self.failed = True
                raise
            finally:
                # Enqueued host callbacks must remain fenced even if a later
                # Python validation/JIT step fails. No allocator errors expected
                # after qualification; the outer worker still owns fatal errors.
                if not self.failed:
                    self._record(stream)

    def capture(self, example):
        return NativeGraph(self, example)

    def close(self):
        with self.lock:
            if self.closed:
                return
            if self.graphs:
                raise RuntimeError('Close managed graphs before their native Engram stage')
            if self.failed:
                raise RuntimeError('Failed stage must remain alive until worker teardown')
            if self.pending:
                self.event.synchronize()
            self.lib.ds41_row_store_close(self.store)
            self.store = None
            self.closed = True
            _LIVE_STAGES.discard(self)


class NativeGraph:
    """Own a callback graph; stage buffer reuse stays ordered on every replay."""
    def __init__(self, stage, example):
        self.stage = stage
        self.closed = False
        with stage.lock:
            self.ids = example.clone()
            self.out = torch.empty((len(example), stage.embedding.part_n_hash_cols, 256),
                                   dtype=torch.bfloat16, device=stage.device)
            stage._validate(self.ids, self.out)
            if not len(example):
                raise ValueError('Empty callback graphs are unnecessary')
            current = stage._wait()
            capture_stream = torch.cuda.Stream(device=stage.device)
            capture_stream.wait_stream(current)
            try:
                with torch.cuda.stream(capture_stream):
                    stage._enqueue(self.ids, self.out)  # JIT and Work descriptors before capture.
                capture_stream.synchronize()
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=capture_stream):
                    stage._enqueue(self.ids, self.out)
                current.wait_stream(capture_stream)
                stage._record(current)
                stage.graphs += 1
            except BaseException:
                stage.failed = True
                raise

    def replay(self, indices):
        stage = self.stage
        with stage.lock:
            if self.closed or indices.shape != self.ids.shape or indices.dtype != self.ids.dtype:
                raise ValueError('Closed graph or changed native Engram graph shape/dtype')
            stage._validate(indices, self.out)
            stream = stage._wait()
            try:
                self.ids.copy_(indices)
                self.graph.replay()
                result = self.out.clone()  # Prior replies must survive later replays.
                stage._record(stream)
                return result
            except BaseException:
                stage.failed = True
                raise

    def close(self):
        with self.stage.lock:
            if self.closed:
                return
            if self.stage.pending:
                self.stage.event.synchronize()
            self.graph.reset()
            self.stage.graphs -= 1
            self.closed = True
