# SPDX-License-Identifier: AGPL-3.0-only
"""Unselected CPU ABI for native exact BF16 row callbacks; no CUDA imports."""
import ctypes as C
import hashlib
from pathlib import Path
import re
import threading
from .ssd_vocab_rows import DirectVocabRows, ROW_BYTES, MAX_BATCH


class Work(C.Structure):
    _fields_ = [('store', C.c_void_p), ('ids', C.c_void_p),
                ('output', C.c_void_p), ('status', C.c_void_p), ('count', C.c_uint64)]


def load_library(path, expected_sha256):
    path = Path(path).resolve()
    if (not re.fullmatch('[0-9a-f]{64}', expected_sha256)
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256):
        raise ValueError('Native vocabulary library checksum mismatch')
    lib = C.CDLL(str(path))
    ptr, u64, i64 = C.c_void_p, C.c_uint64, C.c_int64
    lib.ds41_vocab_row_abi.restype = u64
    if lib.ds41_vocab_row_abi() != 1:
        raise ValueError('Unexpected native vocabulary ABI')
    lib.ds41_vocab_row_open.argtypes = [C.c_char_p, u64, u64, i64, i64, u64, u64]
    lib.ds41_vocab_row_open.restype = ptr
    lib.ds41_vocab_row_lookup.argtypes = [ptr]
    lib.ds41_vocab_row_lookup.restype = None
    lib.ds41_vocab_row_stats.argtypes = [ptr, C.POINTER(u64)]
    lib.ds41_vocab_row_stats.restype = None
    lib.ds41_vocab_row_close.argtypes = [ptr]
    lib.ds41_vocab_row_close.restype = None
    return lib


class NativeVocabRows:
    def __init__(self, checkpoint, library, expected_sha256, *, rank, threads=16):
        if type(rank) is not int or rank not in (0, 1) or type(threads) is not int or not 0 <= threads <= 16:
            raise ValueError('Native vocabulary requires TP2 and0..16 I/O threads')
        # The CPU reference constructor reads only metadata; its executor is
        # lazy, so no Python threads are launched by this open/close sequence.
        with DirectVocabRows(checkpoint, threads=1) as metadata:
            self.path = metadata.path
            self.offset = metadata.offset
            self.file_identity = metadata.file_identity
        self.lib = load_library(library, expected_sha256)
        self.lock = threading.Lock()
        self.store = self.lib.ds41_vocab_row_open(str(self.path).encode(), self.offset,
                    *self.file_identity[2:], rank, threads)
        if not self.store:
            raise RuntimeError('Native vocabulary open failed its file/resource validation')
        self.rank, self.threads = rank, threads

    def read(self, ids):
        if (type(ids) not in (list, tuple) or len(ids) > MAX_BATCH
                or any(type(v) is not int or not -(2**63) <= v < 2**63 for v in ids)):
            raise ValueError('At most256 signed64-bit token IDs are supported')
        with self.lock:
            if not self.store:
                raise RuntimeError('Native vocabulary reader closed')
            indices = (C.c_int64 * max(1, len(ids)))(*ids)
            output = C.create_string_buffer(max(1, len(ids)*ROW_BYTES))
            status = C.c_uint32(0)
            work = Work(self.store, C.addressof(indices), C.addressof(output),
                        C.addressof(status), len(ids))
            self.lib.ds41_vocab_row_lookup(C.byref(work))
            if status.value:
                raise RuntimeError(f'Native vocabulary callback failure flags={status.value}')
            return output.raw[:len(ids)*ROW_BYTES]

    def stats(self):
        with self.lock:
            if not self.store:
                raise RuntimeError('Native vocabulary reader closed')
            result = (C.c_uint64 * 5)()
            self.lib.ds41_vocab_row_stats(self.store, result)
            return list(result)

    def close(self):
        with self.lock:
            if self.store:
                self.lib.ds41_vocab_row_close(self.store)
                self.store = None

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
