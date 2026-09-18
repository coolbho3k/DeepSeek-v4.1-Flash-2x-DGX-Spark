# SPDX-License-Identifier: AGPL-3.0-only
"""Offline, lossless BF16 vocabulary-row reader; NOT a serving integration.

Reads the unchanged checkpoint through O_DIRECT without mapping the table or
retaining a row cache. This CPU reference establishes byte parity and I/O cost.
A native graph callback, TP integration, and memory admission are still needed
before it could replace a resident input embedding. Never use for the LM head.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import mmap
import os
from pathlib import Path
import stat
import struct
import threading

ROWS = 129280
WIDTH = 5120
ROW_BYTES = WIDTH * 2
MAX_BATCH = 256
ALIGNMENT = 4096
BUFFER_BYTES = ((ROW_BYTES + 2 * ALIGNMENT - 2) // ALIGNMENT) * ALIGNMENT


def identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate safetensors header key')
        result[key] = value
    return result


def layout(header, header_bytes, file_bytes):
    if type(header) is not dict:
        raise ValueError('Safetensors header must be an object')
    tensor = header.get('embed.weight')
    if (not isinstance(tensor, dict) or tensor.get('dtype') != 'BF16'
            or tensor.get('shape') != [ROWS, WIDTH]):
        raise ValueError('Only the original full-vocabulary BF16 input embedding is supported')
    offsets = tensor.get('data_offsets')
    if (type(offsets) is not list or len(offsets) != 2
            or any(type(v) is not int for v in offsets)
            or offsets[0] < 0 or offsets[1] - offsets[0] != ROWS * ROW_BYTES
            or 8 + header_bytes + offsets[1] > file_bytes):
        raise ValueError('Invalid or truncated vocabulary tensor range')
    return 8 + header_bytes + offsets[0]


class DirectVocabRows:
    def __init__(self, path, *, threads=8):
        if type(threads) is not int or not 1 <= threads <= 16:
            raise ValueError('Use 1..16 bounded I/O workers')
        self.path = Path(path).absolute()
        self.fd = None
        self.pool = None
        self.buffers = []
        self.lock = threading.Lock()
        self.read_calls = self.disk_bytes = 0
        self.failed = False
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        # Only the bounded header is read through the buffered descriptor.
        descriptor = os.open(self.path, flags)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError('Vocabulary checkpoint must be a regular file')
            prefix = os.pread(descriptor, 8, 0)
            if len(prefix) != 8:
                raise ValueError('Missing safetensors header length')
            count, = struct.unpack('<Q', prefix)
            if not 2 <= count <= 16 * 2**20:
                raise ValueError('Unbounded safetensors header')
            raw = os.pread(descriptor, count, 8)
            if len(raw) != count:
                raise ValueError('Truncated safetensors header')
            header = json.loads(raw, object_pairs_hook=unique_object)
            self.offset = layout(header, count, before.st_size)
            self.file_identity = identity(before)
            self.fd = os.open(self.path, flags | os.O_DIRECT)
            if identity(os.fstat(self.fd)) != self.file_identity:
                raise ValueError('Vocabulary checkpoint changed during opening')
            self.buffers = [mmap.mmap(-1, BUFFER_BYTES) for _ in range(threads)]
            self.pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix='ds41-vocab-io')
        except BaseException:
            self.close()
            raise
        finally:
            os.close(descriptor)

    def _read_group(self, ids, buffer):
        result = {}
        read_bytes = 0
        for row in ids:
            start = self.offset + row * ROW_BYTES
            aligned = start // ALIGNMENT * ALIGNMENT
            inside = start - aligned
            needed = ((inside + ROW_BYTES + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
            with memoryview(buffer)[:needed] as view:
                count = os.preadv(self.fd, [view], aligned)
            if count < inside + ROW_BYTES:
                raise EOFError('Short O_DIRECT vocabulary row read')
            result[row] = buffer[inside:inside + ROW_BYTES]
            read_bytes += count
        return result, len(ids), read_bytes

    def read(self, ids, *, rank):
        """Return exact BF16 bytes, with the native TP2 unowned-row zero mask.

        No local row remapping or numeric conversion: ids are global token IDs.
        Rank outputs must still be combined by the native embedding collective.
        """
        if type(ids) not in (tuple, list) or len(ids) > MAX_BATCH:
            raise ValueError('An explicit batch of at most256 global IDs is required')
        if type(rank) is not int or rank not in (0, 1):
            raise ValueError('This candidate supports only the exact TP2 vocabulary')
        # Native TP2 masks every out-of-vocabulary ID, including graph/image
        # placeholders, to zero. Do not turn such IDs into disk addresses.
        if any(type(v) is not int or not -(2**63) <= v < 2**63 for v in ids):
            raise IndexError('Expected signed64-bit global token IDs')
        with self.lock:
            if self.fd is None or self.failed:
                raise RuntimeError('Vocabulary row reader is closed or failed')
            if identity(os.fstat(self.fd)) != self.file_identity:
                raise ValueError('Vocabulary checkpoint changed after validation')
            lo, hi = rank * (ROWS // 2), (rank + 1) * (ROWS // 2)
            owned = sorted({v for v in ids if lo <= v < hi})
            futures = []
            try:
                for i, buffer in enumerate(self.buffers):
                    if i < len(owned):
                        futures.append(self.pool.submit(self._read_group,
                                       owned[i::len(self.buffers)], buffer))
            except BaseException:
                # submit() may queue a task before failing to create a thread.
                # Drain the entire executor, including any such unreturned
                # future, before allowing buffers to be closed or reused.
                self.failed = True
                self.pool.shutdown(wait=True)
                raise
            values = {}
            failure = None
            # Drain every task before returning/raising: buffers may only be
            # reused or closed once all prior preadv operations have finished.
            for future in futures:
                try:
                    rows, calls, count = future.result()
                    values.update(rows)
                    self.read_calls += calls
                    self.disk_bytes += count
                except BaseException as error:
                    failure = error
            if failure is not None:
                self.failed = True
                raise failure
            if identity(os.fstat(self.fd)) != self.file_identity:
                raise ValueError('Vocabulary checkpoint changed during row reads')
            zero = bytes(ROW_BYTES)
            return b''.join(values[v] if lo <= v < hi else zero for v in ids)

    def close(self):
        with self.lock:
            if self.pool is not None:
                self.pool.shutdown(wait=True)
                self.pool = None
            for buffer in self.buffers:
                buffer.close()
            self.buffers = []
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
