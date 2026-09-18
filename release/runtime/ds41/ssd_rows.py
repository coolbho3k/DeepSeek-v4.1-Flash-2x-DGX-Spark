"""Bounded raw safetensors row lookup using Linux direct I/O.

O_DIRECT avoids letting random engram accesses fill the shared CPU/GPU RAM
with filesystem page cache. Only the explicitly sized LRU retains pages.
Each reader is single-threaded; use independent readers for parallel workers.
"""
from collections import OrderedDict
import json
import mmap
import os
from pathlib import Path
import struct


class DirectRows:
    page_size = 4096

    def __init__(self, path, tensor_name, cache_bytes=32 * 1024**2):
        self.path = Path(path)
        with self.path.open("rb") as stream:
            header_size, = struct.unpack("<Q", stream.read(8))
            if not 0 < header_size < 64 * 1024**2:
                raise ValueError("Invalid safetensors header size")
            header = json.loads(stream.read(header_size))
        info = header[tensor_name]
        if len(info["shape"]) != 2 or info["dtype"] not in ("F8_E4M3", "F8_E8M0", "U8", "I8"):
            raise ValueError(f"Expected a byte-valued 2D table: {info}")
        self.rows, self.row_bytes = info["shape"]
        begin, end = info["data_offsets"]
        if end - begin != self.rows * self.row_bytes:
            raise ValueError("Tensor shape and byte range disagree")
        self.offset = 8 + header_size + begin
        if 8 + header_size + end > self.path.stat().st_size:
            raise ValueError("Truncated safetensors file")
        if cache_bytes < 0:
            raise ValueError("cache_bytes must be nonnegative")
        self.max_pages = cache_bytes // self.page_size
        self.cache = OrderedDict()
        self.read_calls = 0
        self.disk_bytes = 0
        self.cache_hits = 0
        self.fd = os.open(self.path, os.O_RDONLY | os.O_DIRECT)
        # Anonymous mmap provides a page-aligned buffer required by O_DIRECT.
        self.buffer = mmap.mmap(-1, self.page_size)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
            self.buffer.close()
            self.cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _page(self, offset):
        if offset in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(offset)
            return self.cache[offset]
        count = os.preadv(self.fd, [self.buffer], offset)
        self.read_calls += 1
        self.disk_bytes += count
        if count == 0:
            raise EOFError(f"Reading {self.path} at {offset}")
        page = self.buffer[:count]
        if self.max_pages:
            self.cache[offset] = page
            if len(self.cache) > self.max_pages:
                self.cache.popitem(last=False)
        return page

    def read(self, row_ids):
        """Return rows in original order; sort unique IDs to coalesce page hits."""
        ids = [int(row) for row in row_ids]
        if any(row < 0 or row >= self.rows for row in ids):
            raise IndexError("Engram row out of range")
        values = {}
        for row in sorted(set(ids)):
            position = self.offset + row * self.row_bytes
            remaining = self.row_bytes
            pieces = []
            while remaining:
                page_offset = position // self.page_size * self.page_size
                page = self._page(page_offset)
                start = position - page_offset
                take = min(remaining, len(page) - start)
                if take <= 0:
                    raise EOFError(f"Truncated row {row}")
                pieces.append(page[start:start + take])
                remaining -= take
                position += take
            values[row] = b"".join(pieces)
        return b"".join(values[row] for row in ids)


class EngramRows:
    """Read official FP8 rows/scales, dequantizing just the requested rows."""

    def __init__(self, model_dir, layer, cache_bytes=64 * 1024**2):
        model_dir = Path(model_dir)
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
        key = f"layers.{layer}.engram.embed"
        self.weight = DirectRows(model_dir / index[key + ".weight"], key + ".weight", cache_bytes // 2)
        try:
            self.scale = DirectRows(model_dir / index[key + ".scale"], key + ".scale", cache_bytes // 2)
            if self.weight.rows != self.scale.rows or self.weight.row_bytes != self.scale.row_bytes * 32:
                raise ValueError("Engram weights/scales must use groups of 32")
        except Exception:
            self.weight.close()
            if hasattr(self, "scale"):
                self.scale.close()
            raise

    def lookup(self, indices, device="cpu"):
        import torch
        shape = tuple(indices.shape)
        ids = indices.detach().to(device="cpu", dtype=torch.int64).reshape(-1).tolist()
        if not ids:
            return torch.empty((*shape, self.weight.row_bytes), dtype=torch.bfloat16, device=device)
        # bytearray owns writable storage; views remain alive through dequantization.
        weights = torch.frombuffer(bytearray(self.weight.read(ids)), dtype=torch.float8_e4m3fn)
        scales = torch.frombuffer(bytearray(self.scale.read(ids)), dtype=torch.float8_e8m0fnu)
        values = weights.float().reshape(len(ids), -1, 32) * scales.float().reshape(len(ids), -1, 1)
        return values.reshape(*shape, self.weight.row_bytes).to(device=device, dtype=torch.bfloat16)

    def close(self):
        self.weight.close()
        self.scale.close()
