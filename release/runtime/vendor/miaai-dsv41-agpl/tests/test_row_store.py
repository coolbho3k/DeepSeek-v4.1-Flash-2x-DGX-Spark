# SPDX-License-Identifier: AGPL-3.0-only
# Vendored from MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks
# Commit: 979e68a62c90b24d928f5638596e0ceed90e9f34
# Copyright/attribution: Mia's AI Lab and upstream contributors.
# See ../LICENSE and ../LICENSE.MIT; original body below is unchanged.
# Local change: this provenance/license prefix only.

#!/usr/bin/env python3
"""CPU parity for librow_store (packed + DEAD_ID=-1 zeros)."""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path
import random
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (
    Path("/opt/dsv41/librow_store.so"),
    ROOT / "overlay" / "librow_store.so",
)


def _lib():
    for path in CANDIDATES:
        if path.is_file():
            return C.CDLL(str(path)), path
    raise SystemExit("librow_store.so missing — build the serving image first")


lib, lib_path = _lib()
U = C.c_uint64
P = C.c_void_p
lib.row_store_open.argtypes = [C.c_char_p, U, U, U, U]
lib.row_store_open.restype = P
lib.row_store_close.argtypes = [P]
lib.row_store_stats.argtypes = [P, C.POINTER(U)]
lib.row_store_range.argtypes = [P, U, U]


class Work(C.Structure):
    _fields_ = [("store", P), ("ids", P), ("weights", P), ("scales", P), ("count", U)]


lib.row_store_lookup.argtypes = [C.POINTER(Work)]


def main() -> None:
    rows = 4099
    rng = random.Random(413)
    weights, scales = rng.randbytes(rows * 256), rng.randbytes(rows * 8)
    offset = 777
    with tempfile.NamedTemporaryFile() as f:
        f.write(bytes(offset) + weights + scales)
        f.flush()
        store = lib.row_store_open(
            f.name.encode(), rows, offset, offset + len(weights), 272 * 64
        )
        assert store, lib_path
        lib.row_store_range(store, 0, rows)
        ids = [-1, 0, rows - 1, 15]
        indices = (C.c_int64 * len(ids))(*ids)
        w = C.create_string_buffer(len(ids) * 256)
        s = C.create_string_buffer(len(ids) * 8)
        work = Work(store, C.addressof(indices), C.addressof(w), C.addressof(s), len(ids))
        lib.row_store_lookup(C.byref(work))
        assert w.raw[:256] == bytes(256), "DEAD_ID must zero the row"
        assert w.raw[256:512] == weights[:256]
        assert w.raw[512:768] == weights[(rows - 1) * 256 : rows * 256]
        assert w.raw[768:] == weights[15 * 256 : 16 * 256]
        lib.row_store_close(store)
    print(f"test_row_store: ok ({lib_path})")


if __name__ == "__main__":
    os.environ.setdefault("DSV41_RESIDENT_SCALES", "0")
    main()
