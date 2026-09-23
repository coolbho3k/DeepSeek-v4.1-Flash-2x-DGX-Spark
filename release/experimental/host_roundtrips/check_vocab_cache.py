# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only byte parity and latency for the cached/parallel vocabulary store.

Compares every returned row with an independent buffered pread of the same
checkpoint. Covers cold/warm rows, duplicates, direct-mapped slot conflicts,
TP ownership masks, image/dead IDs, file mutation after warm-up and poison.
No CUDA, model loading or serving access.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import struct
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from ds41.native_vocab_rows import NativeVocabRows, Work
from ds41.ssd_vocab_rows import ROWS, WIDTH, ROW_BYTES

SLOTS = 4096


def cache_stats(reader):
    out = (C.c_uint64 * 3)()
    reader.lib.ds41_vocab_row_cache_stats.argtypes = [C.c_void_p, C.POINTER(C.c_uint64)]
    reader.lib.ds41_vocab_row_cache_stats(reader.store, out)
    return list(out)


def timed(reader, ids, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        reader.read(ids)
        samples.append((time.perf_counter() - start) * 1e6)
    return dict(median_us=statistics.median(samples), max_us=max(samples))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    digest = hashlib.sha256(args.library.read_bytes()).hexdigest()
    rng = random.Random(8191)
    cases = []
    timings = []
    with args.checkpoint.open('rb', buffering=0) as reference:
        for threads in (0, 4, 16):
            for rank in (0, 1):
                lo, hi = rank * (ROWS // 2), (rank + 1) * (ROWS // 2)
                with NativeVocabRows(args.checkpoint, args.library, digest, rank=rank, threads=threads) as reader:
                    def expect(ids):
                        return b''.join(os.pread(reference.fileno(), ROW_BYTES, reader.offset + r * ROW_BYTES)
                                        if lo <= r < hi else bytes(ROW_BYTES) for r in ids)
                    base = rng.randrange(lo, hi - 3 * SLOTS)
                    batches = [
                        [], [base], [base], [base, base, base + SLOTS],        # hit, dup, conflict
                        [base + SLOTS, base],                                   # evict and refill
                        [-1, ROWS, lo - 1 if lo else hi, hi - 1, lo, 129264],   # dead/unowned/image
                        rng.sample(range(ROWS), 4), rng.sample(range(ROWS), 3),
                        rng.sample(range(lo, hi), 256),
                    ]
                    hot = rng.sample(range(lo, hi), 64)
                    for _ in range(40):
                        batches.append(rng.sample(hot, 4) + [base + 2 * SLOTS])
                    for ids in batches:
                        assert reader.read(ids) == expect(ids), (threads, rank, ids[:8])
                    hits, misses, cache_bytes = cache_stats(reader)
                    assert hits > 0 and misses > 0 and cache_bytes == SLOTS * ROW_BYTES
                    status = C.c_uint32(0)
                    bad = Work(reader.store, None, None, C.addressof(status), 257)
                    reader.lib.ds41_vocab_row_lookup(C.byref(bad))
                    assert status.value == 1
                    stats = reader.stats()
                    assert stats[2] <= 512 * 1024 and stats[3:] == [threads, 0]
                    cases.append(dict(threads=threads, rank=rank, batches=len(batches), all_bytes_exact=True,
                                      cache_hits=hits, cache_misses=misses))
                    cold4 = [rng.sample(range(lo, hi), 4) for _ in range(15)]
                    cold3 = [rng.sample(range(lo, hi), 3) for _ in range(15)]
                    def med(batches):
                        samples = []
                        for ids in batches:
                            start = time.perf_counter()
                            reader.read(ids)
                            samples.append((time.perf_counter() - start) * 1e6)
                        return statistics.median(samples)
                    timings.append(dict(threads=threads, rank=rank, cold_4_rows_median_us=med(cold4),
                                        cold_3_rows_median_us=med(cold3), warm_4_rows=timed(reader, cold4[0], 15)))
    header = json.dumps({'embed.weight': dict(dtype='BF16', shape=[ROWS, WIDTH],
                        data_offsets=[0, ROWS * ROW_BYTES])}).encode()
    with tempfile.TemporaryDirectory(prefix='vocab-cache-fixture-', dir=args.output.parent) as tmp:
        fixture = Path(tmp) / 'fixture.safetensors'
        with fixture.open('wb') as stream:
            stream.write(struct.pack('<Q', len(header)) + header)
            stream.truncate(8 + len(header) + ROWS * ROW_BYTES)
        with NativeVocabRows(fixture, args.library, hashlib.sha256(args.library.read_bytes()).hexdigest(),
                             rank=0, threads=4) as reader:
            assert reader.read([0, 1]) == bytes(2 * ROW_BYTES)
            assert reader.read([0, 1]) == bytes(2 * ROW_BYTES)  # cached
            with fixture.open('r+b') as stream:
                stream.truncate(8 + len(header) + ROW_BYTES)
            for flags in (2, 8):  # a warm cache must not mask a changed file
                try:
                    reader.read([0, 1])
                except RuntimeError as error:
                    assert f'flags={flags}' in str(error)
                else:
                    raise AssertionError('Mutation/poison was not surfaced after warm-up')
    result = dict(status='vocab_cache_cpu_parity_pass', binary_sha256=digest, cases=cases, timings=timings,
                  mutation_after_warm_cache_detected=True, cuda_imported='torch' in sys.modules,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  vm_hwm_bytes=next(int(l.split()[1]) * 1024 for l in Path('/proc/self/status').read_text().splitlines()
                                    if l.startswith('VmHWM:')))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], timings=timings)), flush=True)


if __name__ == '__main__':
    main()
