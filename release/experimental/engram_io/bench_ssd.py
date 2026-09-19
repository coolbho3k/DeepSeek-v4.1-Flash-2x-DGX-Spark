# SPDX-License-Identifier: AGPL-3.0-only
"""Paired original/MiaAI dense/local page15 reads on real rank-owned tables.

O_DIRECT bypasses filesystem page cache; clear() controls only the 64 MiB
native row cache. Physical SSD/controller caches are neither flushed nor
claimed cold. Each round rotates variant order and verifies every output byte.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np

from native import Reader
from packing import fingerprint, table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=30)
    a = p.parse_args()
    if not 6 <= a.repeats <= 100 or a.output.exists():
        raise ValueError('Fresh result and bounded repeats required')
    manifest = json.loads(a.manifest.read_text())
    if manifest['status'] != 'complete' or not manifest['independent_partition_match']:
        raise ValueError('Complete validated packed shards required')
    rank = manifest['rank']
    variants = ('original', 'dense', 'page15')
    rng = np.random.default_rng(20260918)
    results = dict(status='running', rank=rank, started_at=time.time(), seed=20260918,
        native_threads=96, row_cache_bytes_per_variant=64*2**20,
        cache_note=__doc__, manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(), cases=[])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(results, f)
    try:
        for layer, partition in manifest['partitions'].items():
            layer = int(layer)
            shards = {s['layout']: s for s in manifest['shards'] if s['layer'] == layer}
            info = table(shards['page15']['source']['path'], layer)
            for s in shards.values():
                if (fingerprint(Path(s['path']).stat()) != s['packed_fingerprint']
                        or info != s['source'] or not s['complete_byte_readback']):
                    raise ValueError('Changed source or packed artifact')
            lo, hi = partition['ranges'][rank]
            readers = {v: Reader(a.library, info, lo, hi,
                None if v == 'original' else shards[v]['path'], budget=64*2**20, threads=96) for v in variants}
            sizes = np.array(partition['head_sizes'][rank*12:(rank+1)*12], dtype=np.int64)
            offsets = np.cumsum(np.r_[lo, sizes[:-1]])
            def ids(tokens):
                return (rng.random((tokens, 12))*sizes).astype(np.int64)+offsets
            try:
                # All boundary bytes, including unowned/dead IDs, before timing.
                boundary = np.array([-1, 0, lo-1, lo, lo+1, hi-1, hi, info['rows']-1])
                # NativeStage masks >=total_rows before the C ABI; the native
                # reader deliberately aborts on those malformed IDs. Rank1's
                # exclusive upper bound is exactly total_rows.
                boundary = np.where(boundary < info['rows'], boundary, -1)
                reference = readers['original'].lookup(boundary)
                for v in variants[1:]:
                    for actual, expected in zip(readers[v].lookup(boundary), reference):
                        np.testing.assert_array_equal(actual, expected)
                for tokens in (1, 4, 24, 256):
                    for pattern in ('row_cache_cold', 'row_cache_hot', '80pct_hot'):
                        samples = {v: [] for v in variants}
                        for r in readers.values():
                            r.clear()
                        hot = ids(tokens)
                        hot_expected = readers['original'].lookup(hot)
                        for v in variants[1:]:
                            for actual, expected in zip(readers[v].lookup(hot), hot_expected):
                                np.testing.assert_array_equal(actual, expected)
                        for iteration in range(a.repeats):
                            current = ids(tokens)
                            if pattern == 'row_cache_hot':
                                current = hot
                            elif pattern == '80pct_hot':
                                current = np.where(rng.random(current.shape)<.8, hot, current)
                            order = variants[iteration%3:]+variants[:iteration%3]
                            observed = {}
                            for v in order:
                                r = readers[v]
                                if pattern == 'row_cache_cold':
                                    r.clear()
                                before = r.stats()
                                started = time.perf_counter_ns()
                                observed[v] = r.lookup(current)
                                wall_ns = time.perf_counter_ns()-started
                                after = r.stats()
                                delta = {k: after[k]-before[k] for k in
                                    ('hits', 'misses', 'reads', 'requested_io_bytes', 'lookup_ns', 'lookups')}
                                if delta['lookups'] != 1 or (v=='page15' and delta['requested_io_bytes'] != delta['misses']*4096):
                                    raise AssertionError('Native I/O accounting mismatch')
                                samples[v].append(dict(iteration=iteration, wall_ns=wall_ns, **delta))
                            for v in variants[1:]:
                                for actual, expected in zip(observed[v], observed['original']):
                                    np.testing.assert_array_equal(actual, expected)
                        summary = {v: dict(median_native_us=statistics.median(x['lookup_ns'] for x in rows)/1000,
                            median_wall_us=statistics.median(x['wall_ns'] for x in rows)/1000,
                            total_requested_io_bytes=sum(x['requested_io_bytes'] for x in rows),
                            total_misses=sum(x['misses'] for x in rows)) for v, rows in samples.items()}
                        result = dict(layer=layer, tokens=tokens, rows=tokens*12, pattern=pattern,
                                      repeats=a.repeats, outputs_byte_exact=True, summary=summary, samples=samples)
                        results['cases'].append(result)
                        a.output.write_text(json.dumps(results, indent=2)+'\n')
                        print(json.dumps({k:v for k,v in result.items() if k!='samples'}), flush=True)
            finally:
                for r in readers.values():
                    r.close()
        results['status'] = 'complete'
    except BaseException as error:
        results.update(status='failed', error=repr(error)); raise
    finally:
        results['finished_at'] = time.time()
        a.output.write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
