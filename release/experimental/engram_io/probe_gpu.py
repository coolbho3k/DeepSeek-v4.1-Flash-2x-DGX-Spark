# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded copied/UVA/overlap qualification, not an end-to-end speed claim.

MiaAI owns the original row reader, cache, CUDA callbacks, and dequant path.
This local experiment retains that arithmetic and tests fewer copies and
deferred completion. Uses small synthetic files only; real SSD timings are
separate. Run only on the explicitly stopped serving pair.
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace as NS

import numpy as np
import torch

import engram_candidate_stage as core
from native import Reader
from overlap import DeferredRows, RetrievalStream
from packing import pack, table
from test_cpu import fixture
from ds41.graph_validation import GraphOwner


def embedding(info, rank):
    shared = dict(path=Path(info['path']), rows=info['rows'], max_pages=128)
    return NS(dim=256, part_n_hash_cols=12, chunk_tokens=256, n_hash_cols=24,
        head_start=rank*12, head_end=(rank+1)*12, vocab_start_idx=7,
        vocab_end_idx=info['rows']-7, num_embeddings=info['rows'],
        reader=NS(weight=NS(**shared, offset=info['weight_offset'], row_bytes=256),
                  scale=NS(**shared, offset=info['scale_offset'], row_bytes=8)))


def exact(a, b):
    if not torch.equal(a.view(torch.int16), b.view(torch.int16)):
        raise AssertionError('GPU outputs differ bitwise')


def reference(reader, indices, e):
    ids = indices.cpu().numpy()[:, e.head_start:e.head_end]
    ids = np.where((ids>=e.vocab_start_idx)&(ids<e.vocab_end_idx), ids, -1)
    w, s = reader.lookup(ids)
    w = torch.from_numpy(w).view(torch.float8_e4m3fn).float()
    codes = torch.from_numpy(s).to(torch.int32).repeat_interleave(32, dim=1)
    scales = (codes<<23).view(torch.float32)
    scales[codes==0] = 2.**-127
    scales[codes==255] = float('nan')
    return (w*scales).to(torch.bfloat16).reshape(len(indices), 12, 256)


def check_reference(actual, expected):
    actual = actual.cpu()
    # Independent CPU conversion can canonicalize NaN payloads differently.
    # Finite/inf bits and NaN locations must match. Copied/UVA use bit equality.
    if not torch.equal(torch.isnan(actual), torch.isnan(expected)):
        raise AssertionError('NaN locations changed')
    mask = ~torch.isnan(expected)
    exact(actual[mask], expected[mask])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank', type=int, choices=(0, 1), required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True)
    a = p.parse_args()
    if any(a.results.iterdir()):
        raise ValueError('Fresh GPU result directory required')
    os.environ.update(OFFLOAD_MODE='ssd', DSV41_RESIDENT_SCALES='0', DSV41_IO_THREADS='96')
    torch.cuda.set_device(0)
    result = dict(status='running', rank=a.rank, started_at=time.time(), gpu=str(torch.cuda.get_device_name()),
                  correctness=[], component_timings=[], overlap=[], end_to_end_speed_claim=False)
    report = a.results/'gpu.json'
    report.write_text(json.dumps(result, indent=2)+'\n')
    rng = np.random.default_rng(9141)
    live = []
    try:
        source = a.results/'fixture.safetensors'
        fixture(source, rows=4099)
        info = table(source, 1)
        packed = a.results/'page15.bin'
        pack(source, packed, 1, 7, info['rows']-7, 'page15', chunk_rows=120)
        e = embedding(info, a.rank)
        cpu = Reader(a.library, info, e.vocab_start_idx, e.vocab_end_idx, budget=0, threads=96)
        for mapped in (False, True):
            stage = core.NativeStage(e, a.library, packed=packed, packed_layer=1, mapped=mapped)
            live.append(stage)
            if mapped and (stage.dev_w.data_ptr()!=stage.host_w.data_ptr() or stage.dev_s.data_ptr()!=stage.host_s.data_ptr()):
                raise AssertionError('Expected Spark UVA alias, not a hidden copy')
        copied, mapped = live
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        def indices(tokens):
            ids = rng.integers(-12, info['rows']+12, (tokens, 24), dtype=np.int64)
            if tokens:
                ids[0] = np.resize(np.array([-1, 0, 6, 7, 8, info['rows']-8, info['rows']-7, info['rows'], 2**31]),24)
            return torch.tensor(ids, device='cuda')
        for tokens in (0, 1, 4, 6, 24, 255, 256, 257, 1056, 2048):
            ids = indices(tokens)
            outputs = [torch.empty((tokens,12,256),dtype=torch.bfloat16,device='cuda') for _ in live]
            for stage, out in zip(live, outputs):
                stage.lookup(ids,out)
            torch.cuda.synchronize()
            exact(*outputs)
            check_reference(outputs[0], reference(cpu, ids, e))
            result['correctness'].append(dict(test='eager', tokens=tokens, copied_uva_bits_equal=True,
                cpu_reference_equal=True, dead_image_unowned_out_of_range_masked=True))
        for tokens in (1, 6, 24, 257, 2048):
            examples = indices(tokens)
            graphs = [stage.capture(examples) for stage in live]
            for iteration in range(6):
                ids = indices(tokens)
                retained = []
                for stream in streams:
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        retained.append([graph.replay(ids) for graph in graphs])
                torch.cuda.synchronize()
                for outs in retained:
                    exact(*outs)
                    check_reference(outs[0], reference(cpu, ids, e))
            for stage in live:
                try:
                    stage.close()
                except RuntimeError as error:
                    if 'Close managed graphs' not in str(error):
                        raise
                else:
                    raise AssertionError('Stage freed while a graph still owns callbacks')
            for graph in graphs:
                graph.close()
            result['correctness'].append(dict(test='graph_changed_inputs_cross_stream', tokens=tokens,
                iterations=6, buffer_lifetime_refusal=True, copied_uva_bits_equal=True))
        # Lookup latency, including actual native callbacks/copies/dequant,
        # with alternating order and fixed graph shapes. This is a warmed
        # synthetic fixture, not necessarily all cache hits: record misses.
        # Real SSD comparison is bench_ssd.py.
        for tokens in (1, 4, 24, 256, 2048):
            ids = indices(tokens)
            graphs = [stage.capture(ids) for stage in live]
            for graph in graphs:
                graph.replay(ids)
            torch.cuda.synchronize()
            values = [[], []]
            def stats(stage):
                counters = (C.c_uint64*9)()
                stage.lib.row_store_stats(stage.store, counters)
                return dict(hits=counters[0], misses=counters[1], reads=counters[2])
            before = [stats(stage) for stage in live]
            for iteration in range(30):
                for i in ((0,1) if iteration%2==0 else (1,0)):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record(); graphs[i].replay(ids); end.record(); end.synchronize()
                    values[i].append(begin.elapsed_time(end)*1000)
            for graph in graphs:
                graph.close()
            after = [stats(stage) for stage in live]
            result['component_timings'].append(dict(tokens=tokens, fixture_row_cache_warmed=True,
                cache_counter_delta=[{k:y[k]-x[k] for k in x} for x,y in zip(before,after)],
                copied_median_us=statistics.median(values[0]), mapped_median_us=statistics.median(values[1]),
                samples_us=dict(copied=values[0], mapped=values[1])))
        # Two tables sharing one ordered retrieval stream. A compute interval
        # on the main stream demonstrates actual overlap, not merely enqueuing
        # on another stream. These are synthetic intervals, NOT serving gains.
        for use_mapped in (False, True):
            stage1 = live[int(use_mapped)]
            stage2 = core.NativeStage(e, a.library, packed=packed, packed_layer=1, mapped=use_mapped)
            live.append(stage2)
            retrieval = RetrievalStream(stage1.device)
            deferred = [DeferredRows(s,retrieval) for s in (stage1,stage2)]
            for tokens in (4, 24, 256):
                ids = indices(tokens)
                outs = [torch.empty((tokens,12,256),dtype=torch.bfloat16,device='cuda') for _ in deferred]
                # Warm everything before capture, including the synthetic GPU work.
                for d,out in zip(deferred,outs):
                    d.prepare(ids,out)
                torch.cuda._sleep(1000000)
                for d in deferred:
                    d.consume()
                torch.cuda.synchronize()
                owner = GraphOwner(stage1.device)
                capture_stream = torch.cuda.Stream()
                capture_stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with owner.execution(capture_only=True):
                    with torch.cuda.graph(graph,stream=capture_stream):
                        for d,out in zip(deferred,outs):
                            d.prepare(ids,out)
                        torch.cuda._sleep(1000000)
                        for d in deferred:
                            d.consume()
                        snapshots = [out.clone() for out in outs]
                for iteration in range(6):
                    new_ids = indices(tokens)
                    ids.copy_(new_ids)
                    with owner.execution(capture_only=False):
                        graph.replay()
                    torch.cuda.synchronize()
                    for output in snapshots:
                        check_reference(output, reference(cpu,new_ids,e))
                owner.wait_before_graph_destruction(); graph.reset(); owner.release_after_graph_destruction()
                # Measure on a cold native row cache, leaving physical caches alone.
                for s in (stage1,stage2):
                    s.lib.ds41_row_store_clear_cache(s.store)
                origin, io_start, io_end, compute_start, compute_end, finish = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
                origin.record()
                retrieval.stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(retrieval.stream):
                    io_start.record()
                for d,out in zip(deferred,outs):
                    d.prepare(ids,out)
                with torch.cuda.stream(retrieval.stream):
                    io_end.record()
                compute_start.record(); torch.cuda._sleep(1000000); compute_end.record()
                for d in deferred:
                    d.consume()
                finish.record(); finish.synchronize()
                positions = {name:origin.elapsed_time(event)*1000 for name,event in
                    (('io_start',io_start),('io_end',io_end),('compute_start',compute_start),('compute_end',compute_end),('finish',finish))}
                overlap = max(0.,min(positions['io_end'],positions['compute_end'])-max(positions['io_start'],positions['compute_start']))
                result['overlap'].append(dict(tokens=tokens,mapped=use_mapped,graph_replay_correct=True,
                    offsets_us=positions,actual_overlap_us=overlap,synthetic_compute=True))
            stage2.close(); live.pop()
        result.update(status='complete', copied_staging_bytes=copied.staging_bytes,
                      mapped_staging_bytes=mapped.staging_bytes, max_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
        cpu.close()
        for stage in live:
            stage.close()
        live.clear()
    except BaseException as error:
        result.update(status='failed', error=repr(error)); raise
    finally:
        result['finished_at'] = time.time()
        report.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(status=result['status'], rank=a.rank, report=str(report),
            correctness_cases=len(result['correctness']), error=result.get('error'))),flush=True)


if __name__ == '__main__':
    main()
