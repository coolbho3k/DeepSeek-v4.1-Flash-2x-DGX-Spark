# SPDX-License-Identifier: AGPL-3.0-only
"""Two independent kernel candidates in one bounded GPU qualification session."""
import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path
import runpy
import statistics
import time


def compare_attention(torch, actual, expected):
    a, l = actual
    b, n = expected
    assert torch.isfinite(a).all()
    nmse = ((a - b).square().sum() / b.square().sum().clamp_min(1e-30)).item()
    assert nmse <= 1e-7, ('attention NMSE', nmse)
    finite = torch.isfinite(n)
    assert torch.equal(torch.isfinite(l), finite)
    assert torch.equal(l[~finite], n[~finite])
    delta = (l[finite] - n[finite]).abs().max().item() if finite.any() else 0.
    assert delta <= 2e-4, ('attention LSE', delta)
    return dict(nmse=nmse, max_lse_abs=delta)


def paired_time(torch, graph_type, old, new, rows, repeats=15):
    graphs = {False: graph_type(old, tokens=rows), True: graph_type(new, tokens=rows)}
    samples = {False: [], True: []}
    flush = torch.zeros(48 * 2**20, device='cuda', dtype=torch.uint8)
    for iteration in range(repeats):
        for selected in ((False, True) if iteration % 2 else (True, False)):
            flush.add_(1)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            graphs[selected].replay()
            end.record()
            end.synchronize()
            samples[selected].append(begin.elapsed_time(end))
    for graph in graphs.values():
        graph.close()
    return dict(old_ms=statistics.median(samples[False]), new_ms=statistics.median(samples[True]),
                repeats=repeats, l2_flushed=True, samples=samples)


def attention_tests(torch, graph_type, candidate, save):
    from ds41 import fused_sparse_attention as old, fp4_main_kv as codec
    from check_dcp_attention import pack_cache
    original = old.packed_sparse_attention_with_lse
    changed = candidate.wrap(original)
    cases = []
    configurations = [(r, h, w, m, s, scale) for r, h, w, m, s, scale in (
        (1, 32, 128, 'none', True, 1.), (1, 64, 128, 'fp4', True, 1.),
        (2, 64, 193, 'fp4', False, 1.), (3, 64, 128, 'fp4', True, 1.),
        (4, 64, 128, 'fp4', True, 1.), (8, 64, 193, 'fp4', True, 1.),
        (4, 64, 4096, 'fp4', True, 1.), (4, 32, 128, 'fp8', True, 1.),
        (4, 64, 128, 'fp4', False, 0.), (4, 64, 128, 'fp4', True, -1.),
        (4, 64, 128, 'fp4', True, 16.), (24, 64, 193, 'fp4', True, 1.),
        (1, 64, 0, 'none', True, 1.), (4, 64, 0, 'none', False, 1.))]
    for rows, heads, width, mode, has_sink, qscale in configurations:
        q = (torch.randn((rows, heads, 1024), device='cuda') * .5 * qscale).bfloat16()[..., ::2]
        count = max(32, ((width + 31) // 32) * 32)
        packed, _ = pack_cache((torch.randn((count // 32, 32, 512), device='cuda') * .5).bfloat16())
        raw = torch.full((packed.shape[0], 32 * 584 + 512), 165, device='cuda', dtype=torch.uint8)
        raw[:, :32 * 584].copy_(packed.reshape(packed.shape[0], -1))
        swa = raw[:, :32 * 584].view_as(packed)
        ids = torch.arange(max(1, width), device='cuda', dtype=torch.int32).flip(0).expand(rows, -1).contiguous()
        lengths = torch.full((rows,), width, device='cuda', dtype=torch.int32)
        ids[:, ::11] = -1
        sinks = torch.linspace(-2, 2, heads, device='cuda') if has_sink else None
        options = dict(sinks=sinks)
        if mode != 'none':
            values = (torch.randn((1024, 512), device='cuda') * .5).bfloat16()
            if mode == 'fp4':
                main = torch.zeros((8, 128, 288), device='cuda', dtype=torch.uint8)
                codec.store(main, values, torch.arange(1024, device='cuda'))
            else:
                main, _ = pack_cache(values.reshape(8, 128, 512))
            ci = ((torch.arange(512, device='cuda', dtype=torch.int32) * 7) % 1024).expand(rows, -1).contiguous()
            ci[:, ::13] = -1
            cl = torch.full((rows,), 512, device='cuda', dtype=torch.int32)
            options.update(compressed_cache=main, compressed_indices=ci, compressed_lengths=cl)
        snapshots = [(swa, swa.clone()), (raw, raw.clone())]
        if mode != 'none':
            snapshots.append((main, main.clone()))
        invoke_old = lambda: original(q, swa, ids, lengths, **options)
        invoke_new = lambda: changed(q, swa, ids, lengths, **options)
        errors = [compare_attention(torch, invoke_new(), invoke_old())]
        graph = graph_type(invoke_new, tokens=rows)
        streams = (torch.cuda.current_stream(), torch.cuda.Stream())
        for step in range(4):
            torch.cuda.synchronize()
            with torch.cuda.stream(streams[step % 2]):
                q.mul_(-.91)
                lengths.fill_([0, min(31, width), width + 7, -3][step])
                if mode != 'none':
                    cl.fill_([1, 127, 600, 0][step])
                errors.append(compare_attention(torch, graph.replay(), invoke_old()))
        torch.cuda.synchronize()
        graph.close()
        assert all(torch.equal(a, b) for a, b in snapshots)
        lengths.fill_(width)
        if mode != 'none':
            cl.fill_(512)
        timing = paired_time(torch, graph_type, invoke_old, invoke_new, rows)
        case = dict(rows=rows, heads=heads, width=width, main=mode, sink=has_sink,
                    qscale=qscale, errors=errors, timing=timing, changed_input_replays=4,
                    streams=2, cache_canaries_unchanged=True)
        cases.append(case)
        save('attention', cases)
        print(json.dumps(dict(stage='attention', **{k:case[k] for k in ('rows','heads','width','main','timing')})), flush=True)
        del graph, snapshots
        gc.collect()
    return cases


def indexer_tests(torch, graph_type, candidate, save):
    from ds41.dcp_indexer_graph import paged_logits as original
    from ds41.dcp_topk_graph import decode_topk
    from check_combined_miaai_gpu import known
    changed = candidate.wrap(original)
    cases = []
    for batch, next_n, states, cap, actual_length in (
        (1, 1, 64, 4096, 129), (1, 4, 64, 4096, 1021),
        (2, 3, 128, 4096, 2000), (6, 4, 64, 4096, 2000),
        (1, 4, 128, 65536, 32768), (1, 4, 128, 524288, 32768)):
        columns = (actual_length + states - 1) // states
        pages = columns + 1
        values, scales, dense_q = known(torch, (batch, next_n, 32))
        keys, key_scales, dense_k = known(torch, (pages, states))
        stride = ((states * 68 + 511) // 512) * 512 + 512
        raw = torch.full((pages, stride), 165, device='cuda', dtype=torch.uint8)
        raw[:, :states * 64].copy_(keys.view(torch.uint8).reshape(pages, -1))
        raw[:, states * 64:states * 68].view(torch.int32).copy_(key_scales)
        cache = raw[:, :states * 68].view(pages, states, 1, 68)
        table = (torch.arange(batch * columns, device='cuda').reshape(batch, columns) * 7 % pages).int()
        lengths = torch.full((batch, next_n), actual_length, device='cuda', dtype=torch.int32)
        weights = torch.rand((batch, next_n, 32), device='cuda') * .01
        snapshot = raw.clone()
        def invoke(new):
            fn = changed if new else original
            return fn((values, scales), cache, weights, lengths, table, None, max_model_len=cap)
        def compare(a, b):
            finite = torch.isfinite(b)
            assert torch.equal(torch.isfinite(a), finite)
            assert torch.isneginf(a[~finite]).all()
            torch.testing.assert_close(a[finite], b[finite], rtol=5e-5, atol=5e-5)
            nmse = ((a[finite]-b[finite]).square().sum()/b[finite].square().sum().clamp_min(1e-30)).item()
            assert nmse <= 1e-9, nmse
            k = 512
            actual = torch.empty((batch * next_n, k), device='cuda', dtype=torch.int32)
            expected = torch.empty_like(actual)
            decode_topk(a, lengths, actual, None, k, cap)
            decode_topk(b, lengths, expected, None, k, cap)
            assert torch.equal(actual, expected), 'Selected index membership/order changed'
            return dict(nmse=nmse, exact_selected_indices=True)
        errors = [compare(invoke(True), invoke(False))]
        graph = graph_type(lambda: invoke(True), tokens=batch * next_n)
        for step in range(4):
            lengths.fill_([0, min(511, actual_length), actual_length, min(1, actual_length)][step])
            table.copy_((table + 3) % pages)
            weights.mul_(.97)
            errors.append(compare(graph.replay(), invoke(False)))
        graph.close()
        assert torch.equal(raw, snapshot)
        lengths.fill_(actual_length)
        timing = paired_time(torch, graph_type, lambda: invoke(False), lambda: invoke(True), batch * next_n)
        case = dict(batch=batch, next_n=next_n, states=states, capacity=cap,
                    actual_length=actual_length, errors=errors, timing=timing,
                    changed_input_replays=4, cache_canaries_unchanged=True)
        cases.append(case)
        save('indexer', cases)
        print(json.dumps(dict(stage='indexer', batch=batch, next_n=next_n, cap=cap, timing=timing)), flush=True)
        del graph, dense_k, dense_q, snapshot
        gc.collect()
    return cases


def topk_tests(torch, graph_type, candidate, save):
    from ds41.dcp_topk_graph import decode_topk as original
    changed = candidate.wrap(original)
    cases = []
    for rows, width, k in ((4, 64, 512), (4, 4096, 512), (4, 65536, 1024),
                           (4, 524288, 512), (4, 524288, 2048), (24, 65536, 1024),
                           (1, 262144, 512), (4, 262144, 512), (24, 262144, 512)):
        logits = torch.randint(-4, 5, (rows, width), device='cuda').float()
        logits[:, ::11] = -torch.inf
        logits[:, ::23] = torch.inf
        logits[:, ::29] = torch.nan
        logits[:, ::31] = -0.
        lengths = torch.tensor(([0, min(k,width), min(k+1,width), width]*6)[:rows], device='cuda', dtype=torch.int32)
        outputs = [torch.empty((rows,k),device='cuda',dtype=torch.int32) for _ in range(2)]
        def invoke(new):
            (changed if new else original)(logits,lengths,outputs[int(new)],None,k,width)
            return outputs[int(new)]
        def compare():
            assert torch.equal(invoke(True), invoke(False)), 'Exact top-k order differs'
        compare()
        logits.copy_(torch.randn_like(logits))
        compare()
        logits[:, ::23] = torch.inf
        logits[:, ::29] = torch.nan
        logits[:, ::31] = -0.
        graph = graph_type(lambda: invoke(True), tokens=rows)
        for step in range(4):
            lengths.copy_(torch.tensor(([min(width,step+1), width, min(width,k+step), 0]*6)[:rows],device='cuda',dtype=torch.int32))
            logits.mul_(-1)
            actual = graph.replay().clone()
            assert torch.equal(actual, invoke(False)), 'Top-k graph differs'
        graph.close()
        timings = {}
        for label, length in (('short',min(128,width)), ('mid',min(4096,width)), ('full',width)):
            lengths.fill_(length)
            compare()
            timings[label] = paired_time(torch,graph_type,lambda:invoke(False),lambda:invoke(True),rows,repeats=9)
        case = dict(rows=rows,width=width,k=k,exact=True,changed_input_replays=4,timings=timings)
        cases.append(case); save('topk',cases)
        print(json.dumps(dict(stage='topk',rows=rows,width=width,k=k,timings={n:{k:v for k,v in t.items() if k!='samples'} for n,t in timings.items()})),flush=True)
        del logits,graph,outputs
        gc.collect()
    return cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host-index', type=int, required=True)
    a = p.parse_args()
    import torch
    torch.set_num_threads(2)
    runpy.run_path('/opt/ds41-serving/serve.py', run_name='batch_test_entry')
    import spark_combined_miaai as combined
    combined.register()
    torch.cuda.set_per_process_memory_fraction(.02)
    torch.backends.cuda.matmul.allow_tf32 = False
    import ds41
    ds41.__path__.insert(0, '/work/ds41')
    modules = {name: importlib.import_module('ds41.' + name) for name in
               ('online_decode_attention', 'direct_paged_indexer_v2', 'length_aware_topk_native')}
    from check_combined_miaai_gpu import NativeGraph
    result = dict(status='running', host=a.host_index, candidates={},
        source_sha256={n:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for n,m in modules.items()},
        original_runtime_manifest='caf7eeee7668bc11434eb4b4c06589f81a45a08cdef7ea480d8b8303f757af23')
    def save(name=None, cases=None):
        if name:
            result['candidates'].setdefault(name, {})['cases'] = cases
        Path('/results/complete.json').write_text(json.dumps(result, indent=2) + '\n')
    with torch.inference_mode():
        for label, name, test in (('attention','online_decode_attention',attention_tests),
                                  ('indexer','direct_paged_indexer_v2',indexer_tests),
                                  ('topk','length_aware_topk_native',topk_tests)):
            torch.manual_seed(41917)
            try:
                cases = test(torch, NativeGraph, modules[name], save)
                result['candidates'][label] = dict(status='pass', cases=cases)
            except Exception as error:
                result['candidates'].setdefault(label, {}).update(status='failed', error=repr(error))
                import traceback
                traceback.print_exc()
            save()
        # Keep graph-owner error/refusal semantics after replacing kernels.
        from ds41 import fused_sparse_attention as fused
        from ds41.dcp_topk_graph import decode_topk
        from check_dcp_attention import pack_cache
        guards = {}
        swa, _ = pack_cache(torch.zeros((2,32,512),device='cuda',dtype=torch.bfloat16))
        q = torch.zeros((4,64,512),device='cuda',dtype=torch.bfloat16)
        ids = torch.zeros((4,128),device='cuda',dtype=torch.int32)
        lengths = torch.ones(4,device='cuda',dtype=torch.int32)
        attention = modules['online_decode_attention'].wrap(fused.packed_sparse_attention_with_lse)
        graph = NativeGraph(lambda:attention(q,swa,ids,lengths),tokens=4)
        ids[:,0] = 999999
        try:graph.replay()
        except ValueError as error:
            assert 'Sparse slot exceeds' in str(error)
            guards['attention_bad_slot_rejected'] = True
        else:raise AssertionError('Bad sparse slot escaped graph boundary')
        try:graph.replay()
        except RuntimeError as error:
            assert 'poisoned' in str(error)
            guards['attention_graph_poisoned'] = True
        else:raise AssertionError('Failed graph was reusable')
        topk = modules['length_aware_topk_native'].wrap(decode_topk)
        logits = torch.zeros((4,4096),device='cuda')
        out = torch.empty((4,512),device='cuda',dtype=torch.int32)
        graph2 = NativeGraph(lambda:topk(logits,lengths,out,None,512,4096),tokens=4)
        lengths.fill_(4097)
        try:graph2.replay()
        except ValueError as error:
            assert 'length exceeds' in str(error)
            guards['topk_bad_length_rejected'] = True
        else:raise AssertionError('Bad top-k length escaped graph boundary')
        try:graph2.replay()
        except RuntimeError as error:
            assert 'poisoned' in str(error)
            guards['topk_graph_poisoned'] = True
        else:raise AssertionError('Failed graph was reusable')
        result['guards'] = guards
        result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        if result['peak_allocated_bytes'] > 512 * 2**20:
            raise RuntimeError('Component test exceeded 512 MiB allocated memory')
        result['status'] = 'complete'
        save()
    print(json.dumps({k:v for k,v in result.items() if k != 'candidates'}), flush=True)

if __name__ == '__main__':
    main()
