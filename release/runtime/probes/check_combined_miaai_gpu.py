# SPDX-License-Identifier: AGPL-3.0-only
"""One assembled-stack GPU pass; no full-model or throughput claims.

Runs against a fresh, immutable test kit. Canonical weights/codecs are not
modified. Native full-graph replay boundaries are exercised with changing
metadata; the actual distributed/model/DSpark pass remains separately required.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile

RESULTS = Path('/results')
LIMIT = 512 * 2**20


def save(name, value):
    with (RESULTS / name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    # Observe the cumulative high-water mark without resetting or weakening
    # the suite's512MiB guard. Locate fixture retention/experimental scratch.
    torch=sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        with (RESULTS/'memory-progress.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(after=name,allocated=torch.cuda.memory_allocated(),
                reserved=torch.cuda.memory_reserved(),peak=torch.cuda.max_memory_allocated()))+'\n')


class NativeGraph:
    def __init__(self, function, tokens=4, manager_type=None):
        import torch
        from vllm.config import CUDAGraphMode
        from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager, BatchExecutionDescriptor
        function()  # Native kernels, workspaces and callback descriptors prewarm.
        torch.cuda.synchronize()
        self.manager = object.__new__(manager_type or CudaGraphManager)
        self.manager.device = torch.device('cuda', 0)
        self.manager.use_breakable_cg = False
        self.manager.graphs = {}
        self.key = BatchExecutionDescriptor(CUDAGraphMode.FULL, tokens, 1)
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        context = CudaGraphManager.capture.__globals__['_ds41_capture_context']
        with torch.cuda.stream(stream):
            with context(self.manager, self.key, self.graph), torch.cuda.graph(self.graph, stream=stream):
                self.output = function()
        self.manager.graphs[self.key] = self.graph
        torch.cuda.current_stream().wait_stream(stream)

    def replay(self):
        self.manager.run_fullgraph(self.key)
        return self.output

    def close(self):
        resources = self.manager._ds41_graph_resources
        resources.clear()
        self.manager.graphs.clear()
        assert not resources.owners and not resources.graphs


def topk_cases(torch):
    import ast
    import spark_topk
    from ds41.dcp_topk_graph import decode_topk
    assert spark_topk.decode_topk is decode_topk
    # Reconstruct the unchanged on-disk legacy reference, not the installed
    # replacement; no torch.ops or vendor files are modified by this test.
    tree = ast.parse(Path(spark_topk.__file__).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'decode_topk')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<legacy-topk-reference>', 'exec'), namespace)
    reference = namespace['decode_topk']
    cases = []
    for rows, width, k in ((4, 64, 512), (4, 4096, 512), (4, 8192, 1024),
                           (24, 4096, 2048), (4, 1048576, 1024)):
        logits = torch.randint(-4, 5, (rows, width), device='cuda').float()
        logits[:, ::11] = -torch.inf  # Include ties and live non-finite columns.
        logits[:, ::23] = torch.inf
        logits[:, ::29] = torch.nan
        logits[:, ::31] = -0.
        logits[:, ::37] = 0.
        logits[:, ::41] = torch.finfo(torch.float32).tiny
        lengths = torch.tensor(([0, min(k,width), min(k+1,width), width] * 6)[:rows],
                               device='cuda', dtype=torch.int32)
        output = torch.empty((rows,k), device='cuda', dtype=torch.int32)
        expected = torch.empty_like(output)
        def invoke():
            decode_topk(logits,lengths,output,None,k,width)
            return output
        def compare():
            reference(logits,lengths,expected,None,k,width)
            assert torch.equal(output, expected)
        invoke(); compare()
        # All-live rows provide a like-for-like full-width selection benchmark.
        import time
        lengths.fill_(width)
        timings={}
        for label,fn in (('stable_full_sort',lambda:reference(logits,lengths,expected,None,k,width)),
                         ('partial_integer_keys',invoke)):
            fn();torch.cuda.synchronize()
            started=time.perf_counter()
            for _ in range(5):fn()
            torch.cuda.synchronize()
            timings[label]=(time.perf_counter()-started)*1000/5
        compare()
        graph = NativeGraph(invoke, tokens=rows)
        streams = (torch.cuda.current_stream(), torch.cuda.Stream())
        saved = []
        for step in range(4):
            torch.cuda.synchronize()
            with torch.cuda.stream(streams[step % 2]):
                lengths.copy_(torch.tensor(([min(width,k+step+1), 0, width, min(width,step+1)] * 6)[:rows],
                                          device='cuda', dtype=torch.int32))
                logits.mul_(-1)
                graph.replay(); compare()
                saved.append((output.clone(), expected.clone()))
        torch.cuda.synchronize()
        assert all(torch.equal(a,b) for a,b in saved)
        graph.close()
        cases.append(dict(rows=rows,width=width,k=k,eager_exact=True,replays_exact=4,
                          streams=2,short_rows_and_ties=True,
                          ieee_nan_inf_signed_zero=True,component_timing_ms=timings))
        del graph,logits,lengths,output,expected,saved
        gc.collect();torch.cuda.empty_cache()
    # Replayed invalid lengths must be rejected by the owner before return,
    # not copied to the host inside capture or silently clamped as valid data.
    logits=torch.zeros((1,4096),device='cuda');lengths=torch.ones(1,device='cuda',dtype=torch.int32)
    output=torch.empty((1,1024),device='cuda',dtype=torch.int32)
    graph=NativeGraph(lambda: decode_topk(logits,lengths,output,None,1024,4096),tokens=1)
    lengths.fill_(4097)
    try:graph.replay()
    except ValueError as error:assert 'length exceeds' in str(error)
    else:raise AssertionError('Invalid graph length escaped the owner boundary')
    resources = graph.manager._ds41_graph_resources
    assert resources.owners[graph.key].failed
    try: graph.replay()
    except RuntimeError as error: assert 'poisoned' in str(error)
    else: raise AssertionError('Failed graph owner allowed another replay')
    try: resources.clear()
    except RuntimeError as error: assert 'Retaining poisoned' in str(error)
    else: raise AssertionError('Failed graph resources were destroyed')
    assert resources.poisoned and resources.graphs and resources.owners
    return dict(cases=cases,invalid_replay_rejected=True,legacy_source_unchanged=True)


def candidate_chain_cases(torch):
    from ds41 import vllm_dcp as installed, dcp_candidates as legacy
    from ds41 import dcp_candidates_graph as changed
    from ds41.dcp_topk_graph import decode_topk
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import select_candidate_blocks as native_select
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import apply_candidate_mask as native_mask
    assert installed.select_candidate_blocks is changed.select_candidate_blocks
    reports=[]
    for rows,width,bs,k,repeat in ((4,64,8,7,1),(4,4096,128,5,1),
                                   (24,2048,64,7,4),(4,131072,128,32,1),
                                   (4,524288,8,2048,1)):
        nr=(rows+repeat-1)//repeat
        full=torch.randn((rows,width*2),device='cuda')
        global_lengths=torch.empty(nr,device='cuda',dtype=torch.int32)
        lengths=[torch.empty_like(global_lengths) for _ in range(2)]
        locals_=[torch.empty((rows,width),device='cuda') for _ in range(2)]
        nb=(width*2+bs-1)//bs
        peer_scores=[torch.empty((rows,nb),device='cuda') for _ in range(2)]
        reference=torch.empty((rows,k),device='cuda',dtype=torch.int32)
        reference_mask=torch.empty_like(full)
        def update(step):
            full.normal_()
            # Duplicate scores and -inf holes exercise native tie/padding semantics.
            full[:,::17]=-torch.inf
            full[:,::19]=.25
            counts=([0,1,width*2-1,width*2,13,bs+1]*6)[:nr]
            counts=[min(width*2,max(0,n-step)) for n in counts]
            global_lengths.copy_(torch.tensor(counts,device='cuda',dtype=torch.int32))
            native_select(full,None,global_lengths,k,bs,reference,repeat)
            reference_mask.copy_(full)
            native_mask(reference_mask,None,global_lengths,reference,bs,repeat)
            for rank in (0,1):
                lengths[rank].copy_((global_lengths+1-rank)//2)
                locals_[rank].copy_(full[:,rank::2])
                peer_scores[rank].copy_(legacy.local_block_scores(locals_[rank],None,lengths[rank],
                    bs,2,rank,nb,repeat))
        update(0)
        for rank in (0,1):
            class Group:
                world_size=2
                rank_in_group=rank
                def all_gather(self,tensor,dim):
                    peer=(lengths[1-rank][torch.arange(rows,device='cuda')//repeat,None]
                          if tensor.dtype==torch.int32 else peer_scores[1-rank])
                    return torch.cat((tensor,peer) if rank==0 else (peer,tensor),dim=dim)
            group=Group();out=torch.empty_like(reference);masked=torch.empty_like(locals_[rank])
            selected=torch.empty((rows,512),device='cuda',dtype=torch.int32)
            expanded=torch.empty(rows,device='cuda',dtype=torch.int32)
            def invoke():
                masked.copy_(locals_[rank])
                installed.select_candidate_blocks(masked,None,lengths[rank],k,bs,out,group,repeat)
                installed.apply_candidate_mask(masked,None,lengths[rank],out,bs,2,rank,repeat)
                expanded.copy_(lengths[rank][torch.arange(rows,device='cuda')//repeat])
                decode_topk(masked,expanded,selected,None,512,width)
                return out,masked,selected
            def compare():
                assert torch.equal(out.sort(-1).values,reference.sort(-1).values)
                expected_mask=reference_mask[:,rank::2]
                assert torch.equal(masked,expected_mask)
                expected=torch.full_like(selected,-1)
                for row in range(rows):
                    count=int(lengths[rank][row//repeat])
                    ids=(torch.arange(count,device='cuda') if count<=512 else
                         torch.argsort(expected_mask[row,:count],descending=True,stable=True)[:512])
                    expected[row,:len(ids)]=ids.int()
                assert torch.equal(selected,expected)
            invoke();compare()
            graph=NativeGraph(invoke,tokens=rows)
            streams=(torch.cuda.current_stream(),torch.cuda.Stream())
            for step in range(4):
                torch.cuda.synchronize()
                with torch.cuda.stream(streams[step%2]):
                    update(step);graph.replay();compare()
            torch.cuda.synchronize();graph.close()
            reports.append(dict(rows=rows,width=width,block_size=bs,candidates=k,row_repeat=repeat,
                rank=rank,native_membership_and_mask_exact=True,connected_topk_exact=True,replays=4))
            del graph,out,masked,selected,expanded
        del full,global_lengths,lengths,locals_,peer_scores,reference,reference_mask
        gc.collect();torch.cuda.empty_cache()
    return dict(cases=reports,actual_collectives=False,native_reference=True)


def known(torch, shape):
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], device='cuda')
    codes = torch.randint(0, 16, (*shape, 128), device='cuda', dtype=torch.int32)
    exponent = torch.randint(-4, 2, (*shape, 4), device='cuda', dtype=torch.int32)
    dense = lut[codes.long()] * torch.exp2(exponent.float()).repeat_interleave(32, dim=-1)
    packed = (codes[..., ::2] | (codes[..., 1::2] << 4)).to(torch.uint8).contiguous().view(torch.int8)
    scales = (exponent + 127).to(torch.uint8).contiguous().view(torch.int32).squeeze(-1)
    return packed, scales, dense


def indexer_cases(torch):
    from ds41.dcp_indexer_graph import paged_logits
    cases = []
    for batch, next_n, states in ((1, 1, 64), (1, 4, 64), (2, 3, 128), (6, 4, 64)):
        pages, columns, cap = 32, 16, 4096
        values, scales, dense_q = known(torch, (batch, next_n, 32))
        keys, key_scales, dense_k = known(torch, (pages, states))
        stride = ((states * 68 + 511) // 512) * 512
        raw = torch.zeros((pages, stride), device='cuda', dtype=torch.uint8)
        raw[:, :states * 64].copy_(keys.view(torch.uint8).reshape(pages, -1))
        raw[:, states * 64:states * 68].view(torch.int32).copy_(key_scales)
        cache = raw[:, :states * 68].view(pages, states, 1, 68)
        table = (torch.arange(batch * columns, device='cuda').reshape(batch, columns) * 7 % pages).int()
        lengths = torch.tensor([[1 + 129 * n + r * 11 for n in range(next_n)] for r in range(batch)],
                               device='cuda', dtype=torch.int32)
        weights = torch.rand((batch, next_n, 32), device='cuda') * .01

        def invoke():
            return paged_logits((values, scales), cache, weights, lengths, table, None, max_model_len=cap)

        def compare(actual):
            expected = torch.full_like(actual, -torch.inf)
            for r in range(batch):
                restored = dense_k[table[r].long()].reshape(-1, 128)
                dense = (torch.matmul(dense_q[r], restored.T).relu() * weights[r, :, :, None]).sum(1)
                valid = torch.arange(restored.shape[0], device='cuda')[None, :] < lengths[r, :, None]
                expected[r * next_n:(r + 1) * next_n, :restored.shape[0]] = dense.masked_fill(~valid, -torch.inf)
            mask = torch.isfinite(expected)
            torch.testing.assert_close(actual[mask], expected[mask], rtol=5e-5, atol=5e-5)
            assert torch.isneginf(actual[~mask]).all().item()
            return float(((actual[mask] - expected[mask]).square().sum()
                / expected[mask].square().sum().clamp_min(1e-30)).item())

        nmse = compare(invoke())
        graph = NativeGraph(invoke, tokens=batch * next_n)
        errors = []
        for step in range(4):
            lengths.add_(17)
            table.copy_((table + 3) % pages)
            weights.mul_(.97)
            errors.append(compare(graph.replay()))
        graph.close()
        assert max([nmse, *errors]) < 1e-9
        cases.append(dict(batch=batch, next_n=next_n, states=states, capacity=cap,
            page_stride=stride, eager_nmse=nmse, replay_nmse=errors))
        del graph, raw, cache, keys, key_scales, dense_k, values, scales, dense_q
        gc.collect()
    return cases


def slot_cases(torch):
    from vllm.v1.worker.gpu.block_table import BlockTables
    cases = []
    for rank in (0, 1):
        tables = BlockTables([128, 32, 8], 2, 2048, [16, 16, 16], torch.device('cuda', 0),
            [128, 32, 8], cp_size=2, cp_rank=rank, cp_interleave=1,
            slot_mapping_enabled=[True, True, False])
        tables._ds41_group_owners = (2, 1, 1)
        pages = {}
        for request in (0, 1):
            groups = tuple([10 + request * 32 + g * 64 + i for i in range(16)] for g in range(3))
            pages[request] = groups
            tables.append_block_ids(request, groups, overwrite=True)
        tables.apply_staged_writes()
        request_order = [1, 0]
        positions = [0, 1, 31, 32, 127, 128, 255, 256]
        actual = tables.compute_slot_mappings(torch.tensor(request_order, device='cuda', dtype=torch.int32),
            torch.tensor([0, 4, 8], device='cuda', dtype=torch.int32),
            torch.tensor(positions, device='cuda', dtype=torch.int64), 12)
        expected = [[-1] * 12 for _ in range(3)]
        for i, position in enumerate(positions):
            request = request_order[i // 4]
            if position % 2 == rank:
                local = position // 2
                expected[0][i] = pages[request][0][local // 128] * 128 + local % 128
            expected[1][i] = pages[request][1][position // 32] * 32 + position % 32
        assert actual.cpu().tolist() == expected
        assert (tables.slot_mappings[:, 8:] == -1).all().item()
        cases.append(dict(rank=rank, native_kernel=True, global_sharded=True,
            swa_replicated=True, rings_suppressed=True, padding_exact=True))
    return cases


def attention_cases(torch):
    from ds41 import fp4_main_kv as codec
    from ds41.fused_sparse_attention import packed_sparse_attention_with_lse as attention
    query = torch.randn((4, 64, 512), device='cuda', dtype=torch.bfloat16) * .1
    swa = torch.zeros((4, 128, 584), device='cuda', dtype=torch.uint8)
    main = torch.empty((4, 128, 288), device='cuda', dtype=torch.uint8)
    latent = torch.randn((512, 512), device='cuda', dtype=torch.bfloat16) * .1
    codec.store(main, latent, torch.arange(512, device='cuda'))
    swa_ids = torch.arange(128, device='cuda', dtype=torch.int32)[None, :].repeat(4, 1)
    main_ids = torch.arange(64, device='cuda', dtype=torch.int32)[None, :].repeat(4, 1)
    swa_lengths = torch.full((4,), 128, device='cuda', dtype=torch.int32)
    main_lengths = torch.full((4,), 64, device='cuda', dtype=torch.int32)
    sinks = torch.zeros(64, device='cuda')
    def invoke():
        return attention(query, swa, swa_ids, swa_lengths, compressed_cache=main,
            compressed_indices=main_ids, compressed_lengths=main_lengths, sinks=sinks)
    graph = NativeGraph(invoke)
    for step in range(4):
        query.mul_(.91)
        swa_lengths.sub_(7)
        main_ids.add_(1)
        expected = invoke()
        actual = graph.replay()
        for a, b in zip(actual, expected): torch.testing.assert_close(a, b, rtol=0, atol=0)
    graph.close()
    # Expected metadata failure: addresses are masked by the existing kernel;
    # the native model replay owner must reject before returning an output.
    bad = NativeGraph(invoke)
    main_ids[0, 0] = 1000000
    try:
        bad.replay()
    except ValueError as error:
        assert 'Sparse slot exceeds allocated packed cache' in str(error)
    else:
        raise AssertionError('Graph replay returned invalid-address results')
    resources = bad.manager._ds41_graph_resources
    resources.clear(finalizing=True)
    assert resources.poisoned and resources.graphs and resources.owners
    return dict(replays=4, exact_eager_parity=True, masked_invalid_metadata_rejected=True,
                failed_graph_resources_retained=True)


def engram_cases(torch):
    from safetensors.torch import save_file
    from ds41.ssd_rows import EngramRows
    from ds41.ssd_embedding import SSDHeadEmbedding
    from miaai_engram import NativeStage, _LIVE_STAGES
    library = Path('/opt/ds41-serving/miaai-row-store-v1.so')
    cases = []
    with tempfile.TemporaryDirectory(prefix='combined-engram-', dir='/results') as directory:
        root = Path(directory)
        sizes = [17] * 288
        rows = sum(sizes)
        weights = (torch.randn(rows, 256) * .2).to(torch.float8_e4m3fn)
        scales = torch.full((rows, 8), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
        prefix = 'layers.1.engram.embed.'
        save_file({prefix + 'weight': weights, prefix + 'scale': scales}, root / 'table.safetensors')
        (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {
            prefix + 'weight': 'table.safetensors', prefix + 'scale': 'table.safetensors'}}))
        dense = weights.bfloat16()
        for rank in (0, 1):
            reader = EngramRows(root, 1, cache_bytes=65536)
            # The installed class hook creates the exact stage used by vLLM.
            embedding = SSDHeadEmbedding(rows, 256, sizes, 2, rank, reader, chunk_tokens=256)
            stage = embedding._ds41_native_stage
            assert stage.mode == ('ssd', '0', '96')
            def inputs(count):
                value = torch.randint(-1, rows + 2, (count, 288), device='cuda')
                value[:, ::7] = -1
                return value
            def expected(value):
                ids = value.cpu()[:, embedding.head_start:embedding.head_end]
                valid = (ids >= embedding.vocab_start_idx) & (ids < embedding.vocab_end_idx)
                result = torch.zeros((*ids.shape, 256), dtype=torch.bfloat16)
                result[valid] = dense[ids[valid]]
                return result
            for count in (1, 4, 2048):
                value = inputs(count)
                output = torch.empty((count, 144, 256), device='cuda', dtype=torch.bfloat16)
                embedding.lookup(value, output)
                torch.testing.assert_close(output.cpu(), expected(value), rtol=0, atol=0)
                cases.append(dict(rank=rank, rows=count, exact=True))
            del value, output
            value = inputs(4)
            output = torch.empty((4, 144, 256), device='cuda', dtype=torch.bfloat16)
            def invoke():
                embedding.lookup(value, output)
                return output
            graph = NativeGraph(invoke)
            assert stage.graphs == 1
            streams = [torch.cuda.current_stream(), torch.cuda.Stream()]
            for step in range(4):
                with torch.cuda.stream(streams[step % 2]):
                    value.copy_(inputs(4))
                    torch.testing.assert_close(graph.replay().cpu(), expected(value), rtol=0, atol=0)
            graph.close()
            assert stage.graphs == 0
            embedding.close()
            assert stage.closed and stage not in _LIVE_STAGES
            cases.append(dict(rank=rank, rows=4, owned_graph_replays=4,
                alternating_replay_streams=2, stage_released_after_graph=True))
    return cases


def moe_cases(torch, host_index):
    from check_exl3_prefill_bench import ROOT, CAPTURE, CAPTURE_SHA, digest, sort_once_moe
    from safetensors.torch import load_file
    from safetensors import safe_open
    from ds41.exl3_moe import PackedExpert
    import spark_fused_moe as base
    import spark_grouped_prefill as grouped
    model = Path('/work/artifacts/ds41-exl3-3bpw-candidate-v1')
    index = json.loads((model / 'model.safetensors.index.json').read_bytes())['weight_map']
    experts = {}
    for local, expert in enumerate((0, 48, 96, 144)):
        prefix = f'layers.0.ffn.experts.{expert}'
        keys = [f'{prefix}.{projection}.{field}' for projection in ('w1', 'w3', 'w2')
                for field in ('trellis', 'suh', 'svh', 'mul1')]
        tensors = {}
        for name in sorted({index[key] for key in keys}):
            path = model / name
            assert path.resolve().parent == model and path.suffix == '.safetensors'
            with safe_open(path, framework='pt', device='cpu') as stream:
                for key in keys:
                    if index[key] == name:
                        tensors[key] = stream.get_tensor(key).contiguous().cuda()
        experts[local] = PackedExpert(tensors, prefix, 1 - host_index, 2, limit=10.)
        del tensors
    aliased = {i: experts[i % 4] for i in range(384)}
    dispatch = base._dispatcher
    assert type(dispatch) is grouped.GroupedDispatcher
    torch.manual_seed(4103)
    x = torch.randn((2048, 5120), device='cuda', dtype=torch.bfloat16) * .02
    ids = torch.randint(0, 384, (2048, 6), device='cuda', dtype=torch.int64)
    weights = torch.rand((2048, 6), device='cuda', dtype=torch.float32) / 6
    def nmse(actual, expected):
        assert actual.shape == expected.shape
        # Compare every element, retaining only a256-row FP32 slab instead
        # of several full2048-row conversions beside the real workspaces.
        # Preserve the same2e-5 limit; never reset the allocator peak.
        numerator = denominator = 0.
        for begin in range(0,len(actual),256):
            a=actual[begin:begin+256].float()
            b=expected[begin:begin+256].float()
            assert torch.isfinite(a).all().item()
            numerator += (a-b).square().sum(dtype=torch.float64).item()
            denominator += b.square().sum(dtype=torch.float64).item()
        return numerator/max(denominator,1e-30)
    rows = []
    assert digest(ROOT / CAPTURE) == CAPTURE_SHA
    captured = load_file(ROOT / CAPTURE)
    select = torch.arange(2048) % len(captured['inputs'])
    calibration = tuple(captured[name][select].cuda() for name in ('inputs', 'route_ids', 'route_weights'))
    cases = [('tiny_diffuse', x, ids, weights),
             ('tiny_concentrated', x, torch.zeros_like(ids), weights),
             ('captured_calibration', *calibration)]
    # Only one fixture belongs on the GPU at a time. Outputs used solely
    # for comparison live on CPU; the real2048-row dispatch stays intact.
    cases = [(label,a.cpu(),b.cpu(),c.cpu()) for label,a,b,c in cases]
    del x,ids,weights,calibration
    for case_index, (label, x, ids, weights) in enumerate(cases):
        x,ids,weights=x.cuda(),ids.cuda(),weights.cuda()
        # Splitting changes per-expert counts and can cross FAT_MIN. Neither
        # optimized path is an independent oracle for the other. Compare
        # the changed full-batch path against original PackedExpert at its
        # existing2e-5 limit. Retain smaller-batch and cross-chunk comparisons
        # as diagnostics, not alternative ground truth. GPU attempt4 showed
        # the unchanged tiny-input thin path accounts for the discrepancy.
        expected = torch.cat([dispatch(aliased, x[a:b], ids[a:b], weights[a:b])
            for a, b in ((0, 1024), (1024, 2048))]).cpu()
        actual = dispatch(aliased, x, ids, weights).cpu()
        # These384 IDs alias precisely four immutable weights. Modulo mapping
        # keeps every individual assignment and route weight, including
        # duplicates; it does not combine route weights before FP16 rounding.
        oracle_ids = torch.where((ids >= 0) & (ids < 384), ids.remainder(4), -1)
        reference = sort_once_moe(torch, experts, x, oracle_ids, weights).cpu()
        row = dict(rows=2048, label=label, cross_chunk_nmse=nmse(actual, expected),
            full_vs_canonical_nmse=nmse(actual, reference), half_vs_canonical_nmse=nmse(expected, reference))
        rows.append(row)
        save('moe-row-' + str(case_index) + '.json', row)
        print(json.dumps(dict(stage='combined_moe_parity', **row)), flush=True)
        del reference
    assert dispatch.fat_workspace.bytes == 280173312
    x, ids, weights = x[:4].clone(), ids[:4].clone(), weights[:4].clone()
    del actual, expected, cases, captured
    def invoke(): return dispatch(aliased, x, ids, weights)
    graph = NativeGraph(invoke)
    replays = []
    streams = [torch.cuda.current_stream(), torch.cuda.Stream()]
    for step in range(4):
        with torch.cuda.stream(streams[step % 2]):
            ids.copy_(torch.randint(0, 384, ids.shape, device='cuda'))
            x.mul_(.98)
            expected = invoke()
            replays.append(nmse(graph.replay(), expected))
    graph.close()
    return dict(cases=rows, replay_nmse=replays, grouped_scratch_bytes=280173312,
        alternating_replay_streams=2,
        canonical_parity_pass=all(r['full_vs_canonical_nmse'] <= 2e-5 for r in rows),
        unchanged_half_path_within_same_threshold=all(r['half_vs_canonical_nmse'] <= 2e-5 for r in rows),
        calibration_capture_sha256=CAPTURE_SHA,
        replay_parity_pass=all(n <= 2e-5 for n in replays),
        actual_canonical_experts=4, aliased_route_ids=384, full_layer_bandwidth_measured=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host-index', type=int, choices=(0, 1), required=True)
    parser.add_argument('--registration-child', action='store_true')
    parser.add_argument('--rank-major-query', action='store_true',
        help='Test the unselected rank-major prefill candidate; does not install it in serving')
    args = parser.parse_args()
    cache_root=Path('/results/jit-cache')
    for name in ('TMPDIR','XDG_CACHE_HOME','TILELANG_CACHE_DIR','DG_JIT_CACHE_DIR'):
        assert Path(os.environ[name]).is_relative_to(cache_root)
    assert Path(tempfile.gettempdir())==cache_root/'tmp'
    (Path(os.environ['XDG_CACHE_HOME'])/'torch/kernels').mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.0075)
    torch.backends.cuda.matmul.allow_tf32 = False
    runpy.run_path('/opt/ds41-serving/serve.py', run_name='combined_test_entry')
    import spark_combined_miaai as combined
    assert torch.cuda.memory_allocated() == torch.cuda.memory_reserved() == 0
    descriptor = combined.register()
    dspark_enabled = os.environ.get('DS41_ENABLE_DSPARK', '0') == '1'
    assert descriptor['dspark_enabled'] == dspark_enabled
    vocabulary_enabled = os.environ.get('DS41_ENABLE_SSD_VOCAB', '0') == '1'
    assert descriptor['ssd_input_vocabulary'] == vocabulary_enabled
    if vocabulary_enabled:
        from vllm.models.deepseek_v4_1.nvidia import model as native_model
        import streaming_loader
        assert native_model.VocabParallelEmbedding._ds41_lossless_input_vocabulary
        assert hasattr(streaming_loader.ordered_nonengram_weights, '_ds41_vocabulary_skip')
    assert len(combined._installed) >= 30
    if args.registration_child:
        print(json.dumps(dict(status='combined_child_registration_pass', descriptor=descriptor)), flush=True)
        return
    child = subprocess.run([sys.executable, '-B', __file__, '--host-index', str(args.host_index),
        '--registration-child'], text=True, capture_output=True, check=True, timeout=180)
    assert 'combined_child_registration_pass' in child.stdout
    limits = {name: (Path('/sys/fs/cgroup') / name).read_text().strip()
        for name in ('memory.max', 'memory.swap.max', 'cpu.max')}
    assert limits == {'memory.max': str(4 * 2**30), 'memory.swap.max': '0', 'cpu.max': '200000 100000'}
    from check_exl3_prefill_bench import headroom
    headroom(torch)
    with torch.inference_mode():
        from check_route_prepare_gpu import run as check_route_prepare
        route_prepare=check_route_prepare(torch,NativeGraph)
        save('route-prepare.json',route_prepare)
        from check_staged_moe_decode import run as check_staged_decode
        staged_decode=check_staged_decode(torch,NativeGraph,args.host_index)
        save('staged-decode.json',staged_decode)
        from check_decode_moe_occupancy import run as check_decode_occupancy
        decode_occupancy=check_decode_occupancy(torch,NativeGraph,args.host_index)
        save('decode-occupancy.json',decode_occupancy)
        from check_register_moe_gemv import run as check_register_gemv
        register_gemv=check_register_gemv(torch,NativeGraph,args.host_index)
        save('register-gemv.json',register_gemv)
        from check_dcp_head_exchange import run as check_head_exchange
        head_exchange = check_head_exchange(torch, NativeGraph)
        save('head-exchange.json',head_exchange)
        from check_online_sparse_attention import run as check_online
        online = check_online(torch, NativeGraph, rank_major=args.rank_major_query)
        save('online-attention.json',online)
        from check_dense_decode_tuning import run as check_dense_tuning
        dense_tuning = check_dense_tuning(torch, NativeGraph)
        save('dense-tuning.json',dense_tuning)
        from check_mhc_decode_prenorm import run as check_mhc_prenorm
        mhc_prenorm=check_mhc_prenorm(torch,NativeGraph)
        save('mhc-prenorm.json',mhc_prenorm)
        from check_packed_wo_a_tuning import run as check_wo_a_tuning
        wo_a_tuning=check_wo_a_tuning(torch,NativeGraph,args.host_index)
        save('wo-a-tuning.json',wo_a_tuning)
        from check_combined_attention_batch import run as check_attention_batch
        attention_batch = check_attention_batch(torch)
        save('attention-batch.json',attention_batch)
        indexer = indexer_cases(torch)
        save('indexer.json', indexer)
        topk = topk_cases(torch)
        save('topk.json', topk)
        candidates = candidate_chain_cases(torch)
        save('candidate-chain.json',candidates)
        slots = slot_cases(torch)
        save('slots.json', slots)
        dspark = None
        if dspark_enabled:
            from check_combined_dspark_gpu import run as check_dspark
            dspark = check_dspark(torch, NativeGraph)
            save('dspark.json', dspark)
        engram = engram_cases(torch)
        save('engram.json', engram)
        gc.collect()
        moe = moe_cases(torch, args.host_index)
        save('moe.json', moe)
        attention = attention_cases(torch)
        save('attention.json', attention)
        gc.collect(); torch.cuda.empty_cache()
        vocabulary_loader = None
        if vocabulary_enabled:
            from check_vocab_loader_gpu import run as check_vocab_loader
            vocabulary_loader = check_vocab_loader(torch, NativeGraph, args.host_index)
            save('vocabulary-loader.json', vocabulary_loader)
            gc.collect(); torch.cuda.empty_cache()
        from check_native_vocab_stage_gpu import run as check_vocab_stage
        vocabulary = check_vocab_stage(torch, NativeGraph, args.host_index)
        save('native-vocabulary.json', vocabulary)
        gc.collect(); torch.cuda.empty_cache()
        from check_native_draft_stage_gpu import run as check_draft_stage
        draft_records = check_draft_stage(torch, NativeGraph, args.host_index)
        save('native-draft-records.json', draft_records)
    assert moe['canonical_parity_pass'] and moe['replay_parity_pass'], moe
    peak = torch.cuda.max_memory_allocated()
    assert 0 < peak < LIMIT, dict(peak_allocated_bytes=peak,limit=LIMIT)
    report = dict(status='combined_changed_path_gpu_pass', host_index=args.host_index,
        head_exchange=head_exchange,online_attention=online,dense_tuning=dense_tuning,
        decode_occupancy=decode_occupancy,
        register_gemv=register_gemv,
        staged_decode=staged_decode,
        route_prepare=route_prepare,
        mhc_prenorm=mhc_prenorm,wo_a_tuning=wo_a_tuning,
        attention_batch=attention_batch,
        candidate_chain=candidates,
        parent_child_registration=True, descriptor=descriptor, limits=limits,
        peak_allocated_bytes=peak, allocator_fraction=.0075, indexer=indexer, slots=slots,
        engram=engram, moe=moe, attention=attention, dspark_components=dspark, topk=topk,
        native_vocabulary=vocabulary,
        native_draft_records=draft_records,
        vocabulary_loader=vocabulary_loader,
        full_model_loaded=False, actual_collectives=False,
        dspark_qualified=False, full_serving_launch_admitted=False,
        still_required=['assembled native model/collectives',
                        'full DSpark loading, memory fit and image execution', 'full serving quality and performance'])
    save('complete.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if isinstance(error, SystemExit) and error.code in (None, 0):
            raise
        if not (RESULTS / 'failed.json').exists(): save('failed.json', dict(error=repr(error)))
        raise
