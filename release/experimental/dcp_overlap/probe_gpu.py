# SPDX-License-Identifier: AGPL-3.0-only
# The NativeGraph harness and FP8 fixture are adapted from this recipe's
# AGPLv3 probes, built on MiaAI's serving stack. See README.md for attribution.
"""Deferred two-process/two-GPU correctness probe. NEVER run beside serving.

Run in two fresh containers using the candidate overlay and existing pinned
runtime. One torchrun worker per Spark; this script creates no containers,
changes no networking, loads no model, and never stops a running process.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import runpy
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace


def admit(args):
    if not args.acknowledge_idle_gpus:
        raise RuntimeError('Explicit idle-GPU acknowledgment is required')
    if int(os.environ.get('WORLD_SIZE', '0')) != 2 or int(os.environ.get('LOCAL_WORLD_SIZE', '0')) != 1:
        raise RuntimeError('Use two hosts with exactly one torchrun worker on each')
    active = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    if active:
        raise RuntimeError('A GPU process is already present; this probe never stops it')
    memory = dict((line.split(':')[0], int(line.split()[1]) * 1024)
                  for line in Path('/proc/meminfo').read_text().splitlines()
                  if line.startswith(('MemAvailable:', 'MemFree:')))
    if memory['MemAvailable'] < 8 * 2**30 or memory['MemFree'] < 2 * 2**30:
        raise RuntimeError('Insufficient idle-host headroom for component qualification')
    if args.overlay != Path('/opt/ds41-serving') or not (args.overlay / 'serve.py').is_file():
        raise RuntimeError('Mount the immutable candidate at the runtime native overlay path')


class NativeGraph:
    def __init__(self, function, tokens):
        import torch
        from vllm.config import CUDAGraphMode
        from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager, BatchExecutionDescriptor
        function()
        torch.cuda.synchronize()
        self.manager = object.__new__(CudaGraphManager)
        self.manager.device = torch.device('cuda', torch.cuda.current_device())
        self.manager.use_breakable_cg = False
        self.manager.graphs = {}
        self.key = BatchExecutionDescriptor(CUDAGraphMode.FULL, max(tokens, 1), 1)
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
        self.manager._ds41_graph_resources.clear()
        self.manager.graphs.clear()


class WireGroup:
    """Real two-rank PyNCCL, with baseline rank-major all-gather semantics.

    Not a simulated peer: rank queries, sinks, and partials cross the link.
    Native vLLM group installation still requires subsequent serving tests.
    """
    world_size = 2

    def __init__(self, torch, comm):
        self.torch, self.comm, self.rank_in_group = torch, comm, comm.rank
        self.device_communicator = SimpleNamespace(pynccl_comm=comm)

    def all_gather(self, value, dim):
        torch = self.torch
        receive = torch.empty((2, *value.shape), device=value.device, dtype=value.dtype)
        self.comm.all_gather(receive, value.contiguous(), stream=torch.cuda.current_stream())
        # Match CudaCommunicator's exact movedim/reshape path: dim=0 is a
        # zero-copy view. Do not penalize the baseline with an extra cat.
        return receive.movedim(0, dim).reshape(
            value.shape[:dim] + (2 * value.shape[dim],) + value.shape[dim + 1:])


def pack_swa(torch, values):
    blocks, size, _ = values.shape
    grouped = values[..., :448].float().reshape(blocks, size, 7, 64)
    exponent = torch.ceil(torch.log2(grouped.abs().amax(-1).clamp_min(1e-4) / 448))
    quantized = (grouped * torch.exp2(-exponent).unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    cache = torch.zeros((blocks, size, 584), dtype=torch.uint8, device=values.device)
    pages = cache.reshape(blocks, -1)
    payload = pages[:, :size * 576].reshape(blocks, size, 576)
    scales = pages[:, size * 576:].reshape(blocks, size, 8)
    payload[..., :448] = quantized.reshape(blocks, size, 448).view(torch.uint8)
    payload[..., 448:] = values[..., 448:].contiguous().view(torch.uint8)
    scales[..., :7] = (exponent + 127).clamp(0, 255).to(torch.uint8)
    return cache


def exact(torch, actual, expected, label):
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise AssertionError(label + ': dtype/shape mismatch')
    # Check signed zero and NaN payloads too. Never substitute an NMSE limit.
    if not torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)):
        raise AssertionError(label + ': bitwise mismatch')


def fixture(torch, rank, rows, decode, image, swa_only=False, topk=512):
    from ds41 import fp4_main_kv as codec
    # Replicated SWA bytes; physically distinct compressed shards and TP heads.
    torch.manual_seed(41017)
    swa = pack_swa(torch, (torch.randn(32, 128, 512, device='cuda') * .2).bfloat16())
    torch.manual_seed(41018 + rank)
    main = torch.empty((8, 128, 288), device='cuda', dtype=torch.uint8)
    codec.store(main, (torch.randn(1024, 512, device='cuda') * .3).bfloat16(),
                torch.arange(1024, device='cuda'))
    query = (torch.randn(rows, 32, 512, device='cuda') * .2 + rank * .05).bfloat16()
    width = 4096 if image else 128
    ids = ((torch.arange(width, device='cuda')[None, :] +
            torch.arange(rows, device='cuda')[:, None] * 7) % 4096).int()
    ids[:, ::11] = -1
    ids[:, 2:4] = 5  # Duplicate keys must stay duplicated.
    counts = torch.tensor(([0, 1, 32, width, width - 1] * ((rows + 4) // 5))[:rows],
                          device='cuda', dtype=torch.int32)
    candidates = ((torch.arange(topk, device='cuda')[None, :] +
                   torch.arange(rows, device='cuda')[:, None] * 3) % 2048).int()
    candidates[:, ::13] = -1
    candidates[:, 1:3] = 8
    model = SimpleNamespace(attn_sink=torch.linspace(-2., 2., 32, device='cuda') + rank,
                            compress_ratio=0 if swa_only else 2, scale=512 ** -.5,
                            topk_indices_buffer=candidates)
    metadata = SimpleNamespace(num_decode_tokens=decode, num_prefill_tokens=rows - decode,
        decode_swa_indices=ids[:decode], decode_swa_lens=counts[:decode],
        prefill_swa_indices=ids[decode:], prefill_swa_lens=counts[decode:],
        token_to_req_indices=(torch.arange(rows, device='cuda') % 2).int(),
        is_valid_token=torch.arange(rows, device='cuda') % 7 != 0)
    compressed = SimpleNamespace(block_size=256,
        block_table=torch.stack((torch.arange(8, device='cuda'), torch.arange(7, -1, -1, device='cuda'))).int())
    return model, query, compressed, metadata, main, swa, ids


def check_partials(torch, group, original, values):
    """Compare FP32 outputs/LSEs before the final BF16 store can hide drift."""
    from ds41.dcp_overlap.attention import make_head_attention
    model, query, compressed, metadata, main, swa, ids = values
    if not len(query) or not model.compress_ratio:
        return
    rows = min(len(query), 512)
    math = original.__globals__['bf16_sparse_attention_with_lse']
    heads = make_head_attention(math)
    partition = original.__globals__['partition_indices']
    mapper = original.__globals__['sparse_global_to_local_slots']
    all_q = group.all_gather(query[:rows].contiguous(), dim=1)
    sinks = original.__globals__['split_sink'](group.all_gather(model.attn_sink, dim=0), 2)
    lengths = torch.cat((metadata.decode_swa_lens, metadata.prefill_swa_lens))[:rows]
    si, sl = partition(ids[:rows], lengths, group.rank_in_group, 2, localize=False)
    valid = metadata.is_valid_token[:rows]
    candidates = torch.where(valid[:, None], model.topk_indices_buffer[:rows], -1)
    ci, cl = mapper(candidates, torch.full((rows,), candidates.shape[1], device='cuda', dtype=torch.int32),
        torch.where(valid, metadata.token_to_req_indices[:rows], 0), compressed.block_table,
        128, 2, group.rank_in_group)
    extra = dict(compressed_cache=main, compressed_indices=ci, compressed_lengths=cl)
    reference, lse = math(all_q, swa, si, sl, sinks=sinks, scale=model.scale, **extra)
    tile = 16 if rows < 32 else 32
    for begin in range(0, 64, tile):
        actual, actual_lse = heads(all_q[:, begin:begin + tile], swa, si, sl,
            sinks=sinks[begin:begin + tile], scale=model.scale, **extra)
        exact(torch, actual, reference[:, begin:begin + tile], 'FP32 head partial')
        exact(torch, actual_lse, lse[:, begin:begin + tile], 'FP32 head LSE')


def check_nonfinite_merge(torch, group, transport):
    from ds41 import dcp_head_exchange as baseline
    from ds41.dcp_overlap.attention import allocate_outputs, pack_result, PendingMerge
    rank = group.rank_in_group
    for rows in (1, 33):
        torch.manual_seed(491 + rank)
        full = torch.randn(rows, 64, 512, device='cuda')
        lse = torch.randn(rows, 64, device='cuda')
        lse[:, :6] = torch.tensor([-torch.inf, torch.inf, torch.nan, 1000., -1000., 0.], device='cuda')
        if rank:
            lse[:, :6] = torch.tensor([-torch.inf, torch.inf, 0., -1000., 1000., -torch.inf], device='cuda')
            full[:, 5] = torch.nan
        expected = torch.empty((rows, 32, 512), device='cuda', dtype=torch.bfloat16)
        baseline.merge_packed(full, lse, group.all_gather(baseline.pack_result(full, lse, rank), dim=0), rank, expected)
        output = torch.empty_like(expected)
        peer = (1 - rank) * 32
        remote, remote_lse = allocate_outputs(output, 1)
        remote.copy_(full[:, peer:peer + 32]); remote_lse.copy_(lse[:, peer:peer + 32])
        own = rank * 32
        locals_ = ((full[:, own:own + 16], lse[:, own:own + 16]),
                   (full[:, own + 16:own + 32], lse[:, own + 16:own + 32])) if rows < 32 else (
                   (full[:, own:own + 32], lse[:, own:own + 32]),)
        payload, head_major = pack_result(remote, remote_lse)
        def invoke():
            with transport.session():
                ticket = transport.gather(payload, kind='result')
                PendingMerge(ticket, locals_, output, rank, head_major).finish()
            return output
        exact(torch, invoke(), expected, 'nonfinite merge')
        graph = NativeGraph(invoke, rows)
        for _ in range(3):
            exact(torch, graph.replay(), expected, 'nonfinite graph merge')
        graph.close()


def run_case(torch, group, original, candidate, rows, decode, image, swa_only, timed, trace, topk=512):
    import torch.distributed as dist
    values = fixture(torch, group.rank_in_group, rows, decode, image, swa_only, topk)
    check_partials(torch, group, original, values)
    model, query, compressed, metadata, main, swa, ids = values
    expected, output = torch.empty_like(query), torch.empty_like(query)
    def call(function, destination):
        function(model, query, destination, compressed, metadata, main, swa, swa_only, group=group)
        return destination
    call(original, expected)
    call(candidate, output)
    exact(torch, output, expected, 'eager native forward')
    graphs = [NativeGraph(lambda: call(original, expected), rows),
              NativeGraph(lambda: call(candidate, output), rows)]
    streams = [torch.cuda.current_stream(), torch.cuda.Stream()]
    for step in range(4):
        torch.cuda.synchronize()
        with torch.cuda.stream(streams[step % 2]):
            query.mul_(-.875)
            model.attn_sink.add_(.0625)
            model.topk_indices_buffer[:, 5] = (step * 9) % 2048
            ids[:, 5] = (step * 31) % 4096
            graphs[0].replay()
            graphs[1].replay()
            exact(torch, output, expected, 'owned graph replay')
    torch.cuda.synchronize()
    for graph in graphs:
        graph.close()
    del graphs
    timing = {}
    if timed and rows:
        repeats = 16 if rows <= 64 else 4
        def batch(function, destination):
            for _ in range(repeats):
                call(function, destination)
            return destination
        # More than one attention layer per replay suppresses Python/rank
        # arrival jitter. Both variants are alternated in the same session.
        graphs = [NativeGraph(lambda: batch(original, expected), rows),
                  NativeGraph(lambda: batch(candidate, output), rows)]
        samples = {label: dict(wall=[], gpu=[]) for label in ('baseline', 'overlap')}
        for trial in range(16):
            for index in ((0, 1) if trial % 2 == 0 else (1, 0)):
                label, graph = ('baseline', 'overlap')[index], graphs[index]
                dist.barrier()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin = time.perf_counter()
                a.record(); graph.replay(); b.record(); b.synchronize()
                if trial >= 4:
                    samples[label]['wall'].append((time.perf_counter() - begin) * 1000 / repeats)
                    samples[label]['gpu'].append(a.elapsed_time(b) / repeats)
        for label, data in samples.items():
            timing[label] = dict(median_wall_ms=statistics.median(data['wall']),
                median_gpu_ms=statistics.median(data['gpu']), per_replay=repeats,
                wall_samples_ms=data['wall'], gpu_samples_ms=data['gpu'])
        if trace and rows in (4, 512) and not image and not swa_only and topk == 512:
            dist.barrier()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                for label, graph in zip(('baseline', 'overlap'), graphs):
                    with torch.profiler.record_function(label):
                        graph.replay()
                torch.cuda.synchronize()
            profiler.export_chrome_trace('/results/trace-' + str(rows) + '.json')
        for graph in graphs:
            graph.close()
        if rows >= 32:
            # Production prefill is eager (FULL_DECODE_ONLY), not the captured
            # component path. Preserve synchronous bounds checks in this test.
            samples = {label: [] for label in ('baseline', 'overlap')}
            for trial in range(10):
                for index in ((0, 1) if trial % 2 == 0 else (1, 0)):
                    dist.barrier()
                    begin = time.perf_counter()
                    call((original, candidate)[index], (expected, output)[index])
                    torch.cuda.synchronize()
                    if trial >= 2:
                        samples[('baseline', 'overlap')[index]].append((time.perf_counter() - begin) * 1000)
            timing['eager'] = {label: dict(median_wall_ms=statistics.median(data), wall_samples_ms=data)
                               for label, data in samples.items()}
    return dict(rows=rows, decode=decode, image_width=image, swa_only=swa_only, topk=topk,
                eager_exact=True, graph_replays_exact=4, streams=2, timing=timing)


def invalid_replay(torch, group, candidate):
    """A terminal negative case; never destroy/reuse a poisoned graph owner."""
    model, query, compressed, metadata, main, swa, ids = fixture(torch, group.rank_in_group, 4, 4, False)
    output = torch.empty_like(query)
    def invoke():
        candidate(model, query, output, compressed, metadata, main, swa, False, group=group)
        return output
    graph = NativeGraph(invoke, 4)
    # Each rank owns its deliberately invalid SWA index. The kernel must mask
    # its load and the existing owner must report the flag after graph replay.
    ids[:, 0] = 4096 + group.rank_in_group
    metadata.decode_swa_lens.fill_(1)
    try:
        graph.replay()
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid captured sparse index was not rejected')
    owner = graph.manager._ds41_graph_resources.owners[graph.key]
    if not owner.failed:
        raise AssertionError('Invalid replay did not poison its graph owner')
    try:
        graph.replay()
    except RuntimeError as error:
        if 'poisoned' not in str(error):
            raise
    else:
        raise AssertionError('Poisoned graph replay was admitted')
    # Retain until process exit; do not run CUDA cleanup on a failed owner.
    globals()['_failed_probe_graph'] = graph
    print(json.dumps(dict(status='negative_component_pass', rank=group.rank_in_group,
        invalid_sparse_index_rejected=True, poisoned_graph_replay_rejected=True,
        full_model_qualified=False)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acknowledge-idle-gpus', action='store_true')
    parser.add_argument('--overlay', type=Path, default=Path('/opt/ds41-serving'))
    parser.add_argument('--timing', action='store_true', help='Component timing only, not tokens/sec')
    parser.add_argument('--trace', action='store_true', help='Export small decoded/prefill GPU traces')
    parser.add_argument('--invalid-replay', action='store_true',
                        help='Separate disposable-process negative test; no normal cases afterward')
    args = parser.parse_args()
    admit(args)  # Runs before torch import/CUDA initialization.
    import torch
    import torch.distributed as dist
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.0075)
    sys.path.insert(0, str(args.overlay))
    runpy.run_path(str(args.overlay / 'serve.py'), run_name='dcp_overlap_probe_entry')
    from ds41.dcp_overlap import policy
    from ds41.dcp_overlap.transport import prepare
    from ds41 import vllm_fp4_main as fp4
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    if policy.MODE == 'off':
        raise RuntimeError('Use a query or balanced candidate for the paired probe')
    dist.init_process_group('gloo', timeout=timedelta(seconds=90))
    comm = PyNcclCommunicator(dist.group.WORLD, device=torch.device('cuda', 0))
    group = WireGroup(torch, comm)
    transport = prepare(group)
    candidate = fp4._forward
    original = candidate.__ds41_overlap_original__
    reports = []
    with torch.inference_mode():
        if args.invalid_replay:
            invalid_replay(torch, group, candidate)
            return  # Process exit retains the poisoned owner; no CUDA recovery.
        check_nonfinite_merge(torch, group, transport)
        # Both split-K decode paths, BH16/BH32 boundary, chunk carry/drain,
        # decode-to-prefill transition, image-width SWA, and unchanged SWA-only.
        cases = [(n, n, False, False) for n in (0, 1, 2, 3, 4, 8, 12, 18, 24, 31, 32, 33, 64)]
        cases += [(512, 0, False, False), (513, 0, False, False),
                  (516, 4, True, False), (24, 6, True, False),
                  (4, 4, True, True), (33, 1, True, True)]
        cases = [(*case, 512) for case in cases]
        cases += [(4, 4, False, False, 1024), (24, 24, False, False, 1024),
                  (512, 0, False, False, 1024), (24, 6, True, False, 1024)]
        for rows, decode, image, swa_only, topk in cases:
            row = run_case(torch, group, original, candidate, rows, decode, image, swa_only,
                           args.timing, args.trace, topk)
            if transport.pending or transport.failed:
                raise AssertionError('Unjoined/poisoned overlap transport')
            if torch.cuda.max_memory_allocated() > 512 * 2**20:
                raise AssertionError('Probe exceeded component tensor budget')
            reports.append(row)
            print(json.dumps(dict(rank=group.rank_in_group, case=row)), flush=True)
    dist.barrier()
    print(json.dumps(dict(status='component_pass', mode=policy.MODE, rank=group.rank_in_group,
        cases=reports, actual_two_rank_collectives=True, full_model_qualified=False,
        native_process_group_setup_qualified=False, image_encoder_qualified=False,
        peak_allocated_bytes=torch.cuda.max_memory_allocated())), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
