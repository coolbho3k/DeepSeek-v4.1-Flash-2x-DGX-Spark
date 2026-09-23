# SPDX-License-Identifier: AGPL-3.0-only
"""Feasibility: do prefill collectives overlap with bandwidth-bound compute?

Run one process per Spark (--rank 0/1) with the serving NCCL environment.
Stream C repeats the real prefill collective shapes (33.6 MB head-exchange
all-gather, 2046x5120 BF16 all-reduce); stream W runs either a memory-bound
weight stream (MoE-like skinny GEMMs over 2.5 GB of weights) or a
compute-bound GEMM. Each is timed alone and concurrently.
"""
import argparse
import datetime
import json

import torch
import torch.distributed as dist


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank', type=int, choices=(0, 1), required=True)
    p.add_argument('--master', required=True)
    p.add_argument('--port', type=int, default=29613)
    a = p.parse_args()
    torch.cuda.set_device(0)
    dist.init_process_group('nccl', init_method=f'tcp://{a.master}:{a.port}', rank=a.rank, world_size=2,
                            timeout=datetime.timedelta(seconds=120), device_id=torch.device('cuda', 0))
    comm_stream, work_stream = torch.cuda.Stream(), torch.cuda.Stream()
    gather_in = torch.randn(512 * 32 * 513, device='cuda')            # FP32, 33.6 MB
    gather_out = torch.empty(2 * gather_in.numel(), device='cuda')
    reduce_buf = torch.randn(2046 * 5120, device='cuda', dtype=torch.bfloat16)
    # MoE-like: 64 "experts" of [2304, 5120] BF16 = 1.5 GB streamed with few rows each.
    experts = [torch.randn(2304, 5120, device='cuda', dtype=torch.bfloat16) * .02 for _ in range(64)]
    rows = torch.randn(32, 5120, device='cuda', dtype=torch.bfloat16)
    big_a = torch.randn(2048, 5120, device='cuda', dtype=torch.bfloat16)
    big_b = torch.randn(5120, 8192, device='cuda', dtype=torch.bfloat16)

    def comm():
        for _ in range(4):
            dist.all_gather_into_tensor(gather_out, gather_in)
        dist.all_reduce(reduce_buf)

    def membound():
        for w in experts:
            torch.mm(rows, w.T)

    def computebound():
        for _ in range(6):
            torch.mm(big_a, big_b)

    def timed(pairs, reps=5):
        # pairs: [(stream, fn)]; returns median ms for all to finish.
        samples = []
        for _ in range(reps):
            dist.barrier(); torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True); start.record()
            ends = []
            for s, fn in pairs:
                s.wait_event(start)
                with torch.cuda.stream(s):
                    fn()
                    e = torch.cuda.Event(enable_timing=True); e.record(s); ends.append(e)
            torch.cuda.synchronize()
            samples.append(max(start.elapsed_time(e) for e in ends))
        return sorted(samples)[reps // 2]

    for _ in range(2):
        timed([(comm_stream, comm)]); timed([(work_stream, membound)]); timed([(work_stream, computebound)])
    result = dict(
        comm_alone_ms=timed([(comm_stream, comm)]),
        membound_alone_ms=timed([(work_stream, membound)]),
        computebound_alone_ms=timed([(work_stream, computebound)]),
        comm_with_membound_ms=timed([(comm_stream, comm), (work_stream, membound)]),
        comm_with_computebound_ms=timed([(comm_stream, comm), (work_stream, computebound)]),
    )
    result['membound_overlap_efficiency'] = (
        (result['comm_alone_ms'] + result['membound_alone_ms'] - result['comm_with_membound_ms'])
        / min(result['comm_alone_ms'], result['membound_alone_ms']))
    result['computebound_overlap_efficiency'] = (
        (result['comm_alone_ms'] + result['computebound_alone_ms'] - result['comm_with_computebound_ms'])
        / min(result['comm_alone_ms'], result['computebound_alone_ms']))
    dist.destroy_process_group()
    if a.rank == 0:
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
