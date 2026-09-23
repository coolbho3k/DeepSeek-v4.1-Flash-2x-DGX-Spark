# SPDX-License-Identifier: AGPL-3.0-only
"""Two-Spark NCCL small-message latency at decode sizes, inside CUDA graphs.

Run one process per host with identical arguments except --rank. The NCCL
environment is inherited from the caller (the serving values, optionally
with one variant change). Reports per-collective microseconds, measured as
graph replay time divided by the number of captured collectives.
"""
import argparse
import datetime
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

# Elements: hidden 5120 x decode rows (C1 verify=4 ... C6 verify=24).
ALLREDUCE = {f'{r}x5120': r * 5120 for r in (1, 4, 12, 24)}
ALLGATHER = {f'{n}': n for n in (1024, 4096, 16384, 65536)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank', type=int, choices=(0, 1), required=True)
    p.add_argument('--master', required=True)
    p.add_argument('--port', type=int, default=29611)
    p.add_argument('--label', required=True)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    torch.cuda.set_device(0)
    dist.init_process_group('nccl', init_method=f'tcp://{a.master}:{a.port}', rank=a.rank,
                            world_size=2, timeout=datetime.timedelta(seconds=120),
                            device_id=torch.device('cuda', 0))
    results = dict(label=a.label, env={k: v for k, v in os.environ.items() if k.startswith('NCCL_')},
                   allreduce_us={}, allgather_us={})
    stream = torch.cuda.Stream()

    def timed(fn, count=64):
        with torch.cuda.stream(stream):
            for _ in range(8):
                fn()
        torch.cuda.synchronize(); dist.barrier()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            for _ in range(count):
                fn()
        torch.cuda.synchronize(); dist.barrier()
        samples = []
        for _ in range(9):
            dist.barrier(); torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); g.replay(); e.record(); torch.cuda.synchronize()
            samples.append(s.elapsed_time(e) * 1000 / count)
        return sorted(samples)[4]

    for name, n in ALLREDUCE.items():
        x = torch.randn(n, device='cuda', dtype=torch.bfloat16)
        results['allreduce_us'][name] = timed(lambda: dist.all_reduce(x))
    for name, n in ALLGATHER.items():
        x = torch.randn(n, device='cuda', dtype=torch.bfloat16)
        out = torch.empty(2 * n, device='cuda', dtype=torch.bfloat16)
        results['allgather_us'][name] = timed(lambda: dist.all_gather_into_tensor(out, x))
    dist.destroy_process_group()
    if a.rank == 0:
        print(json.dumps(results), flush=True)
        if a.output:
            with a.output.open('a') as f:
                f.write(json.dumps(results) + '\n')


if __name__ == '__main__':
    main()
