# SPDX-License-Identifier: AGPL-3.0-only
"""Two-Spark test of libfastcomm against NCCL: bit-exactness and in-graph latency.

Run one process per host with the serving NCCL environment. NCCL is used both
as the reference and for bootstrapping (exchanging RDMA connection info).
"""
import argparse
import ctypes as C
import datetime
import json

import torch
import torch.distributed as dist

SIZES = {'1x5120': 5120, '4x5120': 20480, '12x5120': 61440, '24x5120': 122880,
         '48x5120': 245760, '76x5120': 389120}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank', type=int, choices=(0, 1), required=True)
    p.add_argument('--master', required=True)
    p.add_argument('--port', type=int, default=29617)
    p.add_argument('--lib', required=True)
    p.add_argument('--device', default='rocep1s0f1,roceP2p1s0f1')
    p.add_argument('--gid', type=int, default=3)
    a = p.parse_args()
    torch.cuda.set_device(0)
    dist.init_process_group('nccl', init_method=f'tcp://{a.master}:{a.port}', rank=a.rank, world_size=2,
                            timeout=datetime.timedelta(seconds=120), device_id=torch.device('cuda', 0))
    lib = C.CDLL(a.lib)
    lib.fc_create.restype = C.c_void_p
    lib.fc_create.argtypes = [C.c_char_p, C.c_int, C.c_uint64, C.c_void_p, C.POINTER(C.c_int)]
    lib.fc_connect.argtypes = [C.c_void_p, C.c_void_p]
    lib.fc_error.argtypes = [C.c_void_p]
    for name in ('fc_allreduce_bf16',):
        getattr(lib, name).argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_uint64, C.c_int, C.c_void_p]
    lib.fc_allgather.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_uint64, C.c_int, C.c_int, C.c_void_p]
    info = C.create_string_buffer(256); size = C.c_int(0)
    handle = lib.fc_create(a.device.encode(), a.gid, 1 << 20, info, C.byref(size))
    assert handle, 'fc_create failed'
    mine = torch.frombuffer(bytearray(info.raw[:size.value]), dtype=torch.uint8).cuda()
    both = torch.empty(2 * size.value, dtype=torch.uint8, device='cuda')
    dist.all_gather_into_tensor(both, mine)
    peer = bytes(both[(1 - a.rank) * size.value:(2 - a.rank) * size.value].cpu().numpy())
    assert lib.fc_connect(handle, peer) == 0, 'fc_connect failed'
    dist.barrier()

    def fc_allreduce(x, out, ch=0):
        s = torch.cuda.current_stream().cuda_stream
        assert lib.fc_allreduce_bf16(handle, x.data_ptr(), out.data_ptr(), x.numel(), ch, s) == 0

    def fc_allgather(x, out, ch=1):
        s = torch.cuda.current_stream().cuda_stream
        assert lib.fc_allgather(handle, x.data_ptr(), out.data_ptr(), x.numel() * x.element_size(), ch, a.rank, s) == 0

    report = dict(exactness={}, latency_us={})
    torch.manual_seed(1000 + a.rank)
    # Exactness: eager, many iterations with changing data, both channels.
    for name, n in SIZES.items():
        mismatches = 0
        for it in range(300):
            x = torch.randn(n, device='cuda', dtype=torch.bfloat16) * (1 + it % 7)
            ref = x.clone(); dist.all_reduce(ref)
            out = torch.empty_like(x); fc_allreduce(x, out)
            g_ref = torch.empty(2 * n, device='cuda', dtype=torch.bfloat16); dist.all_gather_into_tensor(g_ref, x)
            g_out = torch.empty_like(g_ref); fc_allgather(x, g_out)
            mismatches += int(not torch.equal(ref, out)) + int(not torch.equal(g_ref, g_out))
        report['exactness'][name] = dict(iterations=300, mismatches=mismatches, error=lib.fc_error(handle))

    def graph_time(fn, count=64):
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(4): fn()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize(); dist.barrier()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(count): fn()
        torch.cuda.synchronize(); dist.barrier()
        t = []
        for _ in range(9):
            dist.barrier(); torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
            t.append(e0.elapsed_time(e1) * 1000 / count)
        return sorted(t)[4]

    for name, n in SIZES.items():
        x = torch.randn(n, device='cuda', dtype=torch.bfloat16)
        out = torch.empty_like(x); g_out = torch.empty(2 * n, device='cuda', dtype=torch.bfloat16)
        report['latency_us'][name] = dict(
            nccl_allreduce=graph_time(lambda: dist.all_reduce(x)),
            fc_allreduce=graph_time(lambda: fc_allreduce(x, out)),
            nccl_allgather=graph_time(lambda: dist.all_gather_into_tensor(g_out, x)),
            fc_allgather=graph_time(lambda: fc_allgather(x, g_out)))
    # Graph replay exactness after timing (sequence numbers advance across replays).
    x = torch.randn(20480, device='cuda', dtype=torch.bfloat16)
    out = torch.empty_like(x)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fc_allreduce(x, out)
    bad = 0
    for it in range(300):
        x.copy_(torch.randn_like(x) * (it % 5 + 1)); g.replay()
        ref = x.clone(); dist.all_reduce(ref); bad += int(not torch.equal(ref, out))
    report['graph_replay_exact'] = dict(replays=300, mismatches=bad, error=lib.fc_error(handle))
    dist.barrier()
    if a.rank == 0:
        print(json.dumps(report), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
