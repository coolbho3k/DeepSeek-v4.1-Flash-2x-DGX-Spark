# SPDX-License-Identifier: AGPL-3.0-only
"""Component A/B for the mHC prenorm row-reuse and router projection kernels.

Uses the installed native prenorm variant and torch.mm(out_dtype=fp32) as
controls. CUDA graphs cycle through enough weight copies to exceed L2.
"""
import itertools, json, math, sys
from pathlib import Path
import torch
sys.path.insert(0, '/opt/ds41-serving'); sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds41 import mhc_decode_prenorm as native
import kernels


def graph_time(fns, repeats=8):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns: f()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(repeats):
            for f in fns: f()
    g.replay(); torch.cuda.synchronize()
    t = []
    for _ in range(7):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        t.append(a.elapsed_time(b) * 1000 / (repeats * len(fns)))
    return sorted(t)[3]


def main():
    out = dict(prenorm=[], router=[])
    torch.manual_seed(3)
    with torch.inference_mode():
        # mHC prenorm: native serving uses splits 16 at K=20480 (4 at 5120).
        for k, splits in ((20480, 16), (5120, 4)):
            copies = max(2, math.ceil(96 * 2**20 / (24 * k * 4)))
            fns = [torch.randn(24, k, device='cuda') * 0.01 for _ in range(copies)]
            for rows in (1, 2, 3, 4):
                x = torch.randn(rows, k, device='cuda', dtype=torch.bfloat16)
                o1 = torch.empty(splits, rows, 24, device='cuda'); s1 = torch.empty(splits, rows, device='cuda')
                o2 = torch.empty_like(o1); s2 = torch.empty_like(s1)
                native.forward(x, fns[0], o1, s1, splits, tile_n=4, warps=4)
                kernels.prenorm(x, fns[0], o2, s2, splits, tile_n=4, warps=4)
                exact = bool(torch.equal(o1, o2) and torch.equal(s1, s2))
                t0 = graph_time([lambda f=f: native.forward(x, f, o1, s1, splits, tile_n=4, warps=4) for f in fns])
                best = None
                for tn, wp in itertools.product((4, 8), (4, 8)):
                    native.forward(x, fns[0], o1, s1, splits, tile_n=4, warps=4)
                    kernels.prenorm(x, fns[0], o2, s2, splits, tile_n=tn, warps=wp)
                    ok = bool(torch.equal(o1, o2) and torch.equal(s1, s2))
                    t = graph_time([lambda f=f: kernels.prenorm(x, f, o2, s2, splits, tile_n=tn, warps=wp) for f in fns])
                    if ok and (best is None or t < best[0]): best = (t, tn, wp)
                row = dict(k=k, rows=rows, native_us=t0, candidate_us=best[0], tile_n=best[1], warps=best[2],
                           bit_exact=exact and best is not None)
                out['prenorm'].append(row); print(json.dumps(row), flush=True)
        for n in (384, 128):
            copies = max(2, math.ceil(96 * 2**20 / (n * 5120 * 2)))
            ws = [torch.randn(n, 5120, device='cuda', dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
            for rows in (1, 3, 4, 8, 12, 18, 24):
                x = torch.randn(rows, 5120, device='cuda', dtype=torch.bfloat16)
                ref = torch.mm(x, ws[0].T, out_dtype=torch.float32)
                t0 = graph_time([lambda w=w: torch.mm(x, w.T, out_dtype=torch.float32) for w in ws])
                cands = []
                for bn, bk, wp in itertools.product((16, 32), (64, 128, 256), (4,)):
                    cands.append(('dot', dict(bn=bn, bk=bk, warps=wp), kernels.router))
                if rows <= 8:
                    for bn, bk, wp in itertools.product((1, 2, 4), (128, 256, 512), (2, 4)):
                        cands.append(('fma', dict(bn=bn, bk=bk, warps=wp), kernels.router_fma))
                best = None
                for kind, cfg, fn in cands:
                    try:
                        y = fn(x, ws[0], **cfg)
                        t = graph_time([lambda w=w: fn(x, w, **cfg) for w in ws])
                    except Exception:
                        continue
                    if best is None or t < best[0]:
                        rel = float(((y - ref).abs().max() / ref.abs().max()))
                        best = (t, kind, cfg, rel, float((y == ref).float().mean()))
                row = dict(n=n, rows=rows, cublas_us=t0, candidate_us=best[0], kind=best[1], cfg=best[2],
                           max_rel_diff=best[3], identical_fraction=best[4])
                out['router'].append(row); print(json.dumps(row), flush=True)
    Path('/results/components-v1.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
