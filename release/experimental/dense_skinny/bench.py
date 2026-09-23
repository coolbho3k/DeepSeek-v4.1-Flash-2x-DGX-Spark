# SPDX-License-Identifier: AGPL-3.0-only
"""Component A/B: native B12X MXFP8 linear versus the skinny Triton kernel.

Synthetic weights at the real per-rank decode shapes. Each timed CUDA graph
cycles through enough weight copies to exceed the L2 cache, so every call
streams its weights from DRAM as in the full model. Outputs are compared
element-wise with the native kernel. Not a model-quality or serving result.
"""
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import sys

SHAPES = {'wq_a_wkv': (1792, 5120), 'wq_b': (16384, 1280), 'wo_b': (5120, 4096),
          'shared_gate_up': (2304, 5120), 'shared_down': (5120, 1152)}
ROWS = (1, 3, 4, 8, 12, 16, 24)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sweep-rows', type=int, default=4)
    a = p.parse_args()
    assert os.environ.get('B12X_DENSE_SPLITK_TURBO') == '0'
    import torch
    sys.path.insert(0, '/opt/ds41-serving')
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import spark_b12x_fp32_reduce as reducer
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import Mxfp8LinearLayerConfig
    from vllm.model_executor.kernels.linear.mxfp8.b12x import B12xMxfp8LinearKernel
    from b12x.gemm._shared.block_fp8 import quantize_block_fp8_linear_input_mxfp8 as quantize
    import skinny_mxfp8 as skinny
    reducer.register()
    torch.manual_seed(20260923)
    kernel = B12xMxfp8LinearKernel(Mxfp8LinearLayerConfig())
    results = dict(status='running', shapes={})

    def graph_time(fns, repeats=8):
        # fns: list of zero-arg callables (one per weight copy); returns us per call.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for f in fns:
                f()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(repeats):
                for f in fns:
                    f()
        g.replay(); torch.cuda.synchronize()
        samples = []
        for _ in range(7):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); g.replay(); end.record(); torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / (repeats * len(fns)))
        del g
        return sorted(samples)[len(samples) // 2]

    with torch.inference_mode():
        for name, (n, k) in SHAPES.items():
            copies = max(2, math.ceil(96 * 2**20 / (n * k)))
            layers = []
            for _ in range(copies):
                w = torch.randn((n, k), device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
                s = torch.randint(118, 124, (n, k // 32), dtype=torch.uint8, device='cuda')
                layer = torch.nn.Module()
                layer.weight = torch.nn.Parameter(w, requires_grad=False)
                layer.weight_scale = torch.nn.Parameter(s, requires_grad=False)
                kernel.process_weights_after_loading(layer)
                packed = layer.b12x_mxfp8_packed_weight.weight
                assert packed.values.shape == (n, k) and packed.values.is_contiguous()
                assert packed.scale_rows.numel() == n * k // 32
                layers.append((layer, packed))
            row = results['shapes'][name] = dict(shape=[n, k], copies=copies, rows={})

            def native(layer, x):
                return kernel.apply_weights(layer, x)

            def candidate(packed, x, **cfg):
                q = quantize(x)
                return skinny.forward(q.values, q.scale_rows.view(torch.uint8),
                                      packed.values, packed.scale_rows.view(torch.uint8), **cfg)

            configs = [dict(bn=bn, blocks=bl, splits=sp, warps=wp, stages=st) for bn, bl, sp, wp, st in
                       itertools.product((32, 64, 128), (2, 4, 8), (1, 2, 4), (4,), (2, 3, 4))]
            xs = torch.randn((a.sweep_rows, k), device='cuda', dtype=torch.bfloat16)
            sweep = []
            for cfg in configs:
                try:
                    t = graph_time([lambda p=p, c=cfg: candidate(p, xs, **c) for _, p in layers])
                except Exception as error:  # resource limits for some tiles
                    sweep.append(dict(cfg=cfg, error=repr(error)[:200])); continue
                sweep.append(dict(cfg=cfg, us=t))
            ok = [s for s in sweep if 'us' in s]
            best = min(ok, key=lambda s: s['us'])['cfg']
            row['sweep_rows'] = a.sweep_rows
            row['best_config'] = best
            row['sweep_top5'] = sorted(ok, key=lambda s: s['us'])[:5]
            for m in ROWS:
                x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16) * 0.5
                ref = native(layers[0][0], x)
                out = candidate(layers[0][1], x, **best)
                diff = (out.float() - ref.float())
                nmse = float(diff.square().sum() / ref.float().square().sum().clamp_min(1e-30))
                t_native = graph_time([lambda l=l: native(l, x) for l, _ in layers])
                t_cand = graph_time([lambda p=p: candidate(p, x, **best) for _, p in layers])
                row['rows'][m] = dict(native_us=t_native, candidate_us=t_cand, ratio=t_cand / t_native,
                    native_gbps=n * k / t_native / 1e3, candidate_gbps=n * k / t_cand / 1e3,
                    identical_fraction=float((out == ref).float().mean()), nmse=nmse,
                    max_abs=float(diff.abs().max()))
                print(json.dumps(dict(shape=name, rows=m, **row['rows'][m])), flush=True)
            del layers
            torch.cuda.empty_cache()
            a.output.write_text(json.dumps(results, indent=1) + '\n')
    results['status'] = 'complete'
    a.output.write_text(json.dumps(results, indent=1) + '\n')


if __name__ == '__main__':
    main()
