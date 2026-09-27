"""GPU checks for NVFP4 prefill scoring and the searched scale selection.

Bounded (< 2 GiB device memory, no engine). Checks:
  - key writer bytes (E4M3 scale search) against the oracle;
  - per-group SSE never above /6 or four-over-six (keys and prefill queries); DeepSeek's
    MXFP4 quantizer is compared for information only (no bound is claimed);
  - NVFP4 query quantizer bytes/weights against the reference (RoPE + head scale);
  - FP4 tensor-core prefill logits and the QueryPackage path against float64;
  - speed against vLLM's MXFP4 path (fused query quantization + DeepGEMM logits).
Usage: same container as check_nvfp4_indexer_gpu.py (DS41_INDEXER_K_FORMAT=mxfp4 suffices).
"""
import json

import torch

from ds41 import nvfp4_indexer as nv

torch.manual_seed(2)
device = 'cuda'
report = {}


def rope_table(positions, dim=64, base=160000.0):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    angles = torch.outer(torch.arange(positions, dtype=torch.float64), freqs)
    return torch.cat((angles.cos(), angles.sin()), -1).float()


def group_sse(restored, x):
    return (restored.double() - x.cpu().double()).reshape(-1, 8, 16).square().sum(-1)


def compare(packed, scales, x):
    """Our group SSE against /6, four-over-six and DeepSeek's MXFP4 quantizer on the same values."""
    ours = group_sse(nv.reference_dequantize(packed, scales), x)
    div6 = group_sse(nv.reference_dequantize(*nv.reference_quantize(x, 'div6')), x)
    four = group_sse(nv.reference_dequantize(*nv.reference_quantize(x, 'four_over_six')), x)
    mx = group_sse(nv.reference_mxfp4(x), x)
    return dict(groups=int(ours.numel()), groups_worse_than_div6=int((ours > div6).sum()),
                groups_worse_than_4over6=int((ours > four).sum()),
                sse_vs_div6=float(ours.sum() / div6.sum()), sse_vs_4over6=float(ours.sum() / four.sum()),
                info_sse_vs_mxfp4=float(ours.sum() / mx.sum()), info_groups_worse_than_mxfp4=int((ours > mx).sum()))


def mixed(n):
    ties = torch.tensor([6., 0., -0., .25, -.25, .75, -.75, 1.25, -1.25, 1.75, -1.75, 2.5, -2.5, 3.5, -3.5, 5.])
    halves = torch.randn(n, 8, 16) * torch.logspace(-4, 2, 8)[torch.randint(0, 8, (n, 8))][..., None]
    return torch.cat([torch.randn(n, 128) * 3, halves.reshape(n, 128),
                      torch.randn(n, 128) * torch.logspace(-5, 2, n)[:, None],
                      ties.repeat(8).repeat(n, 1) * torch.logspace(-3, 1, n)[:, None], torch.zeros(4, 128)]).bfloat16()


# 1-2) Key writer at position 0 (identity RoPE): bytes and the /6, 4/6 bounds.
table = rope_table(8192).to(device)
keys = mixed(128)
rows = keys.shape[0]
pages = (rows + 63) // 64 + 1
cache = torch.full((pages, 64, 72), 0xA5, dtype=torch.uint8, device=device)
slots = torch.randperm(pages * 64)[:rows].to(device)
nv.store(keys.to(device), torch.zeros(rows, dtype=torch.int64, device=device), table, cache, slots, compress_ratio=1)
raw = cache.cpu().reshape(pages, -1)
s_ = slots.cpu()
values = torch.stack([raw[s // 64, (s % 64) * 64:(s % 64) * 64 + 64] for s in s_.tolist()])
scales = torch.stack([raw[s // 64, 4096 + (s % 64) * 8:4096 + (s % 64) * 8 + 8] for s in s_.tolist()])
rv, rs = nv.reference_quantize(keys)
report['key_writer'] = dict(rows=rows, value_mismatch=int((values != rv).sum()), scale_mismatch=int((scales != rs).sum()),
                            **compare(values, scales, keys))

# 3) Query quantizer: RoPE + head scale + NVFP4, then the bounds on the scaled values.
t = 257
q_pre = (torch.randn(t, 32, 128) * torch.logspace(-3, 3, 32)[None, :, None]).bfloat16()
q_pre[5] = 0
positions = torch.randint(0, 8192, (t,), dtype=torch.int64)
weights = (torch.randn(t, 32) * 0.5).bfloat16()
ss, hs = 128 ** -0.5, 32 ** -0.5
qv, qs, qw = nv.quantize_queries(q_pre.to(device), positions.to(device), table, weights.to(device), ss, hs)
roped = nv.reference_query_rope(q_pre, positions, table)
rqv, rqs, rqw = nv.reference_queries(roped, weights, ss, hs)
x = roped.float()
amax = x.abs().amax(-1).clamp_min(1e-30) * torch.tensor(1 / 1024, dtype=torch.float32)
m_, e_ = torch.frexp(amax)
scaled = torch.ldexp(x, -torch.where(m_ == 0.5, e_ - 1, e_)[..., None]).reshape(-1, 128)
report['query_quantizer'] = dict(tokens=t, value_mismatch=int((qv.cpu() != rqv).sum()), scale_mismatch=int((qs.cpu() != rqs).sum()),
    weight_max_rel=float(((qw.cpu() - rqw).abs() / rqw.abs().clamp_min(1e-30)).max()),
    **compare(qv.cpu().reshape(-1, 64), qs.cpu().reshape(-1, 8), scaled.bfloat16()))

# 4) FP4 tensor-core logits against float64 (in-range entries).
m, n = 37, 5000
kp, ks = nv.reference_quantize((torch.randn(n, 128) * 1.5).bfloat16())
qp, qsc, hw = nv.reference_queries((torch.randn(m, 32, 128) * 4).bfloat16(), torch.randn(m, 32), ss, hs)
starts = torch.randint(0, 900, (m,), dtype=torch.int32)
ends = (starts + torch.randint(1, 4000, (m,), dtype=torch.int32)).clamp(max=n)
got = nv.nvfp4_logits(qp.to(device), qsc.to(device), hw.to(device), kp.to(device), ks.to(device),
                      starts.to(device), ends.to(device)).cpu()
want = nv.reference_nvfp4_logits(qp, qsc, hw, kp, ks, starts, ends)
live = torch.isfinite(want)
err = (got[live] - want[live]).abs()
report['prefill_logits'] = dict(rows=m, keys=n, max_abs_error=float(err.max()), ref_scale=float(want[live].abs().max()),
                                in_range_finite=bool(torch.isfinite(got[live]).all()))

# 5) QueryPackage -> prefill_logits (chunk slice) against the reference pipeline.
package = nv.QueryPackage(q_pre.to(device), positions.to(device), table, weights.to(device), ss, hs)
lo, hi = 40, 77
k8 = ks.view(torch.int32)
chunk_starts = torch.zeros(hi - lo, dtype=torch.int32)
chunk_ends = torch.randint(1, n, (hi - lo,), dtype=torch.int32)
got = nv.prefill_logits(package, lo, hi, (torch.empty(hi - lo, 32, 128, dtype=torch.float8_e4m3fn, device=device), None),
                        (kp.to(device).view(torch.int8), k8.to(device)), None, chunk_starts.to(device), chunk_ends.to(device)).cpu()
want = nv.reference_nvfp4_logits(rqv[lo:hi], rqs[lo:hi], rqw[lo:hi], kp, ks, chunk_starts, chunk_ends)
live = torch.isfinite(want)
report['query_package'] = dict(rows=hi - lo, max_abs_error=float((got[live] - want[live]).abs().max()),
                               ref_scale=float(want[live].abs().max()))


# 6) Speed: whole prefill scoring step, ours vs vLLM's MXFP4 route.
def timeit(f, reps=5):
    f(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps):
        f()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


from vllm.models.deepseek_v4_1.common.ops import fused_indexer_q_rope_quant  # noqa: E402
from vllm.utils.deep_gemm import fp8_fp4_mqa_logits  # noqa: E402

timing = {}
for m, n in ((256, 32768), (1024, 65536), (2048, 131072)):
    q = (torch.randn(m, 32, 128, device=device) * 3).bfloat16()
    pos = torch.arange(n - m, n, device=device)
    w = torch.randn(m, 32, device=device).bfloat16()
    st = torch.zeros(m, dtype=torch.int32, device=device)
    en = (pos + 1).to(torch.int32)
    big = rope_table(n).to(device)
    kv_nv = torch.randint(0, 256, (n, 64), dtype=torch.uint8, device=device)
    ks_nv = torch.full((n, 8), 0x38, dtype=torch.uint8, device=device)
    kv_mx = torch.randint(0, 256, (n, 64), dtype=torch.uint8, device=device)
    ks_mx = torch.full((n, 4), 127, dtype=torch.uint8, device=device)
    pkg = nv.QueryPackage(q, pos, big, w, ss, hs)

    def ours():
        return nv.prefill_logits(pkg, 0, m, (q, None), (kv_nv, ks_nv), None, st, en)

    def native():
        (qm, qsm), wm = fused_indexer_q_rope_quant(pos, q, big, w, ss, hs, use_fp4=True)
        return fp8_fp4_mqa_logits((qm.view(torch.int8), qsm), (kv_mx.view(torch.int8), ks_mx.view(torch.int32).squeeze(-1)),
                                  wm, st, en, clean_logits=False)

    ratios = []
    for _ in range(3):
        base = timeit(native)
        ratios.append(timeit(ours) / base)
    timing[f'{m}x{n}_causal'] = dict(mxfp4_route_ms=round(base, 3), nvfp4_route_vs_mxfp4=round(sorted(ratios)[1], 3))
report['timing'] = timing
report['peak_allocated_mib'] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
print(json.dumps(report, indent=1))
