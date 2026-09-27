"""GPU checks for the opt-in NVFP4 index keys (ds41/nvfp4_indexer.py).

Bounded: < 1 GiB of device memory, no model, no vLLM engine. Checks the writer
bytes against the nearest-code oracle's scale search (identity RoPE), that no group
is worse than /6 or four-over-six would give, RoPE
positions, compress-ratio-2 group boundaries, skipped slots, the request gather,
and FP8-query x NVFP4-key logits against a float64 reference; then times logits.
Usage (serving image, repo at /work):
  docker run --rm --gpus all -e DS41_INDEXER_K_FORMAT=mxfp4 -v REPO:/work:ro \
    -e PYTHONPATH=/work/release/runtime/serving IMAGE /work/release/runtime/probes/check_nvfp4_indexer_gpu.py
"""
import json
import math

import torch

from ds41 import nvfp4_indexer as nv

torch.manual_seed(0)
device = 'cuda'
report = {}


def rope_table(positions, dim=64, base=160000.0):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    angles = torch.outer(torch.arange(positions, dtype=torch.float64), freqs)
    return torch.cat((angles.cos(), angles.sin()), -1).float()


def reference_rope(keys, positions, table, ratio):
    """Parity arithmetic in float64: fma(-odd, sin, even*cos) and fma(even, sin, odd*cos), then BF16."""
    x = keys.cpu().double()
    even, odd = x[:, 0::2].clone(), x[:, 1::2].clone()
    p = (positions.cpu() // ratio) * ratio
    cos = table.cpu().double()[p][:, :32]
    sin = table.cpu().double()[p][:, 32:]
    ne = (-odd[:, 32:] * sin + (even[:, 32:] * cos).float().double()).float().bfloat16()
    no = (even[:, 32:] * sin + (odd[:, 32:] * cos).float().double()).float().bfloat16()
    even, odd = even.bfloat16(), odd.bfloat16()
    even[:, 32:], odd[:, 32:] = ne, no
    return torch.stack((even, odd), -1).reshape(-1, 128)


def read_rows(cache, slots):
    """Segregated pages: value rows at row*64, scale rows at states*64 + row*8."""
    states = cache.shape[1]
    raw = cache.cpu().reshape(cache.shape[0], -1)
    values = torch.stack([raw[s // states, (s % states) * 64:(s % states) * 64 + 64] for s in slots.tolist()])
    scales = torch.stack([raw[s // states, states * 64 + (s % states) * 8:states * 64 + (s % states) * 8 + 8]
                          for s in slots.tolist()])
    return values, scales


def keys_mix(n):
    parts = [torch.randn(n, 128) * 3,
             torch.randn(n, 128) * torch.logspace(-6, 2, n)[:, None],
             torch.randn(n, 128).clamp(-2, 2) * 1e-5,
             torch.zeros(n, 128)]
    ties = torch.tensor([6., 0., -0., .25, -.25, .75, -.75, 1.25, -1.25, 1.75, -1.75, 2.5, -2.5, 3.5, -3.5, 5.])
    parts.append(ties.repeat(8).repeat(n, 1) * torch.logspace(-3, 1, n)[:, None])
    return torch.cat(parts).bfloat16()


# 1) Writer bytes at position 0 (identity RoPE) against the oracle; never worse than /6 or 4/6.
table = rope_table(4096).to(device)
keys = keys_mix(96)
rows = keys.shape[0]
pages = (rows + 63) // 64 + 2
cache = torch.full((pages, 64, 72), 0xA5, dtype=torch.uint8, device=device)
slots = torch.randperm(pages * 64)[:rows].to(device)
nv.store(keys.to(device), torch.zeros(rows, dtype=torch.int64, device=device), table, cache, slots, compress_ratio=1)
values, scales = read_rows(cache, slots.cpu())
rv, rs = nv.reference_quantize(keys)
sse = lambda p, s: (nv.reference_dequantize(p, s).double() - keys.double()).square().reshape(-1, 8, 16).sum(-1)
ours = sse(values, scales)
r6, r4 = nv.reference_quantize(keys, 'div6'), nv.reference_quantize(keys, 'four_over_six')
untouched = torch.ones(pages * 64, dtype=torch.bool)
untouched[slots.cpu()] = False
report['writer_search'] = dict(
    rows=rows, value_mismatch=int((values != rv).sum()), scale_mismatch=int((scales != rs).sum()),
    groups_worse_than_div6=int((ours > sse(*r6)).sum()), groups_worse_than_4over6=int((ours > sse(*r4)).sum()),
    groups_changed_from_div6=int((scales != r6[1]).sum()),
    sse_vs_div6=float(ours.sum() / sse(*r6).sum()), sse_vs_4over6=float(ours.sum() / sse(*r4).sum()),
    foreign_rows_intact=bool((cache.cpu().reshape(pages, -1)[:, :64 * 64].reshape(pages * 64, 64)[untouched] == 0xA5).all()))

# 2) RoPE positions and 3) compress ratio 2 with skipped slots.
for ratio in (1, 2):
    n = 512
    k = (torch.randn(n, 128) * 2).bfloat16()
    positions = torch.randint(0, 4096, (n,))
    slots = torch.randperm(16 * 64)[:n]
    slots[::7] = -1
    cache = torch.full((16, 64, 72), 0x5A, dtype=torch.uint8, device=device)
    nv.store(k.to(device), positions.to(device), table, cache, slots.to(device), compress_ratio=ratio)
    writes = (slots >= 0) & ((positions + 1) % ratio == 0)
    values, scales = read_rows(cache, slots[writes])
    rv, rs = nv.reference_quantize(reference_rope(k[writes], positions[writes], table, ratio))
    skipped = slots[(slots >= 0) & ~writes]
    sv, _ = read_rows(cache, skipped) if len(skipped) else (torch.full((1, 64), 0x5A, dtype=torch.uint8), None)
    report[f'rope_ratio{ratio}'] = dict(written=int(writes.sum()), value_mismatch=int((values != rv).sum()),
        scale_mismatch=int((scales != rs).sum()), skipped_rows_untouched=bool((sv == 0x5A).all()))

# 4) Request gather.
cache = torch.randint(0, 256, (32, 64, 72), dtype=torch.uint8, device=device)
lens = [70, 1, 200, 0, 129]
cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32)
table_rows = torch.stack([torch.randperm(32)[:5] for _ in lens]).to(torch.int32)
total = int(cu[-1])
values = torch.empty((total + 13, 64), dtype=torch.uint8, device=device)
scales = torch.empty((total + 13, 8), dtype=torch.uint8, device=device)
nv.gather_requests(cache, values[:total], scales[:total], table_rows.to(device), cu.to(device))
ok = True
flat = cache.cpu().reshape(32, -1)
for r, n in enumerate(lens):
    for j in range(n):
        page, row = int(table_rows[r, j // 64]), j % 64
        t = int(cu[r]) + j
        ok &= bool((values[t].cpu() == flat[page, row * 64:row * 64 + 64]).all())
        ok &= bool((scales[t].cpu() == flat[page, 64 * 64 + row * 8:64 * 64 + row * 8 + 8]).all())
report['gather'] = dict(keys=total, exact=ok)

# 5) Logits against float64.
m, n = 37, 3000
q = (torch.randn(m, 32, 128) * 2).to(torch.float8_e4m3fn)
w = torch.randn(m, 32) * 0.05
kp, ks = nv.reference_quantize((torch.randn(n, 128) * 1.5).bfloat16())
starts = torch.randint(0, 500, (m,), dtype=torch.int32)
ends = (starts + torch.randint(0, 2600, (m,), dtype=torch.int32)).clamp(max=n)
got = nv.mqa_logits(q.to(device), w.to(device), kp.to(device), ks.to(device), starts.to(device), ends.to(device)).cpu()
want = nv.reference_logits(q, w, kp, ks, starts, ends)
finite = torch.isfinite(want)
err = (got[finite] - want[finite]).abs()
scale = want[finite].abs().clamp_min(1e-3)
report['logits'] = dict(rows=m, keys=n, masks_equal=bool((torch.isfinite(got) == finite).all()),
    max_abs_error=float(err.max()), max_rel_error=float((err / scale).max()),
    native_adapter_equal=bool(torch.equal(nv.mqa_logits_native_signature(
        (q.to(device).view(torch.int8), None), (kp.to(device).view(torch.int8), ks.to(device).view(torch.int32)),
        w.to(device), starts.to(device), ends.to(device)).cpu(), got)))

# 6) Timing: decode-like (1 row x 64K keys) and prefill-like causal tile (256 x 32K).
def bench(m, n, causal):
    q = (torch.randn(m, 32, 128, device=device)).to(torch.float8_e4m3fn)
    w = torch.randn(m, 32, device=device) * 0.05
    kp = torch.randint(0, 256, (n, 64), dtype=torch.uint8, device=device)
    ks = torch.full((n, 8), 0x38, dtype=torch.uint8, device=device)
    starts = torch.zeros(m, dtype=torch.int32, device=device)
    ends = (torch.linspace(n - m + 1, n, m, device=device).int() if causal
            else torch.full((m,), n, dtype=torch.int32, device=device))
    out = torch.empty((m, n), device=device)
    for _ in range(3):
        nv.mqa_logits(q, w, kp, ks, starts, ends, out)
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(10):
        nv.mqa_logits(q, w, kp, ks, starts, ends, out)
    end.record()
    torch.cuda.synchronize()
    ms = begin.elapsed_time(end) / 10
    useful = float((ends - starts).sum()) * 32 * 128 * 2
    return dict(ms=round(ms, 3), tflops_useful=round(useful / ms / 1e9, 2))

report['timing'] = dict(decode_1x65536=bench(1, 65536, False), prefill_256x32768=bench(256, 32768, True))
report['peak_allocated_mib'] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
print(json.dumps(report, indent=1))
