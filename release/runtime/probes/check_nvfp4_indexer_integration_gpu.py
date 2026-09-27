"""Integration checks for NVFP4 index keys through the real vLLM entry points (no engine).

Requires the opt-in environment (full-FP4 DCP runtime plus
DS41_INDEXER_K_FORMAT=nvfp4). Writes keys with vLLM's
indexer_k_norm_rope_store (dispatched by spark_indexer_k_math's parity hook
to ds41.nvfp4_indexer.store) into 72-byte pages, compares bytes against native
RMSNorm + float64 RoPE + the nearest-code oracle, then scores those pages with
the eager and graph decode scorers against a float64 reference.
"""
import json

import torch

import spark_indexer_k_math as parity
from ds41 import nvfp4_indexer as nv

parity.register()
from vllm import _custom_ops as ops  # noqa: E402
from vllm.models.deepseek_v4_1.common.ops import indexer_k_store as native  # noqa: E402

assert nv.ENABLED and parity._nvfp4 is nv
torch.manual_seed(1)
device = 'cuda'
report = {}


def rope_table(positions, dim=64, base=160000.0):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    angles = torch.outer(torch.arange(positions, dtype=torch.float64), freqs)
    return torch.cat((angles.cos(), angles.sin()), -1).float()


def reference_rope(keys, positions, table, ratio):
    x = keys.cpu().double()
    even, odd = x[:, 0::2].clone(), x[:, 1::2].clone()
    p = (positions.cpu() // ratio) * ratio
    cos, sin = table.cpu().double()[p][:, :32], table.cpu().double()[p][:, 32:]
    ne = (-odd[:, 32:] * sin + (even[:, 32:] * cos).float().double()).float().bfloat16()
    no = (even[:, 32:] * sin + (odd[:, 32:] * cos).float().double()).float().bfloat16()
    even, odd = even.bfloat16(), odd.bfloat16()
    even[:, 32:], odd[:, 32:] = ne, no
    return torch.stack((even, odd), -1).reshape(-1, 128)


def page_rows(cache, slots):
    states = cache.shape[1]
    raw = cache.cpu().reshape(cache.shape[0], -1)
    v = torch.stack([raw[s // states, (s % states) * 64:(s % states) * 64 + 64] for s in slots.tolist()])
    s_ = torch.stack([raw[s // states, states * 64 + (s % states) * 8:states * 64 + (s % states) * 8 + 8]
                      for s in slots.tolist()])
    return v, s_


table = rope_table(8192).to(device)
for ratio in (1, 2):
    n = 700
    k_pre = (torch.randn(n, 128) * 3).bfloat16().to(device)
    weight = (torch.randn(128) * 0.2 + 1).bfloat16().to(device)
    positions = torch.randint(0, 8192, (n,), dtype=torch.int64, device=device)
    slots = torch.randperm(16 * 64, device=device)[:n].to(torch.int64)
    slots[::11] = -1
    cache = torch.zeros((16, 64, 72), dtype=torch.uint8, device=device)
    native.indexer_k_norm_rope_store(k_pre, positions, table, weight, 1e-6, cache, slots, ratio, True)
    normalized = torch.empty_like(k_pre)
    ops.rms_norm(normalized, k_pre, weight, 1e-6)
    writes = ((slots >= 0) & ((positions + 1) % ratio == 0)).cpu()
    got_v, got_s = page_rows(cache, slots.cpu()[writes])
    ref_v, ref_s = nv.reference_quantize(reference_rope(normalized.cpu()[writes], positions.cpu()[writes], table, ratio))
    report[f'vllm_writer_ratio{ratio}'] = dict(written=int(writes.sum()),
        value_mismatch=int((got_v != ref_v).sum()), scale_mismatch=int((got_s != ref_s).sum()))

# Pages for two requests assigned through a block table, then both decode scorers.
states, pages = 64, 24
keys = (torch.randn(pages * states, 128) * 2).bfloat16()
packed, scales = nv.reference_quantize(keys)
cache = torch.zeros((pages, states, 72), dtype=torch.uint8)
raw = cache.view(pages, -1)
for s in range(pages * states):
    p, r = divmod(s, states)
    raw[p, r * 64:r * 64 + 64] = packed[s]
    raw[p, states * 64 + r * 8:states * 64 + r * 8 + 8] = scales[s]
cache = cache.to(device)
table_rows = torch.stack([torch.randperm(pages)[:10], torch.randperm(pages)[:10]]).to(torch.int32)
lengths = [517, 333]
max_model_len = 1024
span = table_rows.shape[1] * states  # keys reachable through the table


def request_keys(r):
    order = [int(table_rows[r, j // states]) * states + j % states for j in range(span)]
    return packed[order], scales[order]


view = nv.quant_view(cache, 128, True)
q = (torch.randn(2, 1, 32, 128) * 2).to(torch.float8_e4m3fn)
w = torch.randn(2, 32) * 0.05
eager = nv.paged_logits((q.to(device).view(torch.int8), None), view, w.to(device),
                        torch.tensor(lengths, dtype=torch.int32, device=device)[:, None],
                        table_rows.to(device), None, max_model_len=max_model_len).cpu()
next_n = 3
qg = (torch.randn(2, next_n, 32, 128) * 2).to(torch.float8_e4m3fn)
wg = torch.randn(2 * next_n, 32) * 0.05
row_lengths = torch.tensor([[515, 516, 517], [331, 332, 333]], dtype=torch.int32)
graph = nv.graph_paged_logits((qg.to(device).view(torch.int8), None), view, wg.to(device),
                              row_lengths.to(device), table_rows.to(device), None,
                              max_model_len=max_model_len).cpu()
errors = dict(eager=0.0, graph=0.0)
masks = dict(eager=True, graph=True)
for r in range(2):
    kp, ks = request_keys(r)
    zero = torch.zeros(1, dtype=torch.int32)
    want = nv.reference_logits(q[r, 0:1], w[r:r + 1], kp, ks, zero, torch.tensor([lengths[r]], dtype=torch.int32))
    got = eager[r:r + 1]
    masks['eager'] &= bool((torch.isfinite(got[:, :span]) == torch.isfinite(want)).all()
                           and torch.isneginf(got[:, span:]).all())
    f = torch.isfinite(want)
    errors['eager'] = max(errors['eager'], float((got[:, :span][f] - want[f]).abs().max()))
    want = nv.reference_logits(qg[r], wg[r * next_n:(r + 1) * next_n], kp, ks, torch.zeros(next_n, dtype=torch.int32),
                               row_lengths[r])
    got = graph[r * next_n:(r + 1) * next_n]
    masks['graph'] &= bool((torch.isfinite(got[:, :span]) == torch.isfinite(want)).all()
                           and torch.isneginf(got[:, span:]).all())
    f = torch.isfinite(want)
    errors['graph'] = max(errors['graph'], float((got[:, :span][f] - want[f]).abs().max()))
report['decode_scorers'] = dict(masks_equal=masks, max_abs_error=errors)
print(json.dumps(report, indent=1))
