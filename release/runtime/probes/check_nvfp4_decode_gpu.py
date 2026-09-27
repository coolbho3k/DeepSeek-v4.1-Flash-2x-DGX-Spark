"""GPU checks for NVFP4 decode queries (DS41_INDEXER_DECODE_QUERY=nvfp4), no engine.

Run with the full opt-in environment:
  DS41_ENABLE_DCP2=1 DS41_ENABLE_FP4_MAIN_KV=1 DS41_ENABLE_FP4_INDEXER=1 DS41_ENABLE_INDEXER_K_PARITY=1
  DS41_INDEXER_K_FORMAT=nvfp4 DS41_INDEXER_DECODE_QUERY=nvfp4
Checks the key-split FP4 kernel (clean and unclean contracts), the eager and capture-safe
NVFP4 decode scorers against float64 over block-table pages, the decode dispatcher
(NVFP4 route and FP8 fallbacks), a real CUDA-graph capture/replay under GraphOwner with
data changed between replays, and decode scoring time (gather included) against the FP8
query scorer and the MXFP4 DeepGEMM route.
"""
import json

import torch

from ds41 import dcp_indexer_graph as mx_graph
from ds41 import nvfp4_indexer as nv
from ds41.graph_validation import GraphOwner

assert nv.ENABLED and nv.DECODE_QUERY == 'nvfp4'
torch.manual_seed(3)
device = 'cuda'
report = {}
ss, hs = 128 ** -0.5, 32 ** -0.5


def rope_table(positions, dim=64, base=160000.0):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    angles = torch.outer(torch.arange(positions, dtype=torch.float64), freqs)
    return torch.cat((angles.cos(), angles.sin()), -1).float()


# 1) Key-split kernel, few rows x many keys, both contracts.
for rows, n in ((1, 50000), (4, 70001), (24, 20000)):
    qp, qs, hw = nv.reference_queries((torch.randn(rows, 32, 128) * 3).bfloat16(), torch.randn(rows, 32), ss, hs)
    kp, ks = nv.reference_quantize(torch.randn(n, 128).bfloat16())
    starts = torch.zeros(rows, dtype=torch.int32)
    ends = torch.randint(n // 3, n + 1, (rows,), dtype=torch.int32)
    want = nv.reference_nvfp4_logits(qp, qs, hw, kp, ks, starts, ends)
    args = [t.contiguous().to(device) for t in (qp, qs, hw, kp, ks, starts, ends)]
    clean = nv.nvfp4_logits(*args, clean=True).cpu()
    loose = nv.nvfp4_logits(*args).cpu()
    live = torch.isfinite(want)
    report[f'split_{rows}x{n}'] = dict(split=nv.key_split(rows, n),
        clean_masks_equal=bool((torch.isfinite(clean) == live).all()),
        clean_max_abs_error=float((clean[live] - want[live]).abs().max()),
        unclean_in_range_max_abs_error=float((loose[live] - want[live]).abs().max()))

# 2) Decode scorers over block-table pages with NVFP4 queries from a QueryPackage.
states, pages, next_n, batch = 64, 40, 3, 2
keys = (torch.randn(pages * states, 128) * 2).bfloat16()
packed, scales = nv.reference_quantize(keys)
cache = torch.zeros((pages, states, 72), dtype=torch.uint8)
raw = cache.view(pages, -1)
for s in range(pages * states):
    p, r = divmod(s, states)
    raw[p, r * 64:r * 64 + 64] = packed[s]
    raw[p, states * 64 + r * 8:states * 64 + r * 8 + 8] = scales[s]
cache = cache.to(device)
view = nv.quant_view(cache, 128, True)
table = torch.stack([torch.randperm(pages)[:16] for _ in range(batch)]).to(torch.int32)
span = table.shape[1] * states
max_model_len = 1536
lengths = torch.tensor([[700, 701, 702], [1000, 1001, 1002]], dtype=torch.int32)
table_rope = rope_table(4096).to(device)
q_pre = (torch.randn(batch * next_n, 32, 128) * 3).bfloat16()
positions = torch.randint(0, 4096, (batch * next_n,))
wts = (torch.randn(batch * next_n, 32) * 0.5).bfloat16()
package = nv.QueryPackage(q_pre.to(device), positions.to(device), table_rope, wts.to(device), ss, hs)
roped = nv.reference_query_rope(q_pre, positions, table_rope)
rqv, rqs, rqw = nv.reference_queries(roped, wts, ss, hs)


def reference_rows():
    out = []
    for r in range(batch):
        order = [int(table[r, j // states]) * states + j % states for j in range(span)]
        rows = slice(r * next_n, (r + 1) * next_n)
        out.append(nv.reference_nvfp4_logits(rqv[rows], rqs[rows], rqw[rows], packed[order], scales[order],
                                             torch.zeros(next_n, dtype=torch.int32), lengths[r]))
    return torch.cat(out)


def compare(got, want):
    live = torch.isfinite(want)
    return dict(masks_equal=bool((torch.isfinite(got[:, :span]) == live).all() and torch.isneginf(got[:, span:]).all()),
                max_abs_error=float((got[:, :span][live] - want[live]).abs().max()))


want = reference_rows()
values, qscales, qweights = nv._decode_query_layout(*package.quantize(0, batch * next_n), batch, next_n)
eager = nv.paged_logits_nvfp4(values, qscales, qweights, view, lengths.to(device), table.to(device),
                              max_model_len=max_model_len).cpu()
graph = nv.graph_paged_logits_nvfp4(values, qscales, qweights, view, lengths.to(device), table.to(device),
                                    max_model_len=max_model_len).cpu()
report['decode_scorers'] = dict(eager=compare(eager, want), graph=compare(graph, want))

# 3) Dispatcher: NVFP4 route, and FP8 fallbacks (no package / ragged padding).
fp8_q = torch.randn(batch, next_n, 32, 128, device=device).to(torch.float8_e4m3fn).view(torch.int8)
fp8_w = torch.randn(batch * next_n, 32, device=device) * 0.05
dispatch = nv.make_decode_logits(nv.graph_paged_logits)
routed = dispatch(package, batch * next_n, False, (fp8_q, None), view, fp8_w, lengths.to(device), table.to(device),
                  None, max_model_len=max_model_len).cpu()
fallback = dispatch(None, batch * next_n, False, (fp8_q, None), view, fp8_w, lengths.to(device), table.to(device),
                    None, max_model_len=max_model_len).cpu()
padded = dispatch(package, batch * next_n, True, (fp8_q, None), view, fp8_w, lengths.to(device), table.to(device),
                  None, max_model_len=max_model_len).cpu()
fp8_ref = nv.graph_paged_logits((fp8_q, None), view, fp8_w, lengths.to(device), table.to(device), None,
                                max_model_len=max_model_len).cpu()
report['dispatcher'] = dict(nvfp4_route_equals_graph_scorer=bool(torch.equal(routed, graph)),
                            no_package_uses_fp8=bool(torch.equal(fallback, fp8_ref)),
                            ragged_uses_fp8=bool(torch.equal(padded, fp8_ref)))

# 4) CUDA-graph capture under GraphOwner, replayed after changing pages and lengths in place.
static_lengths = lengths.to(device).clone()
static_table = table.to(device)
owner = GraphOwner(torch.device('cuda', torch.cuda.current_device()))
g = torch.cuda.CUDAGraph()
with owner.execution(capture_only=True):
    with torch.cuda.graph(g):
        captured = dispatch(package, batch * next_n, False, (fp8_q, None), view, fp8_w, static_lengths,
                            static_table, None, max_model_len=max_model_len)
keys2 = (torch.randn(pages * states, 128) * 2).bfloat16()
packed, scales = nv.reference_quantize(keys2)
raw = torch.zeros((pages, states, 72), dtype=torch.uint8).view(pages, -1)
for s in range(pages * states):
    p, r = divmod(s, states)
    raw[p, r * 64:r * 64 + 64] = packed[s]
    raw[p, states * 64 + r * 8:states * 64 + r * 8 + 8] = scales[s]
cache.copy_(raw.view(pages, states, 72).to(device))
lengths = torch.tensor([[400, 401, 402], [1020, 1021, 1022]], dtype=torch.int32)
static_lengths.copy_(lengths.to(device))
with owner.execution(capture_only=False):
    g.replay()
torch.cuda.synchronize()
report['graph_capture_replay'] = compare(captured.cpu(), reference_rows())


# 5) Decode scoring time, gather included: FP8 queries vs NVFP4 queries vs MXFP4/DeepGEMM.
def timeit(f, reps=10):
    f(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps):
        f()
    b.record(); torch.cuda.synchronize()
    return round(a.elapsed_time(b) / reps, 3)


timing = {}
for requests, context in ((1, 65536), (1, 524288), (6, 131072), (6, 524288)):
    n4 = 4
    pages_needed = -(-context // 128)
    cache_nv = torch.randint(0, 256, (pages_needed, 128, 72), dtype=torch.uint8, device=device)
    cache_nv.view(pages_needed, -1)[:, 128 * 64:] = 0x38
    cache_mx = torch.randint(0, 256, (pages_needed, 128, 1, 68), dtype=torch.uint8, device=device)
    tbl = torch.arange(pages_needed, dtype=torch.int32, device=device).repeat(requests, 1)
    lens = torch.full((requests, n4), context, dtype=torch.int32, device=device)
    q8 = torch.randn(requests, n4, 32, 128, device=device).to(torch.float8_e4m3fn).view(torch.int8)
    w8 = torch.randn(requests * n4, 32, device=device) * 0.05
    pkg = nv.QueryPackage((torch.randn(requests * n4, 32, 128, device=device) * 3).bfloat16(),
                          torch.arange(requests * n4, device=device), table_rope,
                          torch.randn(requests * n4, 32, device=device).bfloat16(), ss, hs)
    qmx = torch.randint(0, 256, (requests, n4, 32, 64), dtype=torch.uint8, device=device).view(torch.int8)
    smx = torch.full((requests, n4, 32, 4), 127, dtype=torch.uint8, device=device).view(torch.int32).squeeze(-1)
    view_nv = nv.quant_view(cache_nv, 128, True)
    timing[f'{requests}x{n4}_rows_{context}_keys'] = dict(
        fp8_queries=timeit(lambda: nv.graph_paged_logits((q8, None), view_nv, w8, lens, tbl, None, max_model_len=context)),
        nvfp4_queries=timeit(lambda: dispatch(pkg, requests * n4, False, (q8, None), view_nv, w8, lens, tbl, None,
                                              max_model_len=context)),
        mxfp4_deepgemm=timeit(lambda: mx_graph.paged_logits((qmx, smx), cache_mx, w8, lens, tbl, None, max_model_len=context)))
report['timing_ms'] = timing
report['peak_allocated_mib'] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
print(json.dumps(report, indent=1))
