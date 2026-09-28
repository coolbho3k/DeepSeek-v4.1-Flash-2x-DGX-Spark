"""GPU checks for the paged-direct NVFP4 decode scorers (FP8 queries), no engine.

Run with the NVFP4 index-key environment (FP8 decode queries, the default):
  DS41_ENABLE_DCP2=1 DS41_ENABLE_FP4_MAIN_KV=1 DS41_ENABLE_FP4_INDEXER=1 DS41_ENABLE_INDEXER_K_PARITY=1
  DS41_INDEXER_K_FORMAT=nvfp4
Checks, bitwise:
  1) graph_paged_logits (keys read in place from pages) against the previous workspace-gather
     scorer, over ragged per-row lengths, zero and full-capacity rows, permuted page tables,
     64- and 128-state pages, and the same error reports for invalid pages and lengths;
  2) graph_candidate_logits followed by apply_candidate_mask against graph_paged_logits
     followed by the same mask, on both DCP ranks, with -1 padding and out-of-range blocks,
     directly and through the decode dispatcher;
  3) a CUDA-graph capture of both scorers, replayed after changing pages, lengths and
     candidates in place;
and times one decode indexer call under CUDA-graph replay (gathered, paged, candidates).
"""
import json
from types import SimpleNamespace

import torch

from ds41 import dcp_candidates_graph as candidates_graph
from ds41 import nvfp4_indexer as nv
from ds41.graph_validation import GraphOwner

assert nv.ENABLED and nv.DECODE_QUERY == 'fp8'
torch.manual_seed(7)
device = 'cuda'
report = {}


def new_owner():
    # One owner per captured graph (an owner records exactly one capture).
    return GraphOwner(torch.device('cuda', torch.cuda.current_device()))


def random_cache(pages, states):
    cache = torch.randint(0, 256, (pages, states, 72), dtype=torch.uint8)
    flat = cache.view(pages, -1)
    # Finite E4M3 scales (no 0x7F/0xFF NaN codes) of both signs' magnitude range.
    flat[:, states * 64:] = torch.randint(0x08, 0x5F, (pages, states * 8), dtype=torch.uint8)
    return cache.to(device)


def fp8_queries(batch, next_n):
    return (torch.randn(batch, next_n, 32, 128, device=device) * 2).to(torch.float8_e4m3fn).view(torch.int8)


def bits_equal(a, b):
    return a.shape == b.shape and bool(torch.equal(a.view(torch.int32), b.view(torch.int32)))


def outcome(f):
    try:
        return 'ok', f()
    except ValueError as error:
        return str(error), None


def mask(logits, lengths, candidates, rank):
    out = logits.clone()
    candidates_graph.apply_candidate_mask(out, None, lengths.reshape(-1).contiguous(), candidates, 8, 2, rank)
    return out


# 1) Paged-direct scorer == workspace-gather scorer, bitwise.
cases = []
for states, batch, next_n, cap in ((64, 1, 4, 4096), (128, 2, 3, 6000), (64, 6, 4, 2048), (64, 3, 1, 5000)):
    columns = -(-cap // states) + 2
    pages = batch * columns + 5
    cache = random_cache(pages, states)
    view = nv.quant_view(cache, 128, True)
    table = torch.stack([torch.randperm(pages)[:columns] for _ in range(batch)]).to(torch.int32).to(device)
    lengths = torch.randint(0, cap + 1, (batch, next_n), dtype=torch.int32)
    lengths[0, -1] = cap
    if batch > 1:
        lengths[1] = 0
    lengths = lengths.to(device)
    q = fp8_queries(batch, next_n)
    w = torch.randn(batch * next_n, 32, device=device) * 0.05
    old = nv._graph_paged_logits_gathered((q, None), view, w, lengths, table, None, max_model_len=cap)
    new = nv.graph_paged_logits((q, None), view, w, lengths, table, None, max_model_len=cap)
    bad_table = table.clone()
    bad_table[0, 0] = pages + 3
    errors_old = outcome(lambda: nv._graph_paged_logits_gathered((q, None), view, w, lengths, bad_table, None,
                                                                max_model_len=cap))[0]
    errors_new = outcome(lambda: nv.graph_paged_logits((q, None), view, w, lengths, bad_table, None,
                                                       max_model_len=cap))[0]
    long_lengths = lengths.clone()
    long_lengths[0, 0] = cap + 1
    length_old = outcome(lambda: nv._graph_paged_logits_gathered((q, None), view, w, long_lengths, table, None,
                                                                max_model_len=cap))[0]
    length_new = outcome(lambda: nv.graph_paged_logits((q, None), view, w, long_lengths, table, None,
                                                       max_model_len=cap))[0]
    cases.append(dict(states=states, batch=batch, next_n=next_n, cap=cap, bitwise_equal=bits_equal(old, new),
                      finite_columns=int(torch.isfinite(new).sum()),
                      invalid_page=[errors_old, errors_new], invalid_length=[length_old, length_new],
                      same_errors=errors_old == errors_new != 'ok' and length_old == length_new != 'ok'))
report['paged_equals_gathered'] = cases

# 2) Candidate scorer + mask == full scorer + mask, bitwise, both ranks.
states, batch, next_n, cap, k = 64, 2, 4, 8192, 512
columns = cap // states
pages = batch * columns + 3
cache = random_cache(pages, states)
view = nv.quant_view(cache, 128, True)
table = torch.stack([torch.randperm(pages)[:columns] for _ in range(batch)]).to(torch.int32).to(device)
lengths = torch.tensor([[5000, 5001, 5002, 5003], [8190, 8191, 8192, 8192]], dtype=torch.int32, device=device)
q = fp8_queries(batch, next_n)
w = torch.randn(batch * next_n, 32, device=device) * 0.05
nb = cap * 2 // 8
rows = batch * next_n
candidates = torch.stack([torch.randperm(nb + 64)[:k] for _ in range(rows)]).to(torch.int32)  # some >= nb
candidates[:, -37:] = -1
candidates[0] = -1                                                                             # empty row
candidates = candidates.to(device)
full = nv.graph_paged_logits((q, None), view, w, lengths, table, None, max_model_len=cap)
candidate_cases = {}
for rank in (0, 1):
    group = SimpleNamespace(rank_in_group=rank, world_size=2)
    direct = nv.graph_candidate_logits((q, None), view, w, lengths, table, candidates, block_size=8,
                                       rank=rank, world=2, max_model_len=cap)
    dispatch = nv.make_decode_logits(nv.graph_paged_logits, lambda group=group: group)
    routed = dispatch(None, rows, False, (q, None), view, w, lengths, table, None, max_model_len=cap,
                      candidates=candidates, candidate_block_size=8)
    want = mask(full, lengths, candidates, rank)
    candidate_cases[f'rank{rank}'] = dict(bitwise_equal=bits_equal(mask(direct, lengths, candidates, rank), want),
                                          dispatcher_bitwise_equal=bits_equal(mask(routed, lengths, candidates, rank),
                                                                              want),
                                          kept_columns=int(torch.isfinite(want).sum()))
report['candidates_equal_full_then_mask'] = candidate_cases

# 3) CUDA-graph capture of both scorers; replay after in-place changes.
static = dict(lengths=lengths.clone(), table=table.clone(), candidates=candidates.clone())
graph = torch.cuda.CUDAGraph()
owner = new_owner()
with owner.execution(capture_only=True):
    with torch.cuda.graph(graph):
        captured_full = nv.graph_paged_logits((q, None), view, w, static['lengths'], static['table'], None,
                                              max_model_len=cap)
        captured_candidates = nv.graph_candidate_logits((q, None), view, w, static['lengths'], static['table'],
                                                        static['candidates'], block_size=8, rank=1, world=2,
                                                        max_model_len=cap)
cache.copy_(random_cache(pages, states))
static['lengths'].copy_(torch.tensor([[100, 101, 102, 103], [7000, 7001, 7002, 7003]], dtype=torch.int32))
static['candidates'].copy_(torch.stack([torch.randperm(nb)[:k] for _ in range(rows)]).to(torch.int32))
with owner.execution(capture_only=False):
    graph.replay()
torch.cuda.synchronize()
fresh_full = nv.graph_paged_logits((q, None), view, w, static['lengths'], static['table'], None, max_model_len=cap)
fresh_old = nv._graph_paged_logits_gathered((q, None), view, w, static['lengths'], static['table'], None,
                                            max_model_len=cap)
report['graph_replay'] = dict(
    full_equals_fresh=bits_equal(captured_full, fresh_full),
    full_equals_gathered=bits_equal(captured_full, fresh_old),
    candidates_equal_full_then_mask=bits_equal(mask(captured_candidates, static['lengths'], static['candidates'], 1),
                                               mask(fresh_full, static['lengths'], static['candidates'], 1)))


# 4) One decode indexer call under CUDA-graph replay (no Python or allocation in the timing).
def replay_ms(f, reps=20):
    g = torch.cuda.CUDAGraph()
    owner = new_owner()
    with owner.execution(capture_only=True):
        with torch.cuda.graph(g):
            f()
    with owner.execution(capture_only=False):
        g.replay()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(reps):
            g.replay()
        b.record()
        torch.cuda.synchronize()
    return round(a.elapsed_time(b) / reps, 4)


timing = {}
cap, states = 524288, 64
columns = cap // states
cache = random_cache(columns + 1, states)
view = nv.quant_view(cache, 128, True)
for batch, next_n, context in ((1, 4, 4096), (1, 4, 65536), (1, 4, 520000), (6, 4, 131072)):
    table = torch.arange(columns, dtype=torch.int32, device=device).repeat(batch, 1)
    lengths = torch.full((batch, next_n), context, dtype=torch.int32, device=device)
    q = fp8_queries(batch, next_n)
    w = torch.randn(batch * next_n, 32, device=device) * 0.05
    rows = batch * next_n
    nb = cap * 2 // 8
    live_blocks = max(1, context * 2 // 8)
    candidates = torch.stack([torch.randperm(live_blocks)[:2048] for _ in range(rows)]).to(torch.int32).to(device)
    timing[f'{batch}x{next_n}_rows_{context}_keys'] = dict(
        gathered=replay_ms(lambda: nv._graph_paged_logits_gathered((q, None), view, w, lengths, table, None,
                                                                   max_model_len=cap)),
        paged=replay_ms(lambda: nv.graph_paged_logits((q, None), view, w, lengths, table, None, max_model_len=cap)),
        candidates_2048x8=replay_ms(lambda: nv.graph_candidate_logits((q, None), view, w, lengths, table, candidates,
                                                                      block_size=8, rank=0, world=2,
                                                                      max_model_len=cap)))
report['timing_ms'] = timing
report['peak_allocated_mib'] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
report['passed'] = (all(c['bitwise_equal'] and c['same_errors'] for c in report['paged_equals_gathered'])
                    and all(c['bitwise_equal'] and c['dispatcher_bitwise_equal']
                            for c in report['candidates_equal_full_then_mask'].values())
                    and all(report['graph_replay'].values()))
print(json.dumps(report, indent=1))
