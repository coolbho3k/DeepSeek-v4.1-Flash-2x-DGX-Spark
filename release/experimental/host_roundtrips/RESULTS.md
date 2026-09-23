# Decode campaign, 2026-09-23: items 1–4

All serving measurements used port 8889, the saved T=1 probabilistic-K3
baseline (`reports/block-verification-v1/baseline-{serial,c6}.json`), and
unchanged weights, KV formats, display KV, C6 and memory limits.

## Selected: vocab row cache + deferred full-graph validation

Local kit `1c8840e0…a678` (`prepare_kit.py` then `prepare_deferred_kit.py`)
now serves on port 8888. Against the saved baseline, all four serial cases
improved +3.9…+4.1% (median **+4.05%**) with acceptance identical to the
vocab-cache-only run; C6 first use 47.30 vs 45.67, warm 53.25 / 52.68 vs 53.47
(flat). English/Chinese smoke replies were correct.

Deferred validation: every captured full graph (target and draft, both TP
ranks, captured unconditionally so collective order matches) ends with an
all-reduce of per-rank nonzero error-flag counts. Replay enqueues only
non-blocking copies of flags and summary into pinned host buffers. Pending
checks drain in `AsyncOutput.get_output` (before sampled tokens leave the
worker) and at the start of every graph execution; a peer's errors therefore
block rank 0's output. Codes, messages, masking and poisoning are unchanged;
piecewise graphs and eager checks remain synchronous. Single-GPU check
(`check_deferred_validation.py`): clean replays pass; own-rank and peer
errors raise and poison; undrained replay is refused.

The public `release/runtime` tree and `recipe-lock.json` are NOT updated yet;
`prepare_kit.py --parent-kit release/runtime` followed by
`prepare_deferred_kit.py` reproduces the change for the public recipe.

## 1. Host round-trips in the decode step — partial win, not promoted

Profiling showed the SSD-offloaded input embedding costs a blocking host
callback per step: 4 serial `O_DIRECT` reads for the verify rows and 3 for the
draft (0.4–1.6 ms GPU idle each). Measured SSD latency under load: p50 89 µs,
mean 113 µs, p99 583 µs per 12 KiB read.

`ds41_vocab_row_store.cpp` adds an exact-byte 4096-slot direct-mapped row
cache (40 MiB/rank, filled only from verified reads, file identity still
checked before and after every lookup) and parallel reads for ≥2 misses,
waking only as many workers as needed. ABI unchanged. CPU parity passed
(duplicates, slot conflicts, TP masks, image/dead IDs, mutation after warm-up).
Cold 4-row lookup 300 → ~165 µs; hits ~4–10 µs.

| Metric | Saved baseline | Vocab cache |
|---|---:|---:|
| Serial decode, median ratio | 1.000 | **1.0096** (all four +0.5…+1.5%) |
| C6 first use | 45.67 | 46.47 |
| C6 warm | 53.47 | 53.18 / 53.26 |

Trace: verify-embedding stall 1.37–1.46 → 0.54 ms. Kit
`a079a82e…60b8`, evidence `reports/host-roundtrips-v1/`.

Located but NOT changed (safety-contract decision needed):

- `graph_validation.GraphOwner.execution` reads 1,256 bytes of graph error
  flags synchronously after the target graph (0.3–1.4 ms GPU idle before
  the LM head); the draft graph does the same with 16 bytes. Deferring to
  `AsyncOutput.get_output` keeps rank 0 "checked before output", but rank 1
  never calls `get_output`, so a correct version needs a cross-rank flag
  reduction inside the graph.
- `dcp_metadata.compressed_slot_mapping` performs ~5 validation syncs per
  call (~10 per step); they can be folded into one.
- The eager preamble before the first target layer is ~5 ms with only
  ~1.7 ms of GPU work (CPU-bound metadata building, ~350 eager kernels).
  Async scheduling is configured (batch queue 2) but consecutive steps did
  not overlap in the 3-iteration trace; finding why is the largest remaining
  decode lever in this area.

## 2. Small-M MXFP8 dense GEMMs — rejected

Real per-rank shapes: wq_a+wkv [1792,5120], wq_b [16384,1280], wo_b
[5120,4096], shared gate/up [2304,5120], shared down [5120,1152]. Streaming
from DRAM (L2-defeating copies, CUDA graphs), native B12X already reaches
186–226 GB/s at 1–24 rows. A Triton split-K FP8-MMA candidate
(`../dense_skinny/`) was numerically near-identical (NMSE ≤ 1e-8) but only
~130 GB/s (1.3–1.8× slower). Upper bound of any kernel win ≈ 2.5 ms/step;
larger in-model durations reflect overlap/contention, not kernel efficiency.
Evidence `reports/dense-skinny-v1/bench-v1.json`.

## 3. Frequency-ranked draft vocabulary — rejected

`../draft_vocab/` projects the shared head and Markov head onto a per-rank
token subset (row-gather GEMV, bit-identical to torch, 235–240 GB/s) and uses
native DSpark's draft→target scatter, so target distribution is unchanged.
Component saving at 16K/rank: ~2.65 ms/step. But calibration-corpus tokens
covered only 81.3% of actually generated tokens; acceptance fell 15–25%
relative, serial median −2.1%, warm C6 −5%. A disjoint 14K-token chat corpus
raised held-out coverage only to 83% (16K/rank) / 92.6% (32K/rank): model
outputs have a long token tail. Output quality was not affected (rejection
sampling); only speed. Evidence `reports/draft-vocab-v1/`.

## 4. NCCL small messages — no configuration gain

Cross-rank pairing of trace collectives: true all-reduce cost ≈21 µs median
(2.4–2.7 ms/step for 88 calls); larger per-rank totals are peer waiting.
All-gather ≈28–31 µs (≈3.2 ms/step, 128 calls). A two-Spark in-graph
microbenchmark (`../nccl_small/`) of the serving NCCL environment versus 1/2
channels, single rail, and forced LL found every variant within baseline
run-to-run noise (~30%). Remaining levers are fewer collectives and less
rank skew, not NCCL tuning. Evidence `reports/nccl-small-v1/results.jsonl`.
