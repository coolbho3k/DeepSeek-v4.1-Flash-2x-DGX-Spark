# fastcomm: two-rank collectives over pinned-host RDMA (2026-09-23)

GB10 has no GPUDirect RDMA, but pinned host memory registers with the NIC
and is coherent for the integrated GPU (see ../rdma_hostmem/). `fastcomm.cu`
stages a message into a pinned send slot, a CPU proxy RDMA-writes it plus an
inline flag into the peer's slot (RC QP, no relaxed ordering), and the peer's
GPU polls the flag then adds or places the data. Kernels are graph-capturable.

## Standalone, two Sparks, in-graph, vs NCCL (single rail, v1)

Bit-exact against NCCL: 0 mismatches over 4,000 eager all-reduce/all-gather
comparisons and 300 graph replays (a two-rank BF16 sum is one commutative add).

| Size | NCCL AR | fastcomm AR | NCCL AG | fastcomm AG |
|---|---:|---:|---:|---:|
| 1×5120 | 31.4 µs | 10.9 µs | 29.3 µs | 9.0 µs |
| 4×5120 | 39.2 µs | 16.6 µs | 34.8 µs | 14.1 µs |
| 12×5120 | 53.3 µs | 28.4 µs | 43.6 µs | 26.1 µs |
| 24×5120 | 69.0 µs | 45.8 µs | 50.1 µs | 43.1 µs |

## Full model (port 8889), candidate kit `27e4ba7d…762a`

Routes BF16 TP all-reduces and DCP query/result all-gathers of ≤ 256 KB
(main stream = channel 0, DCP side stream = channel 1); larger messages and
all of prefill stay on NCCL. Parent: BF16-exchange production kit.

- Teacher-forced prompt log-probabilities (6 × 6,144 tokens) and 12 greedy
  replies: **identical** to the production kit.
- Serial T=1 decode, candidate vs matched control (same kit without fastcomm):
  garden 26.73 vs 25.66, python 31.20 vs 29.60, explanation 33.13 vs 31.86,
  easy prose 28.07 vs 27.35 tok/s — **+4.1% median** (+2.6…+5.4%).
- Warm C6: 53.3 / 55.3 vs 54.4 / 54.3 (flat; C6 messages mostly exceed 256 KB).
- Uncached prefill: 8K 1,221 / 1,223 vs 1,138 / 1,210; 32K 1,198 / 1,246 vs
  1,185 / 1,225 (within noise).

## Dual rail (`fastcomm_dualrail.cu`, rejected)

Splitting each message across both ConnectX ports (two QPs, one flag per rail)
remained bit-exact but was 1–4 µs slower below 256 KB and only tied NCCL at
491 KB (63.6 vs 62.3 µs AR) and 778 KB (81.1 vs 82.2 µs). The second port did
not add usable bandwidth at these sizes, so v1 (single rail, 256 KB) is the
configuration to promote.

## Promoted

Kit `27e4ba7d…762a` serves on port 8888 (fastcomm loaded on both workers).
The public `release/runtime` carries the same fastcomm module, library,
native source and DCP hooks (byte-identical to the served kit); the recipe
lock pins the refreshed manifest. The overlap release test pins the new
`integration.py`/`transport.py` hashes. All 226 repository tests pass.

## Production crash and fix (2026-09-24)

Symptom: both workers died ~30 s after a decode stalled (02:26:44 → 02:27:00),
each with `CUDA error: unspecified launch failure` — fastcomm's intentional
`__trap()` after waiting ~2^36 cycles for peer data.

Cause: one doorbell per channel. The GPU may publish sequence N+1 before the
CPU proxy has consumed N (it only waits for the peer's data, not for its own
send). Under CPU contention (6-CPU worker cgroup, ~100 I/O threads) the proxy
was descheduled, saw N+1 after N, flagged a sequence error and exited
silently; both GPUs then waited forever for data and trapped.

Reproduction: a test-only build injecting random 0–2 ms proxy sleeps. Old
logic: hangs (no result). Fixed logic: 0 mismatches across all sizes and 300
graph replays, no proxy errors.

Fix: one doorbell per slot; the proxy drains each channel strictly in order.
Protocol bound: seq N+2 requires the peer to have received our N, so the GPU
is at most two sequences ahead of the proxy and 4 slots are never overwritten
unread. Proxy errors are now printed to stderr. Library
`23585e6f…113a`; full-model revalidation on 8889: 16/16 benchmark replies
identical to the earlier n-gram run, five C6 waves (50–56 tok/s) with no errors.
Deployed on 8888 (kit `b8399c02…f13c`) and in the public runtime.
