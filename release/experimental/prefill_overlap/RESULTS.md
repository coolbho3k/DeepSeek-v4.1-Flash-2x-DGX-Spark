# Two-micro-batch prefill overlap: feasibility — not viable on GB10 (2026-09-23)

Motivation: in a 2048-token prefill chunk ~460–500 ms of 1.95 s is exposed
communication (GPU otherwise idle); the concurrent-DCP path hides ~115 ms.

`overlap_bench.py` (two Sparks, serving NCCL environment, `run.sh`) runs the
real prefill collective shapes (4 × 33.6 MB FP32 head-exchange all-gathers +
one 2046×5120 BF16 all-reduce, ≈7.5 ms) on one stream and, on another, either
MoE-like memory-bound skinny GEMMs over 1.5 GB of weights (≈7 ms) or a
compute-bound GEMM (≈13 ms). "Efficiency" is the fraction of the shorter job
hidden when both run together.

| NCCL variant | Memory-bound efficiency | Compute-bound efficiency |
|---|---:|---:|
| Serving defaults (two runs) | 0.23 / 0.19 | 0.40 / 0.29 |
| 4 channels | 0.19 | 0.43 |
| 2 channels (± 2 CTAs) | 0.16 / 0.14 | 0.41 / 0.40 |
| 1 channel, 1 CTA | 0.23 | 0.50 |
| Simple protocol | 0.24 | 0.40 |
| Simple, 4 MiB buffers | 0.34 | 0.42 |
| Simple, 2 channels | 0.23 | 0.46 |

Communication and GPU work largely contend on this unified-memory platform
regardless of channel, CTA, protocol or buffer settings.

Consequences for DBO-style micro-batching of prefill:

1. At most ~20–40% of the exposed communication could be hidden
   (≈100–190 ms per chunk).
2. Splitting a 2048-token chunk into two 1024-token micro-batches makes each
   half stream essentially all 384 experts' weights; the grouped MoE is
   already near its ≈390 ms/chunk weight-streaming floor, so this adds up to
   ≈390 ms — more than the best-case saving.

Overlap that keeps the MoE unsplit (attention slices versus the exchange) is
what the existing concurrent-DCP path already does. Not pursued further; the
remaining communication lever is fewer bytes (e.g. BF16 exchange payload,
a numerical change requiring approval).
