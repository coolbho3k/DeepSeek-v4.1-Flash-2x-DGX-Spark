# Model-specific fusion experiments

Measured 2026-09-22 on the existing TP2/DCP2 DeepSeek V4.1 Flash deployment,
one GB10 Spark per rank. This campaign preserves the weights, speculative
configuration, NVFP4 four-over-six main KV, group-32 FP8/BF16 sliding KV,
1792 MiB display KV per rank, zero ordinary KV allocation, and launch limits.
The selected runtime is serving on port 8888 after repeated comparisons.

## What the fresh profiles showed

The first baseline trace contained 3,067 kernels on rank 0. Packed `wo_a`
accounted for 5.339 ms over 43 GEMV calls; mHC postmix accounted for only
0.348 ms over 89 calls. Target MoE gate/up plus down accounted for 20.657 ms.
The six-request trace used the larger projection path: approximately 9.7 ms
over 40 GEMM calls. These are sums of instrumented kernel durations, not
latency contributions: operations overlap, profiling adds overhead, and NCCL
measurements include waiting for the peer. CUDA graphs were already active.

Earlier local experiments already rejected serial/parallel MoE gate-up fusion,
one/two-block persistent scheduling, and decoded-register pipelining for the
representative routes. Repeating those implementations would not establish a
new optimization. This campaign targets weight reuse and redundant gathers.

## Shared-row packed output projection

The original two-through-four-row GEMV reconstructs the same MXFP8 weight
tile separately for each row. The candidate reconstructs it once in a CTA and
uses it for each row, retaining the original BF16 weight-rounding boundary,
FP32 products and reductions, 16 split-K partials, and final BF16 reduction.
It uses the same temporary allocation sizes and no persistent workspace.
Single-row decode and larger GEMM/reconstruct dispatch remain unchanged.

Both Sparks passed all 24 original cases: real checkpoint weights from layers
0, 19 and 39, one through four rows, contiguous and padded input strides,
three activation amplitudes, poisoned outputs and changed-input graph replay.
The baseline graph was first checked against the installed eager operation.

A second sweep compared eight tile/warp combinations on another 18 cases per
rank. N=8 with eight warps produced occasional last-bit differences and was
rejected. All seven other combinations passed these cases; N=16 / four warps
was selected for the flushed-weight measurements across both ranks.

| Verification rows | Rank 0 candidate/baseline time | Rank 1 candidate/baseline time |
|---|---:|---:|
| 2 | 0.9582 | 0.9745 |
| 3 | 0.9181 | 0.9287 |
| 4 | 0.8422 | 0.8392 |

Values are median ratios over layers and input strides, with 48 MiB of cache
flush traffic before each graph replay. Timing order rotates between variants.
Warm-weight four-row results were much faster (approximately 0.40 ratio),
but are not representative of a whole model streaming its weights.

Evidence: `reports/model-fusion-wo-a-components-v1/` and
`reports/model-fusion-wo-a-tuning-v2/`, separate results for both ranks.
The initial tuning run stopped on its first inexact variant; its successor
records and excludes each inexact variant instead of discarding other results.

## Fused, bounded grouped-prefill gathers

The original path launches two native gathers over the entire scratch-row
capacity. The candidate loads the activation once, computes gate and up
Hadamard transforms, and bounds the launch by the current batch's maximum
routed rows. Actual active-row counts remain on the device. The multiply
still rounds to FP16 before the original FP32 Hadamard, then stores FP16.
Routing, expert GEMMs, output accumulation and workspace sizes are unchanged.

Both Sparks passed 23 cases spanning 1, 8, 24, 128, 512 and 2,048 input rows,
including zero-active and full-active routes, full-buffer byte comparisons,
inactive canaries and changed-input graph replay. Four variants compared the
original pair, a bounded pair, a fused full-grid gather and fused bounded gather.

At 2,048 input / 12,288 active rows, fused bounded gather took 1.370 ms versus
1.422 ms on rank 0 and 1.387 ms versus 1.456 ms on rank 1. For a 128-input batch
with zero fat rows, it reduced empty-grid overhead from 57.1 to 5.86 microseconds
on rank 0. These are component measurements with synthetic routing/scales;
real grouped routes and all other MoE work still require serving measurements.

The native build uses 40 registers, no spills or shared memory, and admits
six resident blocks per SM. Binary SHA-256:
`e8194b01e87e068d4349b1ee7821d6bad1b295ba39281cadaeb42bdbe7b5c67f`.

Evidence: `reports/model-fusion-gather-components-v2/` and the build/source
receipts under `artifacts/model-fusion-dual-gather-{source,build}-v1/`.
The first probe incorrectly retained the pre-capture CUDA stream handle;
its graph replay failed. The corrected probe gets the active stream inside
each invocation. The native binary did not change; failed results are retained.

## Rejected mHC postmix/prenorm fusion

The candidate explicitly rounds the residual to BF16 before the next
projection and squared norm. Nonetheless, its fused arithmetic/layout did not
preserve every native boundary exactly. Across 24 cases on each Spark, the
fused-post variant matched all boundaries in only 8 cases; the unfused-post
variant did so in only 3. Neither is selected, regardless of apparent timing.

Evidence: `reports/model-fusion-mhc-components-v2/`. The first harness attempt
failed before measurement because TileLang's default cache directory was
read-only; the successor provides a dedicated writable cache.

## Serving baseline and candidate

The isolated baseline retained six simultaneous sequences, 2,048-token prefill
chunks, temperature/seed schedules and the independent RAM watchdog. Twelve
serial requests generated 400 tokens each. Two six-request waves generated
256 tokens per request. The 32K prompt was confirmed to have zero prefix hits.

| Baseline workload | Measured result |
|---|---:|
| T=0 garden, median decode | 25.407 tokens/s |
| T=0 Python, median decode | 30.658 tokens/s |
| T=0 explanation, median decode | 30.878 tokens/s |
| T=0 easy prose, median decode | 28.222 tokens/s |
| Six requests, T=0 aggregate end-to-end | 53.400 tokens/s |
| Six requests, T=1 aggregate end-to-end | 51.029 tokens/s |
| Uncached 32,766-token prefill | 1,055.579 tokens/s |

Baseline kit:
`ad6aacc19dffa7ede254dbc93dff455f67c88656c794d4d9e489bce59f6efe57`.
Candidate kit:
`5d23190d052e43d61c2a0b1681a925238248d2cf858209bbc85d5f07e6fa599e`.
The candidate derives from a verified clone, with recursive source-pin updates
and the native gather's source/licensing/build receipt included. The original
kit remains intact. Evidence is under `reports/model-fusion-v1/`.

No end-to-end speedup or broad quality claim is established by the component
results. Small serving differences require a repeated control. The server
must be restored to port 8888 with the best verified configuration.

## First end-to-end comparison

The candidate completed the matched suite and produced identical replies on
all 12 serial cases (including T=1 sampling), with unchanged speculative
acceptance for those requests. Median paired decode throughput improved by
2.44%; median step time fell by 2.38%. The 32,766-token uncached retrieval
passed at 1,057.45 tokens/s versus 1,055.58 tokens/s: effectively unchanged.
Six-request aggregate throughput was 52.618 versus 53.400 tokens/s at T=0,
and 51.411 versus 51.029 at T=1. Scheduling and acceptance differed in these
concurrent waves, so these small differences do not establish a batch gain.
A repeated original-runtime control is in progress.

The C1 trace supports the proposed mechanism: rank-0 projection time fell
from 5.339 ms over 43 old GEMVs to 3.398 ms over 43 shared-row calls. The
candidate C6 profile captured an early step with a smaller admitted batch;
it is not comparable to the baseline's larger GEMM step. Request concurrency
does not by itself prove an individual profiled step used six sequences.
The API concurrency benchmark, not this trace pair, supplies batch throughput.

The candidate reported exactly the baseline cache capacity: 3,313,955 aggregate
KV tokens, 1,879,048,192 display bytes per rank and zero ordinary KV bytes.
This aggregate metric is not a promise of that many tokens in one sequence.

## Repeated original-runtime control and selection

The repeated control produced identical replies to the candidate on all 12
serial cases. Candidate median paired decode throughput was 3.65% higher and
step time 3.53% lower. Against the first baseline the gain was 2.44%, while
the original runtime's own repeat differed by about 0.61% in median throughput.
Both comparisons support a modest decode benefit.

The control measured 52.620 / 51.969 aggregate tokens/s for the T=0 / T=1
six-request waves, compared with the candidate's 52.618 / 51.411. Prefill was
1,062.24 tokens/s compared with 1,057.45. The candidate lies within the two
original runs' observed prefill variation; no prefill or concurrency speedup
is claimed. Synthetic component gains do not substitute for these results.

Selected: N=16 / four-warps shared-row projection for rows 2–4, and fused
batch-bounded gate/up gather. Rejected: both mHC variants and the inexact
N=8 / eight-warps projection geometry. No weight, KV, precision-boundary,
capacity, ordinary-memory budget or vision configuration change was selected.

The public recipe contains the same measured implementation (module comments
updated), with source pins and manifests refreshed. All 226 repository tests
passed. The tested local runtime remains a separately pinned immutable clone;
the public recipe's fresh-clone GPU qualification limits remain unchanged.
The standard local launcher now selects that clone, with rollback to the
original kit retained. Final port-8888 validation is in progress.

## Final restored server

The final restart retained the same candidate kit and port 8888. All twelve
serial replies matched both original-runtime runs and the first candidate run.
Median paired decode throughput was 1.04% higher than the first baseline and
2.45% higher than the repeated control. Averaging the two runs per configuration
for each matching prompt, then taking the median ratio across twelve prompts,
gives **2.44% higher decode throughput**. This is a descriptive
measurement, not a statistical confidence interval or a quality benchmark.

Final six-request throughput was 51.919 / 51.791 tokens/s at T=0 / T=1.
The final 32K retrieval returned all three keys correctly. Other API traffic
was present when the harness checked idle counters, so its isolated prefill
timing is excluded. Additional exclusive prefix/image probes were not run on
the busy server. Earlier isolated prefill results remain the comparison.

Both original worker identities were verified running and `/health` returned
200. Final cache capacity stayed at 3,313,955 aggregate tokens. The normal
launcher selects the measured immutable kit with four-over-six NVFP4 main KV,
group-32 FP8 / BF16 RoPE sliding KV, and 1792 MiB display KV per rank. No server
restart is required for committing or pushing these source changes.
