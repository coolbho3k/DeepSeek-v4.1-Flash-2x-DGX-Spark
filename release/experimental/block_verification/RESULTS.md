# Block verification results

Decision: retain standard rejection sampling as the default. Native block
verification passed the focused correctness checks but did not demonstrate a
useful overall serving speedup in this trial. The experimental candidate was
left healthy on port 8889 as requested; persistent/public defaults remain
probabilistic K3 drafting with standard verification.

Reuse the completed probabilistic-drafting baseline under
`reports/draft-sampling-v1/`. No new serving baseline requests were sent. Both
sides use the same four prompts, temperature 1.0, top-p 0.95, seed 41, and
400-token serial limits, plus the same six-request wave with 256 tokens per
request. Each serial candidate was measured twice and C6 three times.

| Serial case | Saved standard mean tok/s | Block mean tok/s | Change |
|---|---:|---:|---:|
| garden | 24.31 | 23.79 | -2.1% |
| python | 29.94 | 28.66 | -4.3% |
| explanation | 29.28 | 30.06 | +2.7% |
| easy_prose | 25.80 | 26.91 | +4.3% |

Median of the four paired throughput ratios: **+0.27%**, effectively flat.
All four serial replies were identical between candidate repeats. Changing the
verification algorithm changes stochastic continuations at the same seed, so
this is not a comparison of identical generated text across the two methods.

| Six-request wave | Saved standard tok/s | Block tok/s |
|---|---:|---:|
| First use | 45.67 | 46.41 |
| Warm 1 | 53.25 | 52.27 |
| Warm 2 | 53.68 | 53.91 |
| Warm mean | 53.47 | 53.09 |

Warm C6 mean: **-0.70%**. Individual warm block waves range from -2.23% to
+0.83% against the saved warm mean. Retain the first-use waves separately;
neither mode's first-use result represents its warm throughput.

The serial differences mainly follow how many tokens each verification step
produces. For garden, that falls from 1.691 to 1.656 with essentially unchanged
step time. Python falls from 2.134 to 2.046, also with nearly unchanged step
time. Prose improves from 1.814 to 1.918 tokens per step, outweighing a 1.4%
increase in step time. These measurements include changed continuations and
routing; they do not isolate the verifier's kernel overhead. The theoretical
expected-acceptance advantage on matched distributions does not require every
finite generated continuation to be faster.

Validation on each GB10:

- 22 selected installed native tests passed: block distribution, acceptance,
  greedy, and placeholder cases.
- 10 additional checks passed: BF16 proposal distributions at temperatures
  0.6 and 1.0, plus full 129,280-token-vocabulary CUDA graph replay for one/six
  requests, temperatures zero/one, changed inputs and placeholders.
- Native verifier source hashes matched across the two installed images.
- Runtime manifests were verified on both nodes. Serving requests had isolated
  counters and passed the existing worker-identity and memory-watchdog checks.

These checks do not prove distributional correctness for every possible model
or establish a model-quality improvement. The performance study covers four
prompts and one seed, uses historical controls, and has no long-prefill result.

The first two attempts on port 8888 were rejected for overlapping API traffic:
one before measurement, one during the first serial request. They are retained
as failed evidence and excluded. All reported timings come from the isolated
port-8889 deployment.

Weights, images, four-over-six NVFP4 main KV, group-32 FP8/BF16 sliding KV,
1,792 MiB display KV per rank, zero ordinary KV, C6, 2,048-token prefill batches,
and 0.92 utilization were unchanged. Capacity remains 3,313,955 aggregate KV
tokens. Only `rejection_sample_method="block"` changed in the two launch
profiles, with the corresponding overlay manifest refreshed.

Candidate manifest:
`e36940ce51dd77ead8cbec097ab5a472f5be1094f10f5da5917ddca097c727b3`.
Local evidence: `reports/block-verification-v1/`, including the saved-baseline
provenance, preparation receipt, native and BF16/graph checks, all three serving
trials, comparison, startup excerpts, failed overlap attempts, and final health.
