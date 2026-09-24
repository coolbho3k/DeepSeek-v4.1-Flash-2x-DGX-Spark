# DSpark K≤5 and drafter critical-path campaign

Experimental variants, not enabled by default. The completed performance
comparison retains the original fixed K3 runtime: the combined K3 repeat was
0.4% slower overall, prefill was effectively tied, and C6 results were mixed.
See [results, exact benchmark inputs and limitations](RESULTS.md).

Full objective: test fixed K=3/4/5, adaptive verification (acceptance EMA and
native confidence), dedicated top-3 draft MoE, KV-only context projection, and
Markov/sampling fusion. Measure combined candidates, retain only demonstrated
wins, leave the confirmed winner serving on port 8888, and publish the tested
recipe to GitHub. Component failures/rejections are results, not permission to
silently omit an experiment.

`rebase_public.py` stages an explicitly pinned candidate on the separately
verified public runtime parent. It preserves the public downloader/launcher
and requires every serving file to be byte-identical to the candidate. Staging
does not select a winner, assert qualification, deploy, or publish anything.

Current component results are not a release recommendation:

- Capacity-36 target MoE: 154 numerical/replay cases per GPU passed against
  the original implementation; target kernel arithmetic is unchanged.
- Draft top-3: 273 cases per GPU across all three real 128-expert banks passed
  the existing numerical bound and changed-input/poisoned graph checks.
  Distinct-route latency improved from about 0.83 to 0.51 ms at five rows and
  4.03 to 2.78 ms at thirty rows. Keep the original one-to-four-row path.
- KV-only context projection: 39 cases per GPU across all three real draft
  layers were bitwise equal to the original KV output, including changed-input
  graph replay. Packed values and scales remain views of existing storage.
- Markov addition/sampling fusion: native logits and sampled tokens matched
  exactly in both the initial and strided-input 48-case/GPU fixtures; savings
  were only microseconds.
  The full-head fusion variant was tested too, but regressed at C6 and with
  FP64 noise, so it is not selected.
- Actual registered draft dispatcher and graph owner: 54 cases per GPU passed
  across three real expert banks, including changed routes, poisoned scratch,
  invalid expert sentinels and native sampling fallbacks.

Fixed K5 completed a matched serving comparison but did not beat K3 overall.
Fixed K4 also completed serial, uncached prefill and C6 comparisons, with mixed
results. Combined K3/K4/K5 and both adaptive policies have been exercised;
none established a better general-purpose default at unchanged limits. Target
temperature-zero text is not claimed bitwise invariant across batch shapes;
component equivalence does not establish end-to-end task accuracy.

`summarize.py` reads completed serial benchmark reports without importing CUDA.
It reports useful decode throughput, mean scheduled draft length and
unconditional acceptance by position. It refuses incomplete reports and mixed
engine counters. Comparisons are descriptive, not statistical significance or
quality claims. Native scheduler counters do not expose the worker's
confidence-trimmed verification budget and omit zero-draft rounds; do not use
them alone to claim confidence pruning occurred. See
[current results and remaining gates](RESULTS.md).

The unchanged checkpoint supports five draft positions and has three draft
transformer layers. These are different quantities. Six requests require up to
36 target rows at K=5, versus the old 24-row bound at K=3. The extended native
workspace aliases the original four serialized MoE buffers; no extra persistent
MoE scratch is needed. This does NOT imply CUDA graphs or activations are free.

Keep canonical target/draft weights, precision, full vision, TP2/DCP2, C6, KV
allocation, utilization, CPU limits, and safety margins unchanged. No live kit
edits, no unowned CUDA graphs, no silent native fallback, no approximate
rejection sampling, no vocabulary pruning. Boot-time graph capacities must
cover every admitted policy. Adaptive verification trims a verified prefix;
it does not make the fixed-size draft backbone itself cheaper.

Qualification order:

1. Fresh unchanged K3 baseline: multiple content types, temperatures 0 and 1,
   fixed seeds, 400 output tokens; separate uncached 32K prefill retrieval.
2. CPU resource, source-integrity, scheduler and graph-key tests; bounded
   CPU-only compilation while serving remains up.
3. Both-GPU maintenance component tests: real expert banks, original numerical
   reference, changed-input/poisoned graph replay, interleaved timings.
4. Matched fixed K3/4/5 and adaptive trials; C1 and C6, request isolation,
   useful tokens/sec, position acceptance, latency, headroom and usable KV.
5. Combine successful component candidates; bisect regressions, repeat the
   winner against baseline, validate normal serving and leave it up. Publish
   only after full correctness/performance/clean-install review.

MiaAI Lab / Wesley Young and contributors receive full credit for the
cooperative MoE kernels and their serving work; those parts and adaptations
are AGPL-3.0-only, with original MIT/ExLlamaV3 notices retained under
`release/runtime/vendor/miaai-cooperative-moe-agpl`. Native vLLM DSpark code
retains its Apache-2.0 notices. Adaptive EMA lessons come from the attributed
GLM recipe (warmup, recovery from censored acceptance, prefix-only verification,
and boot-captured graph lengths); policy below is a local implementation.
