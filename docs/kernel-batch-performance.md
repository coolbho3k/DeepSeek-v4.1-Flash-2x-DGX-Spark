# Combined decode attention and exact top-k experiment

The combined runtime is retained for now and deployment v106 is left running
on port 8888. This is a modest decode improvement, not a general performance
breakthrough or a model-quality certification. These kernels were subsequently
published in GHCR `20260917-rc2`; the public recipe is pinned to that digest.
One full-model startup was shared by the two
selected changes; component kernels were first checked on both GPUs.

## Serving result — 2026-09-17

| Measurement | Original v105 | Combined v106 | Change |
| --- | ---: | ---: | ---: |
| Pooled decode, all 12 matched requests | 28.27 tok/s | 29.51 tok/s | +4.38% |
| Garden, T=0 median | 23.76 tok/s | 25.88 tok/s | +8.92% |
| Python tutorial, T=0 median | 31.46 tok/s | 33.24 tok/s | +5.64% |
| Database explanation, T=0 median | 34.11 tok/s | 35.14 tok/s | +3.03% |
| Photosynthesis explanation, T=0 median | 27.19 tok/s | 29.45 tok/s | +8.29% |
| Published cooperative C1 prompt, T=0 median | 29.98 tok/s | 35.67 tok/s | +18.99% |
| Same C1 prompt, T=0 cycle time | 85.33 ms | 79.33 ms | -7.03% |

The broad suite's cycle times generally fell 6–9%, but speculative acceptance
and outputs changed. Pooled broad-suite acceptance was 2,849 / 5,853 = 48.68%
before and 2,764 / 6,096 = 45.34% afterward. In contrast, C1 T=0 acceptance
rose from 52.35% to 60.99%, contributing to that prompt's larger throughput
gain. The T=1 garden request was 4.73% slower despite its shorter cycles.
None of the 15 paired replies was byte-identical. The attention reduction
order changes, but these small trials do not establish the cause or generality
of acceptance differences, statistical significance, or model-quality parity.
The earlier 15–30% overall decode estimate was too optimistic.

The 32,766-input-token retrieval passed and confirmed zero prefix-cache hits.
Measured server prefill was 33.48 s, or **978.57 input tok/s**; time to first
token was 33.54 s and total request wall time was 34.02 s. The earlier original
runtime v103 request took 32.35 s total. Thus this one prefill observation was
about 5.2% slower, not an improvement; the older baseline did not retain the
same isolated prefill counters. No statistical claim follows from one pair.

Six simultaneous 128-token replies completed in 12.31 s total wall time.
Automatic tool selection/Paris arguments and the image/chart answer passed.
Both containers remained running without OOM kills or restarts. Sampled final
MemAvailable was 2.19 GiB on dgx0 and 3.79 GiB on dgx1. KV allocation and safety
limits were unchanged; this trial did not revalidate maximum usable context.
All 52 CPU regression/comparison tests passed.

Original campaign evidence (not included in the release checkout; published
aggregate results are in `release/validation-summary.json`):

- `reports/kernel-batch-serving-v106.json`
- `reports/kernel-batch-comparison-v106.json`
- `reports/kernel-batch-upstream-c1-v106.json`
- `reports/kernel-batch-upstream-comparison-v106.json`
- `reports/kernel-batch-prefill-v106.json`
- `reports/kernel-batch-smoke-v106.json`

The subsequent GHCR publication contains the new native library, corresponding
AGPLv3 source/notices and required kernel cache; the recipe's immutable runtime
and manifest pins have been updated. No target or drafter weights changed and
no HF artifact was regenerated. Publication did not stop or restart v106.
See `release/GHCR.md` for the digest and audit scope. The performance experiment
itself did not publish; this was a separate, explicitly authorized release step.

## Selected changes

1. **One-pass split-K attention.** Small-row decoding computes per-split online
   softmax and weighted values together, then merges the splits. This removes
   the separate normalizer pass and repeated QK/cache decode. Existing FP4 main
   KV, FP8 sliding-window KV, sink/mask semantics, DCP and split counts remain.
   Larger-row prefill already used an online path and is unchanged.
2. **Length-aware exact radix top-k.** A native CUDA/CUB implementation selects
   the same indices without sorting the entire reserved capacity for short
   contexts. Device-side lengths bound active work; no CPU length readback is
   introduced. Hierarchical merges preserve the existing FP32 ordering,
   tie-breaks, NaN behavior, signed-zero equivalence and padding semantics.

The original cooperative MoE stays enabled. The alternative MoE-X PR #6 kernel
was slower in our matched trial and is not enabled here. Direct-paged MXFP4
scorer candidates v1/v2 were numerically correct but were rejected after
component timing; neither is in the serving candidate.

The attention reduction order changes. Passing numerical gates does not prove
bit-exact model output or unchanged model quality. Exact top-k selection was
checked against the existing implementation, including pathological ties.

## Component evidence

`reports/kernel-batch-gpu-v4/host{0,1}/complete.json` retains final qualification
results. Attention has 14 configurations per rank, covering FP4/FP8, ragged
lengths, empty/sink cases, strided inputs and changed-input graph replays.
The unchanged attention gate is NMSE <= 1e-7 and LSE absolute error <= 2e-4.
Representative affected attention shapes took about 18–30% less GPU time.

Native top-k has nine shapes per rank with random values, ties, infinities,
NaNs, signed zero and graph mutations. Selected indices matched exactly.
Representative rank-0 component timings (milliseconds, reserved width shown):

| Rows / width / K | Original, short length | New, short length | Original, full length | New, full length |
| --- | ---: | ---: | ---: | ---: |
| 4 / 65,536 / 1,024 | 0.459 | 0.029 | 0.458 | 0.105 |
| 4 / 524,288 / 512 | 0.694 | 0.031 | 0.689 | 0.277 |
| 4 / 524,288 / 2,048 | 0.699 | 0.031 | 0.693 | 0.379 |
| 24 / 262,144 / 512 | 3.341 | 0.031 | 3.333 | 0.615 |

Both GPUs also passed invalid-input rejection and graph-poisoning checks.
Frozen-runtime registration, idempotence and both selected kernels replaying
together in one native CUDA graph passed on both GPUs:
`reports/kernel-batch-registration-host{0,1}.log`.

These are component results, not model tok/s. The model has eight index source
layers; top-k improvements cannot be multiplied across every transformer layer.
Full serving measurement is necessary to establish the net benefit.

## Fixed serving configuration and provenance

Baseline: v105, immutable runtime v21, manifest
`caf7eeee7668bc11434eb4b4c06589f81a45a08cdef7ea480d8b8303f757af23`.
Candidate: v106, `artifacts/ds41-runtime-kernel-batch-v1`, manifest
`b81ec8d744c579d57b530ab0d22fa446b2812b113ad6ca1d2511405dd14abfe3`.

Both use identical target/draft weights, container images, TP2/DCP2, DSpark
k=3, full BF16 vision, six-session capacity, 0.92 GPU utilization, 1,048,576
maximum context, 2048-token batching, 1792 MiB external display KV per GPU and
zero ordinary KV allocation. RAM/watchdog/admission limits are unchanged.
The new selection scratch does not imply extra usable KV capacity.

The packager `scripts/prepare_kernel_batch_runtime.py` pins sources and the
compiled native binary, preserves corresponding source and licenses, and
supports independent `--without-attention` / `--without-topk` switches for a
later bisect if necessary. It does not modify the published release.

## Serving measurement protocol

`probes/finish_kernel_batch_trial.py` waits for exact v106 readiness and checks
both kernel selections, cooperative MoE, DSpark and unchanged vision/config.
It never stops or restarts the service. Measurements use the existing independent
RAM watchdog and unchanged request-admission margin.

- Unchanged 12-request suite: four prompts, T=0 seeds 41/1729 and T=1 seed 41,
  400 output-token cap. Reports separate decode throughput, speculative
  acceptance, useful tokens per cycle and wall time per cycle.
- Exact published MiaAI cooperative C1 prompt, matching our v105 repeat.
- Same 32K synthetic retrieval, with added isolated timing/prefix counters.
  The historical baseline only retained total request wall time, so isolated
  uncached prefill speedup cannot be inferred from that baseline.
- Six simultaneous 128-token requests, automatic tool call with arguments and
  an image/chart answer. These are functional smoke tests, not broad quality
  evaluation or a repeat of the prior million-token capacity campaign.

`probes/compare_kernel_batch.py` refuses mismatched configurations, benchmark
code, request schedules or output lengths. Pooled throughput is total decoded
tokens divided by total measured decode time, not an average of rates.

## Attribution

The serving foundation and cooperative MoE integration build on MiaAI Lab's
AGPLv3 recipe, with original MiaAI, ExLlamaV3/Turboderp and other contributor
notices retained in the runtime and repository credits. These local experimental
modules are marked `SPDX-License-Identifier: AGPL-3.0-only`. The native top-k
uses NVIDIA CUB through the installed CUDA/CCCL headers under their existing
licenses; it does not vendor a copy of CUB's implementation.
