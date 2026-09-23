# Model-specific fusion campaign

Status: shared-row projection and fused bounded gather selected after paired
component checks and repeated original/candidate serving measurements. The
selected runtime is healthy on port 8888 with the original KV configuration.

Optimize decode, speculative verification and prefill on the two GB10 Sparks
by fusing dependent operations and reducing intermediate memory traffic.
Preserve weights, arithmetic boundaries, vision, native collective behavior,
NVFP4 four-over-six main KV, group-32 FP8/BF16 sliding KV, display allocation,
cache capacity and memory limits. Serving results, not kernel count, determine
promotion. Keep separate dispatch where decode and prefill need different
geometry. The selected runtime remains serving on port 8888.

The local historical reports already contain five rejected MoE candidates:
serial and parallel gate/up fusion, one- and two-block persistent scheduling,
and decoded-register pipelining. The persistent variants regressed typical
component calls because of occupancy or spills. Do not repeat them unchanged.
Serial gate/up fusion did improve fully shared routes, so route-dependent
selection remains a distinct hypothesis requiring realistic routing evidence.

Implementations and rejected experiment:

- `packed_wo_a_rows.py` reuses each decoded weight tile across small
  speculative batches inside one CTA. It preserves the existing split-K
  layout and per-row FP32 reductions. Fresh paired traces put this operation
  above mHC postmix in decode cost. All 24 original cases on each Spark were
  bit-identical, including changed-input graph replay. A further tile sweep
  selected N=16 / four warps for two through four rows. See `RESULTS.md`.

- `prepare_gather.py` combines grouped-prefill gate/up input gathers and lets
  the launcher bound the grid by the current batch's maximum routed rows.
  Retains FP16 multiplication before the original FP32 Hadamard and the final
  FP16 store. No extra persistent GPU workspace. `build_gather.py` is the
  isolated CPU-only native builder. The paired 23-case gather probe passed
  byte comparisons, inactive canaries and changed-input graph replay.
- `post_prenorm.py` combines mHC postmix with the next small-row projection and
  squared norm, retaining the BF16 intermediate boundary. The existing
  normalization/Sinkhorn/carried-pre-mix consumer remains unchanged. Unlike a
  mathematical reassociation, the projection consumes the rounded residual.
  Rejected: neither tested variant preserved every residual/projection/norm
  boundary exactly. It is excluded from the serving candidate.

`launch.py` reuses an explicitly pinned deployment on port 8889, without
changing persistent user configuration or stopping an existing worker.
`profile.py` captures one native step under the same independent controller,
with the existing 3 GiB profiler admission floor. `summarize_traces.py` reports
GPU activity spans separately from overlapping duration sums.

Current local campaign evidence: `reports/model-fusion-v1/`. It includes the
baseline deployment and digest, historical negative-result references,
serial API benchmarks, paired native traces, and build receipts. The selected implementations and rejection reasons are recorded in
`RESULTS.md`.

`prepare_kit.py` clones a verified existing runtime, overlays only the selected
projection dispatch and gather, recursively refreshes source pins, and emits a
new immutable manifest plus a change receipt. It retains the existing launch
guards, images, model bindings and all cache settings. Copy and verify the kit
on both hosts before `launch.py`. `benchmark.py` waits for the same worker pair
and runs the existing serial, six-request, and uncached 32K probes under the
independent memory watchdog. `compare.py` checks matched settings and requests.

The selected implementations are also retained under `release/runtime/serving/`.
The public recipe keeps its existing image and fresh-clone qualification limits;
these measurements apply to the identified local two-Spark deployment.

The API benchmark and image-smoke drivers use the maintainer’s local probe
library and calibration corpus, which are outside the public export. Component
probes and the native kernel build inputs are included here. The final live
32K retrieval passed; overlapping API traffic invalidated its isolated timing,
and additional exclusive prefix/image checks were not run on the busy server.
