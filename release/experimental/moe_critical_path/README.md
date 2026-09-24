# Next critical-path campaign (not enabled in serving)

Scope: improve actual target/draft step latency and prefill without another
quantization, less image support, smaller KV, more memory utilization, or
relaxed allocation/watchdog bounds. Start from the healthy page15 canary.

## First combined component experiment

1. **Sweep existing cooperative geometries** at physical rows 1, 4, 6, 8, 12,
   16 and 24, varying distinct/shared/partially shared routing. The shipped
   ABI-2 native binary already implements narrow/narrow (0), wide/wide (1,
   current default) and wide/narrow (2). Test target and sparse drafter banks.
   This is a new experiment, not a missing MiaAI switch: MiaAI's incorporated
   runtime also selected geometry 1. A source comment describing generic
   auto-selection is not proof geometry 2 will be faster or equivalent.
2. **Inspect redundant route preparation/output materialization** using the
   same component timing intervals: FP16 conversion + ID mapping + counter
   reset, optional rotations, gate/up, down, and final FP32-to-BF16 conversion.
   The adapter already fuses preparation, so do not claim that existing
   optimization as new. Any proposed output-store fusion must retain the exact
   final rounding and fixed-order expert summation. Do not spend a full-model
   boot on microsecond savings alone.
3. **Prepare a grouped-prefill kernel experiment** only after separating
   gate/up and down time from routing/gather overhead. Inspection confirms
   dequantized-tile reuse across rows and four-stage asynchronous load overlap
   are ALREADY implemented in MiaAI's grouped kernel. The concrete new targets
   are smaller row tiles for underfilled expert segments, a measured thin/fat
   crossover, and combining the two input gathers. Preserve EXL3 MUL1
   interpretation, scaling, nonlinearities and accumulation precision; avoid
   full expert dequantization buffers in RAM.

The initial selector transform is CPU-only and source-pinned. It changes only
which already-compiled geometry is requested. **No variant is enabled** and
no end-to-end speed or quality improvement has been demonstrated. Geometry can change
reduction order: require output-error checks against the current geometry and
canonical expert implementation, edge/duplicate/missing routes, graph replay,
scratch alias/lifetime tests, and eventual acceptance/output validation.

Run GPU experiments in an explicit idle/maintenance window, not beside this
memory-tight live server. Use the same binary/weight inputs on both GPUs,
rotated order, warm and cache-flushed measurements, changed-input graph replay,
and realistic *distinct physical weights*. The old six-expert/384-ID alias
fixture is useful for correctness but not whole-layer bandwidth. Combine
component candidates that pass, then do one serving restart and bisect only
if necessary. Leave the best verified server running afterward.

The probe compares every geometry against the canonical packed-expert
implementation as well as the current kernel. It includes missing/duplicate/
empty routes, changes expert addresses between graph replays, and poisons the
shared scratch before replay.

## Component results, 2026-09-18 campaign

Both GPUs completed 42 geometry cases with 384 distinct real layer-zero
experts, synthetic activations/routes, changed-input graph replay and poisoned
scratch. Canonical-reference NMSE stayed below the existing 2e-5 bound
(approximately 5.3e-6 worst observed). This does not establish full-model
acceptance or broad quality.

- Geometry 0 was approximately 5--7% faster for one-row component calls.
  Geometry 1, the current setting, remained faster for representative
  multi-row verification cases. Do not globally switch to geometry 0.
- In balanced 2048-row prefill component traces, gate/up took about 9.1 ms,
  down 4.2 ms, and the two input gathers together 1.4 ms on each GPU.
  These are one-layer synthetic component timings, not server request times.
- Balanced 128/512-row calls spent about 17--18 ms predominantly inside the
  thin-expert kernel. Their average per-expert row counts are 2/8, versus
  32 for the 2048-row case. The fixed crossover is 16 rows per expert.
- The grouped main loop always performs all 64 tile rows, replicating the
  final valid input for padding and suppressing padded output stores. A
  32-row specialization can avoid that padded arithmetic but will not halve
  all kernel costs: compressed-weight reads/dequantization remain, and a
  globally smaller tile would reread weights for genuinely full 64-row tiles.

`run_components.py --prefill-sweep` selects a separate component-only sweep
of thresholds 2/4/8/16/32/64, checking full outputs against the current
dispatcher and sampled rows against canonical experts. It rotates timing
order and uses balanced/random/skewed routing; the original threshold is
restored before process exit. It never changes a serving artifact.

Both ranks subsequently completed that nine-case threshold sweep using 384
distinct experts (peak allocator use 3.69 GiB). Relative to threshold 16,
threshold 2 reduced component latency by 7--30% at 128/512 rows and 15--18%
for skewed 2048-row routing. Balanced 2048-row timings were effectively
unchanged (about 0.7% difference); random 2048-row differences were 2--4%.
Worst full-output NMSE against threshold 16 was 2.60e-6; worst sampled-row
canonical NMSE was 3.29e-6. These synthetic component checks do not replace
graph/edge-case or model acceptance checks. The serving threshold remains 16.

`prefill_rows.py` prepares a smaller first native experiment: skip completely
padded 16-row arithmetic blocks in the existing main loop. It preserves all
valid-row expressions and the 64-row ABI, input pipeline, scratch and shared
memory layout. Source-pin/structural/row-coverage tests pass on CPU. The
candidate has **not been compiled, GPU-qualified or installed**; it does not
claim the occupancy benefit of a true 32-row tile.

Next decode hypothesis: co-locate gate/up work to reduce intermediate writes
and cross-block completion fences/counters while preserving existing math.
This is not implemented or demonstrated faster. Extra register pressure may
outweigh the saving. Input preparation is already fused; weight reuse and
prefetching are already present. Do not remove synchronization without
replacing its producer/consumer ordering, or change accumulation/rounding
as a supposed quality-neutral optimization.

More ambitious decode experiments, not implemented or measured:

1. Co-locate gate/up plus activation in a block while retaining each
   projection's current reduction and rounding schedule; measure register
   pressure as well as eliminated global handoffs.
   A concrete first design targets geometry-1 multi-row verification only:
   compute the gate and up 128-column tiles sequentially with reusable
   accumulator registers, retain their FP16 results in block-local shared
   memory (8 rows x 128 columns x 2 bytes per projection), then run the
   existing activation/down-input transform locally. Preserve per-projection
   K partitioning and FP16 fold cadence. This can replace gate/up global
   partial stores and the gate/up last-arrival handoff, without altering the
   down kernel. It needs a local-row output mode in the GEMV helper, not just
   a pointer substitution (current output indices use global route slots).
   Keep the existing single-row path initially: halving block count there
   could reduce GPU utilization even if the fused arithmetic is faster.
2. A bounded persistent scheduler could start an expert's down work once its
   complete activation is available, without the current layer-wide kernel
   boundary. This needs safe resident-work scheduling, release/acquire
   ordering, and graph-replay validation; a naive spin-wait grid can deadlock.
3. Explicitly pipeline register dequantization with tensor-core work, beyond
   the existing packed-weight prefetch. Preserve the FP16-to-FP32 fold
   cadence even if prefetch depth changes. The current 3-bit build has zero
   reported register spill loads/stores, so removing spills is not a finding.

Do not assume B200-specific `tcgen05`/TMEM kernels are portable to GB10.
NVIDIA CUTLASS explicitly identifies SM120/SM121 as lacking that path:
https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py

The campaign uses API port **8889** for all subsequent launches and baseline
restorations. Public recipe defaults remain unchanged at 8888.

## NCCL: check evidence before tuning

Read-only inspection of both running workers found:

```
NCCL_NET=IB
NCCL_IB_HCA==rocep1s0f1:1,roceP2p1s0f1:1
NCCL_IB_MERGE_NICS=1
NCCL_IB_DISABLE=0
```

These are observed host names, **not public defaults**. Both rails are already
selected; the single socket interface is for bootstrap/control. Configuration
does not prove both rails carry traffic. Snapshot per-port RoCE counters on
both hosts around a measured request and check rank readiness/timeline skew.
Only pursue bandwidth/channel/protocol tuning if the evidence indicates a
bandwidth limitation. Do not increase communication buffers speculatively.

Historical v54 decode profiles showed about 16.1 ms in AllReduce on one rank
versus 3.2 ms on the other. These old profiles predate current cooperative/DCP
updates and are **not a current bottleneck breakdown**. Their asymmetry cautions
against treating summed NCCL duration as wire-transfer time: peer waiting can
be included. Obtain a new bounded trace when sufficient profiler headroom is
available; do not weaken the existing 3-GiB profiler admission requirement.

All adaptations are AGPL-3.0-only. Full credit to MiaAI Lab / Wesley Young and
Turboderp for the cooperative kernel foundation; original notices remain in
the recipe's vendor/source trees.
