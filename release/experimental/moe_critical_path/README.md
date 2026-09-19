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
   gate/up and down time from routing/gather overhead. Promising larger work
   is reuse of dequantized tiles across rows routed to the same expert, with
   overlapped loads/compute. Preserve EXL3 MUL1 interpretation, scaling,
   nonlinearities and accumulation precision; avoid full expert dequantization
   buffers in RAM. This needs a measured roofline, not a speculative rewrite.

The initial selector transform is CPU-only and source-pinned. It changes only
which already-compiled geometry is requested. **No variant is enabled** and
no speed or quality improvement has been demonstrated. Geometry can change
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

The prepared probe compares every geometry against the canonical packed-expert
implementation as well as the current kernel. It includes missing/duplicate/
empty routes, changes expert addresses between graph replays, and poisons the
shared scratch before replay. These are planned GPU checks, **not yet executed**;
only the selector transformation and probe syntax have CPU validation.

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
