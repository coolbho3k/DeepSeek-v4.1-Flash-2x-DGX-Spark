# Local DCP overlap measurements — 2026-09-18

The first section contains **attention component** timings; the next records
the full-model canary. No weights, cache precision, sparse selection, image
visibility, memory limits or network settings changed. The frozen tested
implementation was subsequently promoted into the public recipe's source
overlay; the GHCR image itself was not changed.

## Concurrent schedule, frozen candidate v5

Parent: `artifacts/ds41-runtime-local-v107`, manifest SHA256
`58e33a7b637f953eb8aa7eeab3d705d0326af420a754ca0aa87efa3bb531a354`.

Candidate: `artifacts/ds41-runtime-dcp-overlap-v5`, manifest SHA256
`2874a3d7c1c88a05cfc9a659856b5af75c787f974f52ae7c3e696153bbf4391e`.

Evidence: `reports/dcp-overlap-gpu-v5/summary.json`,
`reports/dcp-overlap-repeat-v5/summary.json`, and
`reports/dcp-overlap-negative-v5/complete.json` (local, ignored artifacts).

Both runs passed all 23 cases on both physical GPUs over the normal two-rail
NCCL connection. Gates include FP32 partials/LSEs, final BF16 bit patterns,
nonfinite merges, four changed-input graph replays on alternating streams,
empty/ragged/duplicate/masked rows, image-width attention, mixed decode/prefill,
and cross-slab carry. Invalid indices were masked and rejected at the graph
boundary; a poisoned owner's next replay was rejected. The output-address
variants also have a CPU AST test proving unchanged arithmetic source.

Paired A/B timings alternate order, discard warmup, use the slower GPU for each
corresponding sample, then take the median. Decode has 16 forwards per graph;
wide prefill has four. Production-style eager prefill is measured separately.
The following speedups span the initial and repeat runs at **top-k 512**:

| Workload | Component throughput improvement |
|---|---:|
| 1-row captured decode | 0.7–9.2% |
| 4-row captured decode | 6.2–9.7% |
| 24-row captured decode | 24.8% |
| 512-row eager text prefill | 15.6–19.8% |
| 513-row eager text prefill | 4.0–12.3% |
| 516-row eager mixed/image-width prefill | 8.0–10.4% |
| 24-row captured mixed/image-width | 9.3–9.7% |

The unchanged SWA-only control is effectively flat. Traces show real kernel
overlap. Traced durations themselves are not used as throughput estimates:
profiler overhead and inter-rank skew can distort them. Peak allocated tensors
for the whole component harness were 274,685,440 bytes per GPU; this includes
fixtures and both variants' graphs and is **not incremental serving memory**.

Even a 25% improvement in this component does not imply a 25% model speedup.
Small-row gains are modest and noisy; no statistical significance or
50-token/sec claim is established.

## Full-model local canary

Run `ds41-release-v1789769024223362195` completed startup, all six target/draft
graph shapes, 12 serial benchmark requests, a fully uncached 32,766-token
prefill/retrieval, text/multilingual/image/auto-tool smoke checks, and six
simultaneous 128-output-token requests. **The server was left running on dgx0,
port 8888**, per the owner's instruction.

Local evidence: `reports/dcp-overlap-v5-{serving,prefill,smoke,summary}.json`.
The public launcher's state records the canary and owns its normal stop/status
operations. `.env.ds41` is unchanged: an ordinary future restart still selects
the previous parent kit unless the owner explicitly selects this candidate.

| Measurement | Observed |
|---|---:|
| Pooled serial decode, all 12 requests | 30.52 tok/s |
| Individual serial decode, temperatures 0 and 1 | 22.35–37.38 tok/s |
| Easy-prose decode, temperature 0 median | 30.10 tok/s |
| Fully uncached 32K prefill | 1,026.0 tok/s |
| C6: 768 output tokens, including request/prefill time | 12.04 seconds |

Compared with the **saved historical v106** run using the identical 12-request
schedule, pooled decode is 3.43% higher (29.51 to 30.52 tok/s); median per-step
time is 79.97 to 77.25 ms. The same 32K prefill is 4.85% higher (978.57 to
1,026.01 tok/s). This is **not a fresh controlled full-model A/B**; clock/load
variation and changed generated sequences can affect these numbers. Ten of
twelve replies and ten acceptance fractions match that historical run exactly;
the other two do not. This is not a perplexity or broad quality evaluation.

Utilization remains 0.92, TP2/DCP2, C6. The display-backed cache is still
1.75 GiB per GPU, with no ordinary CUDA KV pool; vLLM reports 3,313,955 aggregate
KV tokens and the configured per-request limit remains 1,048,576. This turn did
not fill that entire cache. Startup allocated GPU bytes are unchanged; the
reserved allocator pool increased by **4 MiB per GPU**. Final observed host
available RAM was approximately 2.5 GiB on dgx0 and 3.7 GiB on dgx1. No memory
boundary, image setting, weight, quantizer, or public runtime artifact changed.

## Earlier schedules

The v3 `balanced` and v4 `query` schedules were bit-exact in 19 component cases,
but split attention executed serially. The extra smaller kernel launches
erased much of the communication overlap. Four-row decode was flat or slower;
small mixed-image latency regressed about 39–84%. They were not enabled in
serving. Their fixture used 1024 selected keys, not the model's normal 512.

The concurrent schedule fixes that serialization and writes packed results
directly. It retains the same attention tiles and reduction order. Original
v1 failed a source-pin check before the kernel test; v2 timings had a baseline
packing-copy bias. Only v3 onward use the corrected timing harness.

All experimental code is AGPL-3.0-only. Full credit to MiaAI / MiaAI-Lab for
the upstream two-Spark stack and performance foundation; these measurements
are of this recipe's new scheduling work, not an upstream performance claim.
