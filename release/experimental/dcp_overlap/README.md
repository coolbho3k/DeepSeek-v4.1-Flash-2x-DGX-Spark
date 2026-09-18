# Experimental DCP communication overlap

**Development tools; the tested `concurrent` schedule is now in the recipe.**
Normal users only need the main README and `start-server.sh`, not this builder.
See [release results](../../../docs/dcp-overlap-performance.md). The original schedules passed
component correctness but had performance regressions. The revised `concurrent`
schedule passed two complete two-Spark component runs (23 cases per rank),
including bitwise FP32/BF16 parity, owned graph replay and image-width attention.
Its invalid-input graph test also passed on both ranks. See [RESULTS.md](RESULTS.md)
for measured gains and limitations. The local full-model canary also passed
startup, serial benchmarks, 32K retrieval, image/tool checks and C6 requests;
it was left running. This is not broad quality/long-context release qualification.

Testing and server restarts were explicitly authorized. The public source overlay
now carries the frozen tested code; the GHCR image, weights, KV allocation and
memory limits are unchanged. This is still
**TP=2 / DCP=2**, not pipeline parallelism.

## What changes

The current path exchanges both ranks' queries, computes their attention
partials, then exchanges the partial results. This candidate computes the
heads whose queries are already local while query transfer is in flight.

The selected experimental `concurrent` schedule runs own-head attention on the
caller stream and peer-head attention on the existing side stream after the
query gather. The result gather follows peer attention on that same stream;
both streams rejoin before merging the FP32 results. Shared output buffers
are allocated on the caller stream and retained through the join. Address-only
kernel variants write directly into the result payload, removing a packing
copy. Eager masked-load error flags are checked together after the join, before
the forward returns; captured errors retain the native graph-owner boundary.

For small decode batches, `balanced` uses this schedule:

```text
Compute stream: own 16 heads | peer 32 heads | own remaining 16 | FP32 merge
NCCL stream:    query gather |              | result gather   |
                            ^ query join                     ^ result join
```

For wide prefill, it keeps the original 32-head MMA tiles. Result transfer
can overlap the following chunk's local attention; the final chunk drains
before returning. `query` leaves own heads together for an easier bisect;
it still allows that inter-chunk result overlap. `off` returns the original
forward function and creates no extra CUDA stream.

The implementation uses the existing PyNCCL communicator, with an explicit
side stream and CUDA-event dependencies. It adds no process group, persistent
GPU tensor workspace, networking configuration, allocator override, or systemd
unit. Startup must prepare the stream outside capture. Failed submissions
poison the transport and retain outstanding buffers, rather than attempting
CUDA recovery or reusing possibly-live memory.

Weights, quantization, cache bytes, sparse key order, duplicate keys, image
visibility, and per-head reduction order are intended to stay unchanged.
The merge retains the existing FP32 arithmetic and one final BF16 cast.
**Bitwise equivalence is established for the recorded component cases only.**
Reordering kernels can also change scheduling, resource contention, compiler
specializations, and transient memory use. A speedup is not guaranteed.

## Files

- `policy.py`: default-off selection, bounded head schedule, byte accounting.
- `transport.py`: existing-communicator fork/join, graph ownership, lifetimes.
- `attention.py`: existing attention kernels on head tiles and FP32 merge.
- `packed.py`: address-only attention/split-merge output variants; AST parity tested.
- `integration.py`: startup-only forward/worker integration and late mapper binding.
- `prepare.py`: standard-library builder for a fresh private runtime-input bundle.
- `test_cpu.py`: deferred policy, source-anchor, pin, integration, and mock-lifetime tests.
- `probe_gpu.py`: deferred real two-rank forward/graph correctness and optional timing probe.
- `run_pair.py`, `summarize.py`: bounded real-pair execution and paired-rank timing summaries.
- `launch_candidate.py`: reuse the public launcher for an explicit local canary;
  does not stop a server, read secrets, edit `.env.ds41`, or change serving settings.

## After permission to test

First run the CPU suite from the repository root; this does not import Torch
or connect to either GPU. It creates temporary runtime-input copies, not weights:

```bash
python3 -B -m unittest release.experimental.dcp_overlap.test_cpu
```

For further experiments, use an explicitly verified **pre-overlap** parent kit.
The current public kit already contains overlap and intentionally fails the
builder's double-installation check. Prepare separate immutable candidates:

```bash
python3 -B release/experimental/dcp_overlap/prepare.py \
  --parent /absolute/path/to/current-verified-kit \
  --parent-sha256 PARENT_MANIFEST_SHA256 \
  --output /absolute/path/to/new-private-overlap-kit \
  --mode concurrent
```

Use the digest recorded in the current deployment receipt, not an untrusted
download's self-reported digest. `--mode off` and `--mode query` build separate
controls. The builder verifies the whole parent inventory, refuses overwrites,
adds the startup hooks in the **copy**, propagates source pins, refreshes the
overlay manifest, and records the result as **unqualified**. It does not change
the current launch profile, install dependencies, copy anything to dgx1, build
native binaries, start containers, or publish an image.

The GPU probe must run on **idle GPUs in fresh test containers**, one process
on each Spark, using the existing pinned runtime image, normal runtime mounts
and import paths, and the candidate mounted read-only at `/opt/ds41-serving`.
It requires Torch/Triton/vLLM already supplied by that runtime. It does not
load weights or need the display-memory pool. The script refuses an existing
GPU compute process and requires `--acknowledge-idle-gpus` before CUDA import.

Use the existing two-host test runner's rendezvous and network settings; do
not launch a second test worker into a serving container. The torchrun child
command is `probe_gpu.py --acknowledge-idle-gpus` (optionally `--timing`). It
expects `WORLD_SIZE=2`, `LOCAL_WORLD_SIZE=1`, and the standard torchrun rank
and rendezvous variables. Save **both ranks'** output. Put an external timeout
around the test job: a failed rank must not leave the peer waiting indefinitely.
The implementation does not itself stop processes, invent launch networking,
or claim that passing the component test qualifies native worker startup.

## Required gates before enabling serving

1. CPU suite, parent-copy source pins, repeated registration, and unchanged
   runtime configuration pass for all four modes.
2. On both GPUs, require bitwise equality against the parent forward for
   split-K decode sizes, the 16/32-head boundary, empty batches, ragged keys,
   duplicates/padding, masked invalid tokens, image-width SWA, mixed
   decode/prefill, and a 513-row cross-chunk batch. Require owned graph replays
   to remain exact with changed inputs and alternating replay streams.
3. Separately exercise invalid metadata, bounds errors, and graph/transport
   poisoning in disposable processes. Check nonfinite merge cases and FP32
   partials/LSEs independently; final-output equality alone is insufficient.
4. Inspect a two-rank trace to verify actual NCCL/attention overlap. Compare
   captured critical-path time and transient/graph-pool memory, not the sum of
   NCCL event times. Measure both captured decode and eager prefill; the latter
   matches the production FULL_DECODE_ONLY configuration.
5. Only after those pass, perform a matched full-model launch and A/B test:
   logits/tokens, speculative acceptance, multilingual/code/prose, images,
   tools, C1/C6 decode and prefill, and long-context memory stability. Preserve
   the current KV allocation, utilization and safety limits for the comparison.

The component probe implements step 2, independent FP32 partial/LSE and
nonfinite-merge checks, and optional timing. Run `--invalid-replay` separately
in fresh disposable processes to check masked invalid loads and rejection of
a second replay through a poisoned graph owner. These do not cover every
step 3 failure mode or qualify steps 4–5. Real image-encoder behavior and vLLM
native process-group startup need the full-model test. The main fixture now
uses the model's actual 512-key top-k setting; 1024-key cases are retained as
stress tests. Earlier v3/v4 measurements used only 1024 keys.

At the maximum 512-row slab, the explicit query receive buffer is 32 MiB,
the result send payload about 32.06 MiB, and its receive buffer about 64.13 MiB.
These are **not additional persistent allocations or a peak-memory estimate**:
the parent already has communication buffers, and the candidate changes their
lifetimes. A deferred previous chunk, attention scratch, NCCL resources, and
CUDA graph pools must all be measured. Do not increase utilization or reduce
the safety reserve on the assumption that overlap is free.

Rollback is selecting the original immutable parent kit for a later restart.
There are no changes to undo in an existing worker or parent kit.

## License and credit

All code in this experimental directory is explicitly **AGPL-3.0-only**.
Full credit to **MiaAI / MiaAI-Lab** for the original two-Spark serving stack
and upstream performance work on which this recipe builds:
[DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks).
This overlap scheduler is new work in this recipe, not an upstream MiaAI
performance claim. The merge and probe harness adapt this recipe's existing
AGPLv3 code. Preserve all parent notices and vendored AGPL/MIT licenses.
vLLM/PyTorch/NCCL remain dependencies under their respective licenses; no
NCCL implementation is copied here.
