# Native MTU qualification: two GB10 hosts, active RDMA MTU 1024

Qualification performed 2026-10-10 UTC for PR #6 / issue #5. This is a
limited two-host result, not broad hardware/fabric or sustained-load qualification.

## Build and integrity

- Driver 580.173.02 on both GB10 SM121 hosts; NCCL 2.30.7+cuda13.3.
- Both selected RoCE ports on each host: active MTU 1024, hardware max 4096.
- Candidate compiled with CUDA nvcc for sm_121, without GPU access.
- Candidate library SHA256:
  `ebc318c18241302c55a19493e83a8a9abff5bf626cc8dcdd5419fa8541f0aeaa`.
- Unchanged parent runtime manifest SHA256:
  `965f19c3559c37e8ff6db92461f134ffee60918f4491dea92b78f10ec649c6f7`.
- Fresh candidate manifest SHA256:
  `94fcd1f3813fc185211fa8f5469ba93965785fbf65de4e857b1b720fcda4581c`.
- Both candidate kit copies passed the original hash verifier (893 files),
  and normal serving source/library attestation remained enabled.
- Two CPU regressions passed (MTU negotiation and integrated-kit pin refresh),
  plus 17 existing DCP overlap/release checks. Invalid/legacy advertisements
  and all 25 standard endpoint MTU pairs are covered on CPU.

The original runtime, model files, manifests, kernel caches and original
containers were untouched. The rebuilt library and refreshed source pins
existed only in a separately generated candidate kit. A direct comparison
confirmed the original staged serving overlay equals the original runtime's
serving directory before candidate substitution.

## Native collective test

The external harness used the same shipped serving image, original RoCE/GID
environment and two CPU-bootstrapped rank adapters as issue #5. No PyNccl
bypass hook was enabled; native `fc_allgather` and `fc_allreduce_bf16` were
invoked through the candidate's real fastcomm dispatch. The existing PyNccl
communicator was used only for bootstrap/reference infrastructure.

Each sequence all-gathered BF16 `[rows,5120]` on an explicit side stream,
joined its completion event to the caller stream, and all-reduced the same
shape on the caller stream. Both gathered rank values and sums were checked
on CPU after synchronization.

| Check | Both ranks |
| --- | --- |
| Negotiated QP MTU | 1024 bytes (local=1024, peer=1024) |
| Eager sequences | 128 passed, rows `[18,1,6,24,18,18,6,18]` repeated 16 times |
| Captured sequence | Side-stream gather/event join/caller-stream reduction |
| Graph replays | 100 passed, correct values each replay |
| Container exit | 0 |
| Captured Xid / NV_ERR_NO_MEMORY | None |

The original unpatched library failed on the first corresponding eager
sequence after approximately 28 seconds with Xid 43 on both hosts and ample
free memory. The candidate passed with the actual 1024-byte port MTUs,
without changing NIC/network MTUs. This provides a native A/B result, not
just the previously successful PyNccl bypass.

## Full model startup and inference

One existing serving pair was stopped, followed by one full-model restart
using the candidate kit. No fallback restart was needed. Both rank commands
had neither `DS41_FASTCOMM_PYNCCL` nor `DS41_DRAFT_EMBED_PYNCCL` set. Startup
logged the native negotiated MTU on both ranks.

- Started 03:01:29 UTC; application startup complete 03:10:28 UTC (~8m59s).
- DeepSeek V4.1 Flash EXL3 3-bit, TP2/DCP2, DSpark speculation 3.
- Original utilization 0.92, max length 1048576, max sequences 6,
  max batched tokens 2048, PDL enabled, dual RoCE and all original GPU/DRM/
  shared-memory/device/container settings preserved.
- Independent host watchdogs retained the original 512 MiB stop threshold;
  minimum sampled MemAvailable was 4059164672 / 5270917120 bytes before the
  post-inference snapshot. Neither watchdog triggered; neither container OOM-killed.
- `/health` returned 200.
- Three temperature-0 completion requests (`max_tokens=16`) succeeded:
  “The capital of France is” -> “ Paris.”;
  “The opposite of hot is” -> “ cold. The opposite of hot is cold.”;
  repeated capital prompt -> “ Paris.”.
- No Xid or driver NV_ERR_NO_MEMORY appeared in either captured kernel log
  through the post-inference snapshot. Both workers remained running.

These are startup/correctness smoke checks, not a throughput benchmark,
long-context/load test, or evidence covering arbitrary routed fabrics.
Hardware tests used symmetric MTU 1024 only; asymmetric and other MTUs are
CPU-policy-tested, not hardware-qualified here.

## Repeating the candidate build

Build `fastcomm.cu` using `build.sh` into a fresh output directory. For a
verified parent that already integrates fastcomm, run:

```sh
python3 -B release/experimental/fastcomm/prepare_kit.py \
  --parent-kit PARENT_KIT --parent-sha256 PARENT_MANIFEST_SHA256 \
  --library BUILD_OUTPUT/libfastcomm.so --replace-existing \
  --output FRESH_CANDIDATE_KIT --receipt FRESH_RECEIPT.json
```

The original integration workflow without `--replace-existing` remains
available for pre-fastcomm parent kits. Use the normal watchdog-protected
two-node launch with the verified candidate overlay and private writable
caches, retaining all original device/DRM/network settings. The lightweight
sequence is included in issue #5; the existing `bench_fc.py` provides further
eager/graph tests. Test only with idle GPUs; do not mutate a live verified kit.
