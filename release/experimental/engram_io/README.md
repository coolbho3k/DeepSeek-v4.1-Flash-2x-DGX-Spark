# Engram I/O campaign — component and local serving results

All code here is AGPL-3.0-only. The native reader, bounded row cache, worker
pool, and dense weight+scale packed format come from MiaAI-Lab, commit
`979e68a62c90b24d928f5638596e0ceed90e9f34`. Full credit to Mia's AI Lab and
upstream contributors. See `../../runtime/vendor/miaai-dsv41-agpl/LICENSE`
and `LICENSE.MIT`. The original vendored source is not modified.

Local additions: fifteen complete rows per 4 KiB page, byte/latency accounting,
recipe integration, and experiments in retrieval overlap and GPU-readable host
staging. No change to row bytes, quantization, image behavior, KV precision or
capacity, TP2/DCP2, six-session limit, or serving memory safety boundaries.

## Required completion evidence

1. Fresh unchanged-server baseline: private reports
   `engram-io-baseline-v2-{serving,prefill,summary}.json`.
   Pooled decode 30.355 tok/s (23.790–36.380), easy-prose T0 median 30.370,
   45.093% draft acceptance, uncached 32,767-token prefill 1,082.660 tok/s.
   The NFS copy was active; this is not an isolated-disk measurement.
2. CPU byte-exact original/dense/page15 reader and packer tests, including
   invalid headers, boundaries, partial pages, rank ownership and dead IDs.
3. Paired real-SSD component timings on both Sparks with identical access IDs,
   ordering/warmup/cache policy recorded. Production has twelve local heads.
4. Full rank-owned packed artifacts with source identity and verified bytes.
   Canonical HF weights remain untouched; all output is separate.
5. GPU dequant, image/dead IDs, capture/replay, buffer-lifetime and bounds checks
   for copied and GPU-readable host staging on both GPUs.
6. Explicit experiments in asynchronous retrieval vs synchronous retrieval;
   traces/timings must establish actual overlap, not merely another stream.
7. Frozen candidate integration and full serving tests: acceptance/output
   checks, text/code/varied prompts, image/tool calls, decode, prefill, C6,
   memory and unchanged reported KV capacity. Record rejected variants too.
8. Leave the best verified setup running, document results and how to reproduce
   the supported recipe, and preserve the known-good fallback. Do not claim a
   faster configuration based only on component tests.

## Evidence collected so far

- `reports/engram-io-full-v1`: nine CPU tests passed on each host. Independent
  bucket calculation matched the installed vLLM `EngramLayout` exactly.
  Both full rank-owned tables were packed in **both** layouts on both hosts,
  with source fingerprints, per-range source hashes, packed hashes, and bounded
  write/fsync/readback verification. This took about 8.3 minutes on dgx0 and
  11.8 minutes on dgx1. The original checkpoint files were read-only.
- `reports/engram-io-ssd-v2`: 24 paired cases per host, 30 repeats per case,
  identical IDs and rotating original/dense/page15 order. All returned bytes
  matched. Requests cover 1, 4, 24 and 256 tokens, twelve local heads, cold
  native row cache, fully repeated IDs, and 80% hot IDs. O_DIRECT bypasses the
  filesystem page cache; physical SSD/controller caches were not flushed.
  For cold 256-token batches, original/page15 medians were approximately
  21.2/10.7–11.2 ms on dgx0 and 10.3–10.4/5.3–5.4 ms on dgx1. This is a
  component result, not an end-to-end inference speedup.
- `reports/engram-io-gpu-v2`: both GPUs passed eager and changed-input graph
  replay checks, dead/image/unowned/out-of-range masking, independent CPU
  dequant comparison (including FP8/E8M0 special values), exact copied/UVA
  output bits, cross-stream reuse, graph-lifetime refusal, and owned two-stage
  deferred graphs. CUDA event intervals demonstrate about 0.4 ms of actual
  overlap with the synthetic GPU work. This is **not** a model-compute trace
  or a serving speed claim. Peak test tensor allocation was under 130 MiB.
- Direct host views eliminate 811,008 bytes of intermediate CUDA storage per
  table (about 1.55 MiB for both tables per rank). The copy-elimination timing
  benefit alone is small/inconsistent; there is no demonstrated KV-capacity
  increase. Larger synthetic cases do not fit entirely in their small row
  cache; v2 records the actual hit/miss deltas instead of assuming all hits.

Component tests use a 20-core CPU quota to avoid artificial bulk-test
throttling; the full serving trial preserves its existing six-core quota.
The background NFS archive remained active, so these are not isolated-disk
measurements. Prior failed/preliminary attempts are retained: node0's first
packer rejected zero-byte model-view mount targets, and the first rank1 SSD
benchmark incorrectly passed the exclusive upper bound to the raw C ABI.
The production stage masks that bound before calling C; the corrected v2 CPU
benchmark does the same. Neither failure modified original weights.

## Current serving trial

The combined page15 + GPU-readable staging + deferred retrieval candidate is
`artifacts/ds41-runtime-engram-io-v1`, manifest SHA256
`f3387571f67d436e9b68bba8e3d526a06289558c7e4e3c6162cb1e3eed3492be`.
The local full-serving comparison **completed** and the server was left running.
The tested kit remains immutable, including its historical pre-test qualification
labels. Pooled decode was 30.784 versus 30.355 tok/s; warmed uncached 32K prefill
was about 1,078 versus 1,083 tok/s. Eleven of twelve replies matched; one coding
reply differed reproducibly by seed. Images, tools and six simultaneous requests
passed. No broad quality or new long-capacity claim is made. See the public
[results and atomic upgrade instructions](../../../docs/engram-io-performance.md).
The ordinary public launcher downloads the packed assets and uses the matching
prebuilt reader; users do not run these campaign builders. Inspect existing
portable state before any lifecycle operation; publication does not restart it.

## Known-good fallback

Baseline server (currently stopped): `ds41-release-v1789769024223362195`, port 8888. Parent kit
`artifacts/ds41-runtime-dcp-overlap-v5`, SHA256
`2874a3d7c1c88a05cfc9a659856b5af75c787f974f52ae7c3e696153bbf4391e`.
The original frozen runtime remains unchanged.
Server may be stopped for this user-authorized campaign, but avoid needless
restarts. No native GPUDirect Storage: Spark supports compatibility mode only;
do not load nvidia-fs. Host-readable GPU staging is a distinct experiment.

The parent prepares both Engram layers on the main stream before decoder
computation. The candidate uses one ordered retrieval stream and inserts a
completion dependency when each Engram layer consumes its rows. Both graph
ownership and eager-buffer reuse are retained. The original hashing, image
mask, TP all-gather, and subsequent gate arithmetic are untouched.

## Reproducing the local campaign

The scripts are explicit, experimental tools; none implicitly stops a server
or reads credentials. Use `--help` for required paths and separately recorded
digests. `start_packing.py` and `run_components.py` require the recorded serving
pair to be stopped; the latter starts identified test containers and returns
their IDs. Monitor those exact containers with `docker inspect`/`docker logs`;
do not restart a still-running operation after a client-side timeout.

1. Save the unchanged serving baseline before stopping the server.
2. `start_packing.py` snapshots the small code, tests the native reader, verifies
   the independent partitions against installed vLLM, and prepares rank-owned
   dense/page15 files in a fresh directory on each Spark. It handles explicit
   deployment `model_bindings`, not just symlinks in model views.
3. `run_components.py --kind ssd` compares the complete real local artifacts.
   `--kind gpu` exercises copied/UVA staging and owned deferred graphs. Their
   result directories must be fresh; inspect exit status as well as JSON.
4. `prepare_candidate.py` builds a fresh, immutable runtime input kit using
   the pinned parent, reader binary, upstream Engram SHA, and both complete
   rank manifests. `--no-mapped`, `--no-overlap`, and `--layout` support ablation
   candidates without weakening source pins or safety limits. Copy only that
   kit to the peer, verify its hash, and use the existing explicit
   `../dcp_overlap/launch_candidate.py` lifecycle helper.
5. The campaign's private `probes/finish_engram_io_trial.py` compares the exact
   baseline requests and uncached prefill nonce, checks text/image/tool behavior
   and six simultaneous requests, and leaves the tested server up. It refuses
   changed serving/KV settings or missing independent RAM supervision.

Both experimental layouts together use about 192 GiB per Spark. Only the
selected layout is mounted in serving, read-only. Packing is a one-time write;
lookups use read-only O_DIRECT reads and a bounded RAM row cache. These are
lossless local storage caches of the existing HF Engram weights, not new
quantizations. Neither main nor draft canonical weights need replacement.

The background NFS archive was active during these measurements; it was not
paused. Record background load when comparing results. Do not weaken container
isolation to control unrelated host processes.

## Corresponding native build

Normal users consume the reviewed 75-KiB prebuilt reader in the Git overlay.
For source inspection/modification, the tested build used an ARM64 Ubuntu
toolchain with `g++ -O3 -std=c++17 -shared -fPIC -pthread row_store.cpp -o
librow_store.so` in a fresh output directory. `row_store_core.cpp` is included
beside the wrapper. It derives from the credited MiaAI source above. A rebuilt
binary needs new digest pins and CPU/GPU qualification; never overwrite a live
runtime library or bypass its attestation checks.
