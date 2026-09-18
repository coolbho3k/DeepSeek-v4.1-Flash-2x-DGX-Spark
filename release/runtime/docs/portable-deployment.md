# Portable two-Spark launcher: implementation and qualification

The launcher is implemented and CPU-tested, but has NOT yet started a model.
Do not treat a command plan or the tests below as a completed public deployment.
The v21 pair used the original qualified local controller, completed1M/reuse
and post1M regression, and was stopped cleanly at2026-09-13 16:00Z for export.

`serving/portable_node.py` and `serving/portable_pair.py` in the workspace are
shipped as `tools/portable_node.py` and `tools/portable_pair.py` in the v4
runtime-input kit. The frozen v1/v2 archives predate these files and have not
been changed in place. V3 also remains frozen; v4 adds the verified bound-view path.

## What differs from the local controller

Host paths, installed image IDs, SSH endpoint, fabric addresses/interfaces and
non-root UID/GID come from a deployment JSON rather than author-specific
constants. Integrity checking uses public kit/model/cache manifest digests,
the public installed-image identity, and each host's OWN full-hash weight
receipt. It does not require the author's calibration/build/serving receipts.

The Docker model-serving arguments and resource flags match qualified v21
on both hosts, modulo deployment paths/name. A random owner label permits
read-only recovery of a lost create acknowledgement without accepting an
unrelated same-named container. It does not change model execution.

The pair controller remains a foreground RAM watcher AFTER the API becomes
ready. Every successful sample checks exact container ownership, original
start times, health state and at least768MiB MemAvailable per host. A low-RAM,
stopped/restarted worker, wrong AOT mapping or startup deadline triggers a stop
of only the recorded pair. A stop acknowledgement is insufficient: the watcher
continues until both workers are observed stopped. Transient observation
failures retry the same IDs, never create/start another model. Ctrl-C/SIGTERM
requests an exact-pair shutdown; unexpected controller errors do the same.

## Required release assets

Before a real deployment can be attempted, each host needs:

- A matching installed ARM64 runtime image, checked against the public image
  identity. The launcher never pulls, builds or installs an image.
- A verified runtime kit containing the launcher, unchanged worker/serving
  overlay, native MUL1 kernel and qualified MXFP8 AOT library.
- A materialized model directory matching the public release manifest (or
  the fully verified bound view described below), plus
  a fresh private full-hash receipt made on that host with the kit's standalone
  downloaded-release verifier. Merely inspecting a manifest is insufficient.
- A separately reviewed auxiliary-cache directory and its pinned
  `cache-manifest.json`. It must preserve required warm artifacts/mtimes for
  the matching image, including CUDA, Triton, FlashInfer and other tested
  cache roots. The archive has now been packed and independently extracted,
  with all12,957 payload hashes and exact mtimes checked. Privacy/licensing and
  physical deployment qualification remain pending.
- An existing dedicated writable run root, passwordless noninteractive SSH,
  working Docker/NVIDIA runtime and the correct active RoCE interface/GID.

The cache schema is `ds41_auxiliary_runtime_cache_v1` with a `files` map from
canonical relative paths to byte sizes, SHA256 and nanosecond mtimes. The launcher copies only
that allowlist into a fresh per-run cache, preserves mtimes, hashes each copied
file, and records fingerprints before start. It never copies `hf/`, `tmp/`
or lock files; the prepared archive also excludes ordinary logs. Empty HF/temp directories are created separately. Permitted
cache roots are cuda, triton, flashinfer, tilelang, vllm, torch-extensions,
exllamav3, numba and torch. An allowlist is not a public-data review by itself.
The preserved stopped v20 source produced a598,794,240-byte archive with
571,815,939 payload bytes. All packing/extraction happened after the v21
workers were stopped, and the source cache was left unchanged. The launcher
also checks manifest-provided mtimes before and after staging. Its larger
per-file fingerprint receipt has a separate16MiB limit; configuration inputs
retain their4MiB limit.

## Configure and inspect a plan

Start from the kit's `deployment.example.json`.
Replace ALL zero-digest placeholders, image IDs, paths, remote username and
UID/GID values. The example is deliberately incapable of passing real asset
verification unchanged. Physical host0 is local/logical rank1; physical host1
is remote/logical rank0 and serves the API. API/master ports remain8041/29541.

```bash
python3 -B /path/to/kit/tools/portable_pair.py --config /path/to/deployment.json
```

This prints two Docker command plans without querying Docker, writing run
files, reading weights, or launching anything. The sample describes target
paths; the command does not create `/srv` directories or install prerequisites.

Once all assets have been independently verified and GPUs are idle, the
intended execution command is:

```bash
python3 -B /path/to/kit/tools/portable_pair.py \
  --config /path/to/deployment.json --execute
```

Execution must not substitute an unverified model/cache or a manifest-only check.
Execution checks both hosts before staging, creates each container once,
rechecks original0.90 free/available memory reserves and staged fingerprints,
then starts both nodes. The same Python process remains the RAM watcher.
Run it in a managed terminal/session. At remote API readiness it verifies
actual AOT process mappings and starts a loopback SSH tunnel tied to the exact
API container's lifetime; tunnel creation is not itself a local endpoint test.
`--no-tunnel` leaves access on the remote loopback only.

For an existing recorded deployment, the following only reattaches the watcher;
it never starts or restarts workers. Only one watcher may hold its lock:

```bash
python3 -B /path/to/kit/tools/portable_pair.py \
  --config /path/to/deployment.json --watch-existing --no-tunnel
```

Per-node plan/preflight/inspect/recover-created actions are available through
`portable_node.py`. Recovery requires the original random owner label and
complete saved Docker contract; it observes an attempted create, not a retry.
A saved start attempt can never be dispatched again by the start action.
Keep private run journals and full-hash receipts out of the public release.

## Optional zero-copy bound model view

When another full weight copy will not fit, `tools/verify_mapped_release.py`
can prepare a view using `--prepare-sources PRIVATE-sources.json`. The source
map must name an explicit original file for EVERY one of the129 public files;
the tool copies only metadata and creates55 empty shard mount targets. It
writes a private55-entry bindings JSON. No source hardlinks or mutations occur.
Use the tool's `--help` for the required manifest, view, bindings and output
arguments; all outputs must be fresh and outside preserved input trees.

On EACH host run its `--full` mode with those bindings and the independently
pinned public manifest, producing that host's private full-hash receipt.
This freshly hashes all426,134,193,164 public bytes, not just metadata. A later
`--check-receipt` only validates unchanged fingerprints and is explicitly not
another full payload hash. Put the exact bindings dictionary in the optional
`model_bindings` field of that node's deployment configuration, set `model`
to the view and `model_receipt` to its own full-hash receipt. The launcher binds
every original shard read-only over the matching empty `/model` target.

The physical host view is NOT an upload directory: its shard files are empty.
Never upload it as weights. Only the correctly bound container exposes the
complete public payloads. Both actual CPU-only container views have now passed
all129-file inventory checks, all55 shard/source fingerprint checks, all56
read-only mount checks and74 freshly hashed metadata checks. No GPU serving
claim follows from those probes alone.

Full verification can leave substantial clean input pages in RAM. The optional
`tools/advise_verified_release_cache.py` validates the same-host bound receipt
and advises only its exact original shards/metadata with POSIX_FADV_DONTNEED.
Default is a read-only plan; `--execute --output PRIVATE-receipt.json` additionally
requires idle Docker/GPU and48GiB available RAM. It does not delete or rewrite
files, flush global caches, or guarantee a reclaimed amount. The unchanged
startup MemFree/MemAvailable checks decide whether serving may start afterward.

## Current evidence and limits

CPU tests compare both Docker commands with actual qualified v21 source,
exercise unsafe configuration/resource/ownership/cache refusals, prove
observation-only lost-create recovery and one-shot start refusal, and run
five watcher state machines: RAM floor, observation timeout, stop timeout,
unexpected restart and wrong AOT mapping. The watcher remains active after
readiness and confirms terminal states. A read-only preflight check on the
actual busy local host also refuses another deployment without a Docker action.

No physical model has been launched by this code. Before calling the recipe
reusable from public assets, complete cache/runtime distribution and a physical
deployment test with text, native vision and long-context/cache reuse. Preserve
the0.90 ceiling, original BF16 vision, SSD engrams, native FP8 KV, one sequence,
TP2/DCP2,1056-token eager scheduling and1,009,612,800B KV cap per GPU throughout.
