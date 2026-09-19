# Lossless Engram I/O update — September 18, 2026

The recipe ships the locally tested **page15 layout + GPU-readable host staging
+ deferred retrieval** together. This is not a new quantization: original
FP8/E8M0 row bytes, main/draft weights, images, routing, TP2/DCP2 and all memory
settings are unchanged. MiaAI Lab / Wesley Young provided the native row store,
bounded cache, callback/dequantization path and dense packed-row foundation.
Our page15 layout and integration are adaptations under **AGPL-3.0-only**.

## Measured outcome

| Measurement | Original Engram path | Combined update |
| --- | ---: | ---: |
| Pooled serial decode, same 12-request schedule | 30.355 tok/s | 30.784 tok/s |
| Median speculative-step latency | 77.394 ms | 76.907 ms |
| Draft acceptance | 45.093% | 45.753% |
| Easy prose, temperature-zero median | 30.370 tok/s | 30.510 tok/s |
| Uncached 32,767-token prefill, warmed kernels | 1,082.660 tok/s | 1,077.379 / 1,077.729 tok/s |

This is **a modest/noisy end-to-end change**, not a 2x serving improvement.
Cold bulk row retrieval alone was approximately twice as fast. The first full
prefill was 1,023.309 tok/s while both hosts compiled a previously missing DCP
top-k specialization. The two subsequent uncached requests above had zero
prefix-cache hits and warmed kernels. The background NFS archive was active;
this was not an isolated-disk benchmark or a statistically powered A/B.

The 12 requests covered gardening, code, explanations and easy prose, with
two temperature-zero seeds and a temperature-one seed per topic. Eleven replies
matched the baseline exactly, including all temperature-one replies. One
temperature-zero coding reply changed wording; its seed-dependent difference
repeated in follow-up testing. The cause is **not established**. Exact storage
and dequantization proofs are not a broad end-to-end quality guarantee.

Images, automatic tool calls, English/Chinese prompts and six simultaneous
128-token generations passed. The six-request batch took 12.128 seconds for
768 output tokens including prefill/request overhead. KV reporting remained
**3,313,955 aggregate tokens**, with a 1,048,576-token per-request limit. The
multi-million-token capacity test was not repeated for this update.

## Correctness and scope

- CPU tests checked original/dense/page15 rows byte-for-byte, ownership,
  dead/invalid IDs, duplicates, padding, headers, truncation and no-overwrite.
- Both full rank-owned tables were verified during packing, including source
  fingerprints, owned weight/scale digests and full packed-file digests.
- Both GPUs passed eager and changed-input graph replay checks, special FP8/
  E8M0 values, exact copied-versus-host-view outputs, stream/lifetime checks,
  and two-stage deferred graph capture. Synthetic CUDA event intervals showed
  actual retrieval overlap; this was not a full-model overlap trace.
- Direct GPU host views save only about **1.55 MiB per GPU** of staging, not
  another meaningful KV allocation. Their isolated timing benefit was mixed.

The exact tested serving code and native reader are shipped, not rebuilt by
users. The underlying GHCR image remains pinned and unchanged. The small new
native reader is carried by the Git serving overlay, with its complete
corresponding source. Ordinary Triton warmup/first-use compilation may occur.

## Atomic upgrade and rollback

1. Original `engrams/*.safetensors` remain untouched on Hugging Face. The
   canonical model/drafter revisions in the recipe lock do not change.
2. New data lives in a separate `engram-page15-v1/` namespace. Both ranks are
   staged first; all parts, manifest and model-card notice are promoted in one
   Hub commit. Only then does one Git commit pin that complete immutable HF
   revision, manifest and matching ABI-2 reader. There is no cross-service
   transaction; this publication order prevents dangling recipe references.
3. The launcher downloads only its own rank's two tables, about **97.66 GiB
   extra disk space per Spark**. It still retains the original checkpoint for
   metadata/compatibility; this release does not remove the old files. Transport
   parts are at most 8 GiB and are streamed into one assembly file. Each part
   and the complete assembled table are checksum-verified. Interruptions leave
   a resumable partial, not a usable table or a new preparation pointer.
4. Runtime kits and packed caches are versioned separately. Old runners read
   their old snapshot/layout; new runners require their exact manifest, format,
   ownership and verified local file identities. Missing or mixed versions
   fail closed. Running workers retain their existing frozen kit and mounts.

After updating Git, use the ordinary documented `./start-server.sh --restart`
when ready for downtime. No conversion/build commands are needed. To roll back,
select the preceding Git commit and restart deliberately; old weights and kits
are preserved. Do not replace files in a running kit. An explicit
`EXISTING_DEPLOYMENT` continues to choose its own old frozen kit until both its
path and digest are deliberately updated or the override is removed.

Fresh-clone preparation has offline and public-metadata checks, **not a fresh
two-node GPU qualification**. Publication does not restart the local server.
