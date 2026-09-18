# Concurrent DCP attention — September 18, 2026

The recipe now enables the locally tested `concurrent` overlap schedule in its
shipped serving overlay. It remains TP2/DCP2, not pipeline parallelism. Own-head
attention runs while queries are exchanged; peer-head attention runs on the
communication stream, and packed results avoid an extra copy. Both streams
join before merging. The quantizers, attention reduction order, image visibility,
KV allocation and memory limits are unchanged. Code and adaptations are
**AGPL-3.0-only**, with full credit to [MiaAI's foundation](../CREDITS.md).

## Results and limits

Two offline runs covered 23 cases on each physical GPU, including bitwise FP32
partials/LSE and BF16 final-output comparisons, changed-input graph replay,
ragged/duplicate/masked keys, image-width attention and cross-slab prefill.
Invalid-index graph tests passed on both GPUs. At the model's 512-key setting:

| Attention component | Throughput improvement across the two runs |
|---|---:|
| Four-row decode | 6–10% |
| 24-row decode, representative of C6 verification | about 25% |
| 512-row eager text prefill | 16–20% |
| Mixed/image-width cases | about 8–10% |

These are component gains, not whole-model token rates. The local full-model
canary completed startup and target/draft graph capture, 12 serial requests,
32K retrieval, images, automatic tools, multilingual requests and C6 generation:

| Serving measurement | Observed |
|---|---:|
| Pooled serial decode, 12 requests at temperatures 0 and 1 | 30.52 tok/s |
| Individual serial decode range | 22.35–37.38 tok/s |
| Easy-prose decode, temperature 0 median | 30.10 tok/s |
| Fully uncached 32,766-token prefill | 1,026.0 tok/s |
| Six simultaneous requests, 768 total output tokens | 12.04 seconds including prefill/request overhead |

The identical request schedule's **historical v106** result was 29.51 tok/s
pooled decode and 978.57 tok/s prefill: increases of 3.43% and 4.85%. This was
**not a fresh controlled full-model A/B**. Ten of twelve replies and acceptance
fractions matched the old run exactly; two differed. Sampling/output variation,
clocks and host load affect the comparison. It is not a broad quality,
perplexity, statistical-significance or 50-tok/s claim.

Startup allocated GPU bytes were unchanged; the reserved allocator pool grew
by **4 MiB per GPU**. The cache is still display-backed 1.75 GiB per GPU with
utilization 0.92 and C6. vLLM reported 3,313,955 aggregate KV tokens; the
per-request limit remains 1,048,576. The full multi-million-token capacity test
was **not repeated** with this change. Fresh two-node installation remains
unqualified; see [release validation](release-validation.md).

## Installation and upgrades

Normal users use `./start-server.sh` as documented in the README. No experimental
builder, original campaign files, model conversion or runtime compilation is
required. The six overlap implementation files are byte-for-byte the frozen
v5 files used in the local canary, protected by runtime/source inventory pins.

The overlay is mounted from the Git recipe over the existing digest-pinned
GHCR rc2 image. Its compiled dependencies and native libraries are unchanged;
therefore this source-only update does not require republishing that image.
The image's archived corresponding source/cache still describe rc2; the new
overlay's corresponding AGPL source is included in this Git checkout. Ordinary
Triton JIT warmup or first-use compilation can occur for new signatures absent
from rc2's cache. There is no Docker/framework/C++ build fallback.

Git updates do not change a running server: each launch uses a frozen kit.
After updating a normal installation, a deliberate `./start-server.sh --restart`
prepares the new pinned kit and reuses existing verified model/image downloads.
Do not run that command merely to inspect the update. If configured with
`EXISTING_DEPLOYMENT`, that explicit deployment still chooses its own frozen
kit; update both its path and SHA256 deliberately, or remove both settings to
return to the normal public-recipe path. See [configuration](configuration.md).

Detailed development measurements and local evidence identifiers are in
[the experiment record](../release/experimental/dcp_overlap/RESULTS.md).
The older `query` and `balanced` schedules are not selected; they had image
and small-decode regressions.
