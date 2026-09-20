# Responses and agent-prefix cache update

Adapted from MiaAI Lab / Wesley Young and mrexodia's upstream
[PR12](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks/pull/12)
and [PR21](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks/pull/21).
All adaptations are **AGPL-3.0-only**; see [credits](../CREDITS.md).

- Responses `input_text` and `output_text` are normalized as text. Images,
  tools, native reasoning budgets and validation remain enabled.
- Both ranks enable prompt-token details, exposing
  `usage.prompt_tokens_details.cached_tokens` for Chat Completions.
- `PREFIX_CACHE_RETENTION_INTERVAL=4096` retains additional reusable
  sliding-window checkpoints. Set it to `0` to restore semantic-only retention,
  or use `--prefix-cache-retention-interval 0` when starting the server.
  This recipe accepts multiples of 256 through 1048576 to preserve hybrid
  scheduler alignment. Invalid values are rejected before an owned-server stop.

These checkpoints share the existing evictable KV pool: no extra GPU memory
is allocated. More retained checkpoints can consume some otherwise free cache
blocks until evicted. Our speculative replay-tail and image prefix-cache fixes
are retained. This targets earlier conversation forks/agent reuse; it is not
a decode kernel speedup or a promise of improved cold prefill.

Unchanged: 96 Engram threads, 2048-token prefill, six sessions, TP2/DCP2,
0.92 utilization, 1792 MiB display KV per rank, original memory safeguards,
canonical 3bpw target/draft and full vision. No new weights or native binary
build is needed: the recipe ships a verified source overlay on its pinned image.

## Local before/after check (2026-09-19 Pacific)

One unchanged-server baseline followed by one restart. Same synthetic prompts,
temperature 0, TP2/DCP2/K3, six slots and 0.92. Server timing counters isolate
each request; cached-token details agree with native cache-hit counters.

| Measurement | Before | Updated |
| --- | ---: | ---: |
| Garden decode | 29.04 tok/s | 27.39 tok/s |
| Python decode | 35.97 tok/s | 36.91 tok/s |
| Uncached 30,427-token prefill | 28.70 s | 30.32 s |
| First exact repeat, time to first token | 0.67 s | 2.42 s |
| Return after an unrelated turn, time to first token | 0.61 s | 0.57 s |
| Earlier 11,431-token fork, cached tokens | 0 | 8,192 |
| Earlier fork, time to first token | 10.76 s | 3.36 s |
| Appended conversation turn, time to first token | 0.71 s | 0.70 s |

The useful result is **3.2× faster first-token latency for the earlier fork**.
Exact repeats and appended turns still reused 30,208 tokens. All eight
deterministic Chat Completions outputs matched the baseline byte-for-byte.
This is a functional regression check, not a broad accuracy evaluation.

Do not infer a decode/cold-prefill improvement from this update. These are
single samples; the candidate was freshly restarted and emitted first-use
Triton JIT warnings. The first cached repeat was slower, while the subsequent
same-prefix request returned to the baseline range. No kernel bodies changed.

The valid Responses text-block request changed from HTTP400 (unsupported
`input_text`) to HTTP200 with the expected answer. Two six-request waves at
temperatures 0 and 1 completed at 65.46 and 53.20 aggregate output tok/s;
draft-token acceptance fractions were 46.31% and 43.89%. Those aggregate
throughput figures include prefill and are not per-session decode rates.

Automatic tools, native images, concurrent English/Chinese requests and an
uncached 128,007-token prompt also passed. Allocated KV capacity remains
3,313,955 tokens, with a 1,048,576-token per-request ceiling. This run did not
repeat the full-capacity stress test. The updated pair was left serving on
port 8888. All 172 tests in the clean staged-source export and the separate
speculative-cache alignment regression passed (the wider local working-tree
suite also passed all 186 tests).

Local evidence: `reports/upstream-api-cache-baseline-v1.json`,
`reports/upstream-api-responses-valid-baseline-v1.json`,
`reports/upstream-api-cache-candidate-v1.json`, and
`reports/upstream-api-cache-c6-v1.json`; practical qualification is recorded in
`reports/upstream-api-cache-qualified-v1.json`. Private campaign reports are deliberately
not shipped in the public clone. The serving source overlay tested here is
byte-identical across all 109 files in the public and local candidate kits;
the private launcher's existing asset bindings are not a fresh-download test.
