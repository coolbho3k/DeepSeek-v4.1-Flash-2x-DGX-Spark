# CPU-only Engram cache experiment

Standalone microbenchmark; not a serving patch. It does not open model files,
call CUDA, change the running server, or alter runtime manifests.

The baseline includes the exact shipped `row_store_core.cpp` and invokes its
serial `serve` function. Full credit to **MiaAI Lab / Wesley Young and upstream
contributors** for the reader, cache, locking and worker-pool foundation at
`979e68a62c90b24d928f5638596e0ceed90e9f34`. The harness and experimental adaptations
are **AGPL-3.0-only**. Original notices and licenses are retained under
`../../runtime/vendor/miaai-dsv41-agpl/`.

The experiment separates direct output copying, scalar/NEON/SVE key searches,
batched statistics, and grouping a batch by its 256 lock stripes. Locks remain
held while reading mutable cache entries. Miss fallback runs outside the lock.
Grouping can change replacement order, but must not change returned row bytes.

Timings are synthetic cache-hit component measurements, **not model tok/s**.
They exclude the production worker-pool dispatch, real SSD misses, CUDA host
callback dispatch, and GPU consumption of mapped buffers. Ordinary anonymous
host output memory is used; this is not a display-memory bandwidth experiment.
Correctness separately covers cold/replaced entries through an in-memory
reference backing store, ownership boundaries, image/dead IDs, duplicate IDs,
tail counts, all cache ways, and concurrent callers. No real weights are read.

The baseline uses generic AArch64 compilation, as the shipped native build did.
Candidate functions explicitly enable SVE2. All variants use the same libc
`memcpy`; no assumption is made that libc copies are scalar. A parked thread
disables glibc's single-thread mutex fast path so locking resembles serving.
Trials rotate variant order and input banks. Results include distributions and
paired speed ratios, with separate repeated-hot and rotating-working-set cases.

The current stage requires twelve local heads per rank and chunks at most 256
tokens: 12 rows per single token, 48 per four-token K3 verification, 288 for six
such requests, and 3,072 per full prefill chunk. These are component shapes, not
assertions that all those rows hit the cache. The 3,456- and 24,576-row cases are
larger synthetic stress shapes, not the current per-callback prefill shape.

Build and run on an SVE2-capable ARM64 host (GCC):

```bash
g++ -O3 -std=c++17 -pthread bench.cpp -o /tmp/engram-cpu-bench
nice -n 15 /tmp/engram-cpu-bench --cpu 19 --repeats 15 --cache-mib 64
```

The benchmark pins itself to one selected CPU. Validate that CPU exists and
inspect free memory first. Default runtime memory is under 160 MiB, including
the 64 MiB synthetic row cache. Use a process memory/CPU limit when sharing a
host with serving. The correctness concurrency check uses two threads on that
same CPU; it is a race-safety check, not a multithread scaling benchmark.

Summarize completed JSON-lines output without any serving access:

```bash
python3 summarize.py /path/to/benchmark.jsonl
```

## Measured results, 2026-09-19

Tested on dgx0, one pinned core at a time: Cortex-X925 (CPU 19, two confirmation
runs) and Cortex-A725 (CPU 14). GCC 13.3, `-O3 -std=c++17 -pthread`; runtime SVE
vector length was 128 bits, and disassembly confirmed SVE instructions in the
candidate. Each configuration used 15 recorded trials plus two warmups, with
seven variants, eight input/output banks, and randomized variant order.
No server restart, GPU allocation, checkpoint read, cache flush, or clock change.
The server remained on port 8888; aggregate request counts were zero at the
checks bracketing measurements. This was not an isolated-host experiment.

Representative rotating-working-set medians on X925, in microseconds per
serial component batch (ranges cover the two production-shape runs):

| Rows / shape | Original | Direct + batched counters | Grouped scalar | Grouped SVE |
| --- | ---: | ---: | ---: | ---: |
| 12 / one token | 0.271–0.277 | 0.209–0.213 | 0.462–0.473 | 0.526–0.543 |
| 48 / four-token K3 verification | 1.085–1.100 | 0.880–0.905 | 1.391–1.436 | 1.603–1.654 |
| 288 / six such requests | 7.522–7.601 | 6.550–6.953 | 7.649–7.947 | 8.965–9.331 |
| 3,072 / 256-token prefill chunk | 147.901–150.601 | 174.494–192.703 | 119.820–120.899 | 94.215–99.410 |

SVE is useful in the grouped prefill-shaped X925 case: its paired throughput
was 18–22% higher than the grouped scalar candidate. This comparison includes
SVE ID validation and key search together, not either operation in isolation.
At 3,072 rows the hot and masked patterns also favored grouped SVE on X925;
on A725 its incremental gain over grouped scalar was much smaller or absent.
Direct SVE key search without grouping was generally unhelpful. Simply removing
the temporary row copy was not consistently faster and often regressed bulk
performance. Small decode shapes favored scalar counter batching, but absolute
savings were only fractions of a microsecond to roughly one microsecond.

An earlier broader stress case of 24,576 random resident rows improved about
2.3× with grouping on both core types. **That exceeds the current 3,072-row
per-callback prefill maximum and must not be advertised as a serving gain.**

Every final run passed 1,225 exact-byte cases, including masked/unowned IDs,
cache-disabled fallback, cold caches, replacements, duplicate IDs, odd cache
ways, boundary/tail sizes, output guards, and concurrent callers with exact
hit/miss totals. An additional `-fsanitize=undefined,bounds
-fno-sanitize-recover=all` correctness-only build passed. This is not a general
race-detector proof or CUDA callback/graph qualification. Runtime peak RSS was
about 119 MiB (address space capped at 512 MiB); each full final run used
16–22 CPU-seconds. The production library and manifests remain unchanged.

Interpretation: retain this as a possible bulk cache-hit fast path, not an
always-on SIMD rewrite. A production candidate would need a small-batch
fallback, preserve asynchronous miss dispatch, and be tested with real hit
rates, the worker pool, mapped output buffers, and full serving. These results
do **not** demonstrate a decode/prefill tok/s improvement; the saved CPU time
is small and may already overlap GPU work.

Private raw evidence (not uploaded):

- `reports/engram-cpu-sve-v1/dgx0-x925-production-shapes.jsonl`
- `reports/engram-cpu-sve-v1/dgx0-x925-production-repeat.jsonl`
- `reports/engram-cpu-sve-v1/dgx0-a725-production-shapes.jsonl`
- `reports/engram-cpu-sve-v1/dgx0-ubsan.jsonl`

Provenance for the final measured build:

- Harness SHA256: `c2b46a2a86c962578359db9236b821f208b4016a4eccbd7a42f6cae3f64efff6`
- Included reader SHA256: `e4d8a813dff09b509685b571afb085ed43d37cce43f06ce1deeca75da9288f4c`
- Benchmark binary SHA256: `c271c468ef434d46c9229394ac2b7af0dd202d88f064a651d5038783bf4b1951`
