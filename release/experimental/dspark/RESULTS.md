# DSpark longer-draft campaign — retain the original K3 default

Selection: retain the original fixed K3 serving runtime. The back-to-back
confirmation did not establish an overall benefit from the new draft kernels;
longer/adaptive drafting did not establish a better general-purpose default.
This is a practical default selection, not a statistically proven claim that
K3 wins every workload. Experimental variants are not enabled in the recipe.

Canonical target and quantized drafter weights, full vision, TP2/DCP2, C6,
utilization 0.92, the display-backed 1.75-GiB-per-rank KV allocation and existing
memory limits remain unchanged. The checkpoint has three transformer layers
but supports five draft positions; no new weights are needed for K4/K5.

## Comparison at a glance

Serial ratios are geometric means across the same eight content/temperature
groups, relative to the initial unchanged K3 run. C6 is aggregate end-to-end
throughput, not per-request decode. These are descriptive measurements; the
unchanged and combined K3 repeats are included below.

| Variant | Serial ratio | Uncached 32K prefill tok/s | C6 tok/s, T0 / T1 |
| --- | ---: | ---: | ---: |
| Unchanged K3, initial | 1.000 | 1,036.84 | Not measured in this run |
| Unchanged K3 repeat | 1.033 | 1,054.06 | 62.05 / 54.10 |
| K3 + draft kernels | 1.019 | 1,048.75 | 57.30 / 56.73 |
| K3 + draft kernels, repeat | 1.029 | 1,053.64 | 60.55 / 56.89 |
| Fixed K4 | 1.001 | 1,046.26 | 54.99 / 51.66 |
| K4 + draft kernels | 1.012 | 1,050.91 | 56.38 / 53.48 |
| Fixed K5 | 0.951 | 1,032.00 | 50.74 / 50.33 |
| K5 + draft kernels | 0.965 | 1,032.14 | 52.42 / 49.34 |
| Adaptive EMA 1/3/5 + kernels | 0.977 | 1,029.23 | 55.39 / 55.07 |
| Adaptive EMA 1–5 + kernels | Insufficient headroom | — | — |
| Native confidence K≤5 + kernels | Incomplete: headroom | — | — |

All completed variants retained the same allocated KV capacity. None of these
figures establishes a new full-capacity or broad task-quality qualification.

The unchanged K3 repeat completed on isolated port 8889. Its serial geometric
mean was about 3.3% faster than the initial baseline. Against this new control,
the earlier combined K3 and K4 results are 1.4% and 2.0% slower respectively.
Thus the earlier small apparent gains are not confirmed. Control T0 gardening/
Python/explanation/easy prose: 26.50/37.39/36.72/31.04 tok/s; T1:
23.05/31.33/34.40/30.49. Uncached 32,767-token retrieval passed at 1,054.06 tok/s.
C6 T0/T1 aggregate throughput was 62.05/54.10 tok/s; median client decode
estimates 11.84/10.72 tok/s; draft acceptance 0.440/0.429. Post-test headroom
was 2.29/3.80 GiB. Evidence: `dspark-k3-control-v2-*`.

The back-to-back combined K3 repeat then completed all twelve serial cases,
uncached 32K retrieval and both C6 waves. Against the repeated unchanged
control, its serial speed ratio was **0.99596** (0.4% slower); prefill was
**1,053.64 vs 1,054.06 tok/s**. T0 gardening/Python/explanation/easy prose:
27.30/33.40/36.35/30.55 tok/s; T1: 24.07/33.63/34.48/29.31. C6 aggregate
throughput was **60.55/56.89 vs 62.05/54.10 tok/s** at T0/T1: mixed, not a
consistent win. Median client decode estimates were 11.69/10.68 tok/s, draft
acceptance 0.448/0.424, and final headroom 2.80/3.73 GiB. Evidence:
`dspark-k3-kernels-repeat-v2-*`.

The new kernels demonstrated component-level savings, but this full serving
comparison does not justify a default runtime change. Preserve the original
K3's existing serving payload, GHCR/HF pins, KV allocation and quality path.
Publish the experimental source and negative/mixed results for follow-up work;
do not advertise a new general speedup or promote adaptive variants that failed
the unchanged headroom requirement.

## Fixed-length controls

Twelve 400-token serial requests: four content types, temperature-zero seeds
41/1729 and temperature-one seed41. Table entries are temperature-zero medians
of useful decode tokens/sec from isolated per-request server counters.

| Content | K3 | K4 | K5 |
| --- | ---: | ---: | ---: |
| Gardening | 25.97 | 23.17 | 22.29 |
| Python | 34.94 | 34.33 | 32.55 |
| Explanation | 35.51 | 37.83 | 37.09 |
| Easy prose | 30.67 | 28.90 | 27.19 |

Across the eight content/temperature groups, geometric-mean throughput ratios
against K3 were 1.001 for K4 and 0.951 for K5. These are descriptive ratios,
not a statistically powered A/B. NFS archival was active during all trials.
Temperature-zero text was not bitwise invariant across different K; the Python
prompt also varied within the unchanged baseline. Broad task accuracy has not
been qualified, and component checks must not be presented as that proof.

Uncached approximately 32K-token prefill was 1,036.84 / 1,046.26 / 1,032.00 tok/s
for K3/K4/K5; all three synthetic retrieval checks passed with zero prefix hits.
C6 waves produced 1,536 tokens each: K4 aggregate end-to-end throughput was
54.99 / 51.66 tok/s at temperatures 0/1; K5 was 50.74 / 50.33. These include
request/prefill overhead and are not per-request isolated decode rates. The
matched K3 C6 control and confirmation repeats are recorded above.

All fixed runs allocated 3,313,955 aggregate KV tokens with a 1,048,576-token
per-request limit. This is allocation evidence, not a new full-capacity stress
test. K4 post-test host headroom was approximately 2.14/3.63 GiB; K5 was 2.38/3.54.

Local immutable evidence identifiers:

- K3 reports: `dspark-k5-baseline-v1-{serving,prefill,summary}.json`.
- K4 reports: `dspark-k4-v2-{serving,prefill,summary,c6}.json`.
- K5 reports: `dspark-k5-v3-{serving,prefill,summary,c6}.json`.

## Components and combined trials

Capacity-36 extends the attributed MiaAI target kernel's geometry without
changing arithmetic; 154 cases/GPU were bitwise equal to the original path.
For rows above 24 the reference used two calls, not a native 36-row baseline.
Attention graph/numerical checks passed on both GPUs, including 524,288 local
positions. They do not establish full-model long-context quality.

The specialized top-3 draft kernel passed 273 direct cases/GPU and 54 actual
registered-dispatcher/graph-owner cases/GPU. Compared with the canonical BF16
expert computation its maximum direct-test NMSE was 5.88e-6; it is not claimed
bitwise equal. Select it only for 5–30 rows. The KV-only projection passed 39
cases/GPU with bitwise-identical KV output and zero copied packed weights.

Markov addition plus sampling fusion passed both contiguous and actual strided
48-case/GPU fixtures with exact cached logits and tokens. Full-head fusion
was also implemented/tested but is rejected: it regressed at C6, severely with
FP64 sampling noise. It is not included in combined candidates.

Combined fixed K5 completed the same serial, prefill and C6 schedule. Its
geometric-mean serial speed ratio was 0.965 against K3, versus 0.951 for plain
K5. This is not a general win. Uncached 32,758-token prefill was 1,032.14 tok/s,
retrieval passed, and C6 aggregate end-to-end throughput was 52.42/49.34 tok/s
at temperatures 0/1. Post-test host headroom was 2.70/3.55 GiB; the allocated
KV capacity was unchanged. Evidence: `dspark-k5-kernels-v1-*` reports.

Combined K3 also completed the matched schedule. Across the eight groups its
geometric-mean serial speed ratio was 1.0188 versus unchanged K3, with individual
groups both faster and slower; the repeats above did not confirm a general gain. T0
gardening/Python/explanation/easy prose: 27.12/33.27/35.55/30.89 tok/s. T1:
23.73/33.24/33.94/28.85. Uncached 32,758-token prefill was 1,048.75 tok/s with
retrieval passing. C6 aggregate end-to-end throughput was 57.30/56.73 tok/s at
temperatures 0/1; median client decode estimates were 11.16/10.88 tok/s and
draft acceptance fractions 0.439/0.422. Allocated KV was unchanged. Evidence:
`dspark-k3-kernels-v1-*`. These are the initial measurements, not the later
unchanged-K3 control and combined-K3 confirmation results above.

Combined K4 completed all twelve serial cases, uncached 32K retrieval and both
C6 waves. Its eight-group geometric-mean speed ratio was 1.0121 versus the
initial K3 baseline, with mixed per-content results. T0 gardening/Python/
explanation/easy prose: 23.90/34.70/38.80/28.93 tok/s; T1:
22.71/35.60/32.77/29.41. Prefill was 1,050.91 tok/s, with retrieval passing and
zero prefix hits. C6 aggregate end-to-end throughput was 56.38/53.48 tok/s;
median client decode estimates were 11.66/10.56 tok/s and draft acceptance
fractions 0.371/0.372. Post-test headroom was 2.36/3.63 GiB. Allocated KV and
all serving limits were unchanged. Evidence: `dspark-k4-kernels-v1-*`.
This does not demonstrate a clear overall advantage over combined K3.

The initial EMA-prefix K5 variant captured all 30 C1–C6 / prefix1–5 target
graphs, passed readiness, and used a 2.27-GiB CUDA graph pool. dgx0 then had
approximately 0.91 GiB available, so the unchanged 1-GiB admission floor rejected
benchmarks before any prompt was sent. It was explicitly stopped; no crash or
speed/correctness result is claimed. A lower-memory 1/3/5-prefix variant passed
CPU registration and full-model startup, using a 1.59-GiB graph pool with the
same KV allocation. Its first serial test recorded seven cases before an
external LAN request broke isolation; the incomplete comparison is excluded,
not treated as a model failure or a demonstrated speedup. The same immutable
build was rerun on an isolated campaign port and completed all twelve serial
cases, uncached 32K retrieval and both C6 waves. Its eight-group geometric-mean
serial speed ratio was 0.9774 versus unchanged K3 (about 2.3% slower). T0
gardening/Python/explanation/easy prose: 24.72/32.16/35.79/29.36 tok/s; T1:
20.97/30.04/34.89/29.88. Mean scheduled draft lengths varied from 2.48 to 3.50,
confirming the policy shortened proposals. Uncached 32,758-token prefill was
1,029.23 tok/s, retrieval passed, and C6 aggregate end-to-end throughput was
55.39/55.07 tok/s at temperatures 0/1. This is not a general throughput win.
Evidence: `dspark-ema135-isolated-v1-{serving,prefill,summary,c6}.json`.

Native confidence needed an SM121 compatibility adaptation: upstream only
advertises device/CPU query-length mismatch for SM90/SM100. The existing native
flattened metadata path was tested with the SM121 FP4 scorer on both GPUs:
26 cases each, C1–C6, differing CPU/GPU lengths, compression1/2, padding,
per-token DCP bounds and three changed/poisoned graph replays. All passed;
peak GPU memory was about 69 MiB. Metadata was prepared outside capture, as in
serving; the first fixture incorrectly captured it and was corrected. The
adapter changes admission only, not metadata arithmetic or sampling, and is
restricted to the source-pinned confidence candidate. Native confidence K5
required a second compatibility change: native adaptive verification forced
piecewise prefill graphs, while this runtime uses eager prefill and full decode
graphs. The confidence-only adaptation preserves that existing graph policy
inside the composed DCP cache initializer. CPU preflight passed 111 native
variable-length dispatch cases, with 18 full target descriptors and no piecewise
graphs. An additional source-pin packaging error was caught before weights
loaded and fixed; freezing and CPU preflight now verify every backend source
hash. The earlier v4 candidate completed graph capture (1.82 GiB) and native
cost profiling with the unchanged 3,313,955-token KV allocation, but the final
warmup failed a DCP slot-mapping buffer check. It never served a benchmark
request, so that version supplied no throughput or quality result. Neither
worker was OOM-killed. The native manager supplies a padded max-C+1 query-start
buffer, while the DCP hook requires exact active-C+1 geometry. A confidence-only
zero-copy prefix view preserves the hook's existing exact-shape check. The v6
candidate passed actual slot-kernel tests on both GPUs at C1–C6: old failures
reproduced, independent eager reference, three changed-input/poisoned graph
replays, replicated SWA and disabled-ring handling. Peak GPU allocation was
44,032 bytes. The same v6 candidate passed all 26 flattened-metadata cases per
GPU again (about 69 MiB peak). Full-model v6 then passed final warmup and served
requests: seven recorded T0 results ranged from 23.41 to 38.55 tok/s. However,
dgx0 headroom repeatedly fell below the unchanged 1-GiB request-admission floor.
The serial benchmark stopped at that check; prefill and C6 were not measured.
The workers remained healthy, not OOM-killed. This candidate is rejected for
insufficient headroom at the current limits, not presented as a qualified win.
These optional settings alone do not prove work is skipped; the comparison
must inspect useful throughput. Native scheduler counters count
scheduled drafts rather than the confidence-trimmed worker budget, so a separate
worker-side measurement is required to establish how much confidence pruning
actually occurs. The v6 candidate logs bounded aggregate worker-budget counters
without reading GPU data, changing policy or logging request contents. In the
live v6 run, a sampled aggregate at 1,408 calls showed 7,065 scheduled drafts
and a 3,133-draft verification budget (includes startup): actual trimming was
confirmed. This is not a complete per-request acceptance denominator.
Scheduler counters do reflect the EMA scheduler's shorter proposed prefixes.
EMA only trims verification; the backbone still generates the full K5 block.

Handoff remaining: check normal serving on the retained K3 runtime, leave it
healthy on port 8888, and publish the reviewed recipe. Fixed and
combined K3/K4/K5 and both adaptive policies have now been exercised; rejected
variants remain rejected rather than silently weakening their memory limits.

## Benchmark scope and inputs

Serial requests use `max_tokens=400`, `stream=true`, no thinking, temperature 0
with seeds 41 and 1729, then temperature 1 with seed 41. `top_p` is 1 at T0 and
0.95 at T1. Each request must have exactly one matching delta in the native
prefill/decode/token counters and leave the engine idle. Decode rate excludes
the first token and uses native decode duration. Reported summaries are medians
within each content/temperature group, then an eight-group geometric-mean
speed ratio. They are not pooled end-to-end latency or confidence intervals.

Exact English prompts (the missing space in `least800` is intentional here to
preserve the actual inputs):

- Garden: `Write a detailed description of a quiet garden in the morning. Use complete sentences and include sensory details. Write at least800 words.`
- Python: `Write a complete Python implementation of an LRU cache with get and put methods, type hints, and unit tests. Explain its invariants and time complexity. Include enough code and explanation for a tutorial of at least800 words.`
- Explanation: `Explain how a relational database executes a SQL query, from parsing through planning, indexes, joins, execution, and result delivery. Use a concrete example throughout. Write a detailed tutorial of at least800 words.`
- Easy prose: `Explain photosynthesis to a curious ten-year-old. Use plain complete sentences and familiar examples. Write at least 800 words.`

C6 sends those four prompts plus Spanish and Chinese prompts concurrently,
256 output tokens each, seed 41, one wave per temperature. The additional
prompts are `Describe en español cómo preparar un huerto pequeño. Escribe al menos 800 palabras.`
and `请用中文详细解释水循环，至少写八百字，包括蒸发、凝结和降水。`.
Aggregate throughput is all six requests' completion tokens divided by the
whole wave's elapsed time. Client decode estimates use streaming first/last
content timestamps and must not be confused with isolated server timings.

Each startup performs normal runtime warmup, but does not prewarm every sampled
benchmark signature; first-use sampling JIT warnings were observed. Background
NFS archival remained active. The repeated unchanged baseline demonstrates
run-to-run variation. These synthetic inputs test performance/acceptance and
basic functionality, not perplexity, multilingual quality or general accuracy.

Full credit to MiaAI Lab / Wesley Young and contributors for the cooperative
MoE foundation. Those parts and adaptations are AGPL-3.0-only; original
MIT/ExLlamaV3 notices are retained. Native vLLM retains Apache-2.0 notices.
