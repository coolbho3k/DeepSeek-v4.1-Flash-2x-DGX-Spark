# Temperature-one draft sampling results

Selected as the serving default: native probabilistic DSpark drafting, fixed
K3, standard rejection sampling. Both the public launch profile and the local
server select this mode. `prepare.py` reproduces the candidate from a verified
greedy-draft parent.

The installed API default temperature is 1.0. Both sides use top-p 0.95, seed
41, the same prompts, weights, KV formats and limits. Per user instruction,
reuse the last model-fusion baseline; no new baseline requests were sent.

| Serial case | Saved greedy tok/s | Probabilistic mean tok/s | Change |
|---|---:|---:|---:|
| garden | 22.62 | 24.31 | +7.5% |
| python | 27.54 | 29.94 | +8.7% |
| explanation | 28.27 | 29.28 | +3.6% |
| easy_prose | 26.97 | 25.80 | -4.3% |

Each candidate case was run twice; serial replies were identical between candidate repeats. Median of the four paired throughput ratios: **+5.5%**.

| Six-request wave | Aggregate end-to-end tok/s | Change vs saved baseline | Draft acceptance |
|---|---:|---:|---:|
| Saved greedy baseline | 51.79 | — | 30.58% |
| Candidate first use | 45.67 | -11.8% | 27.53% |
| Candidate warm 1 | 53.25 | +2.8% | 28.46% |
| Candidate warm 2 | 53.68 | +3.7% | 28.57% |

The two warm waves average **53.47 tok/s (+3.2%)**. The first wave is retained above; its initial latency was materially higher, but the precise cause was not isolated. Do not report only the warm result as if every trial improved.

Acceptance improved for the Python and SQL serial prompts, but decreased for
the garden, prose and six-request cases. The result does not establish that
probabilistic drafting universally improves acceptance or speed. Different
continuations change routing, draft difficulty and work per generated token.
These are descriptive performance measurements, without statistical confidence
or a new model-quality/distributional-correctness claim. Temperature zero and
long-prefill performance were not measured in this trial.

KV capacity remained 3,313,955 aggregate tokens, backed by 1,792 MiB of display
memory per rank. Four-over-six NVFP4 main KV, group-32 FP8 sliding KV with BF16
RoPE, C6, 2,048-token prefill batches and 0.92 utilization were preserved.

Evidence: `reports/draft-sampling-v1/{preparation-v2,candidate,candidate-repeat,
candidate-c6-repeat,comparison}.json`. Baseline:
`reports/model-fusion-v1/{final-serial,final-c6}.json`.

Candidate kit manifest:
`66cc4fb87e96d9816d7c1ebb6fd8b6109e7dc3e423b1392bb2a72036d7a6aaa1`.
The only payload changes are the sampling selection in the two launch-profile
copies and the serving overlay manifest. Runtime integrity was verified on
both hosts.
