# Prompt-lookup drafting on top of DSpark (2026-09-23)

`ngram_draft.py`: after DSpark proposes K=3 tokens, a GPU kernel searches each
request's token history (last 64K tokens) for the most recent earlier
occurrence of its last 3 tokens; if found, the 3 tokens that followed replace
DSpark's proposals and their cached draft distribution becomes a point mass.
Probabilistic rejection sampling therefore stays exact for the target model.
The wrapper is installed inside the reviewed DSpark patch set (a later rewrap
is correctly refused by the startup binding check); the worker's load hook only
attaches request-state access.

Candidate measurements (port 8889, 400-token serial requests, `bench.py`):

| Case | T | Tokens/step | Acceptance | Decode tok/s |
|---|---:|---:|---:|---:|
| code_edit | 0 / 1 | 3.44 / 3.47 | 0.82 / 0.83 | 47.6 / 44.2 |
| json_rename | 0 / 1 | 3.00 / 3.00 | 0.67 / 0.67 | 44.1 / 44.0 |
| typo_fix (36-token reply) | 0 / 1 | 2.92 / 2.92 | 0.64 / 0.64 | 42.8 / 42.9 |
| quote_explain | 0 / 1 | 2.38 / 2.38 | 0.47 / 0.47 | 34.6 / 34.2 |
| garden | 0 / 1 | 1.76 / 1.73 | 0.26 / 0.25 | 27.9 / 27.0 |
| python | 0 / 1 | 2.48 / 2.48 | 0.50 / 0.49 | 37.6 / 37.1 |
| explanation | 0 / 1 | 2.35 / 2.25 | 0.45 / 0.42 | 35.2 / 33.7 |
| easy_prose | 0 / 1 | 1.99 / 1.81 | 0.33 / 0.27 | 30.7 / 28.1 |

Plain DSpark on free text averages ~1.7–2.1 tokens per step. Garden and easy
prose (same wording as the fastcomm run) at T=1: 27.0 vs 26.7 and 28.1 vs
28.1 tok/s — no prose regression. A matched control on the copy-heavy cases
was skipped at the maintainer's request; their speedup (estimated +30–60%)
is not separately measured.

Promoted: kit `9fdb4cad…06d6` on port 8888; the public `release/runtime` carries
the same module and hooks (byte-identical), recipe lock updated, all 226
repository tests pass.
