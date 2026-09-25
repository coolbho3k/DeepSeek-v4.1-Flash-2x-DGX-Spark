# DSpark context KV was written as group 64 and read as group 32 (fixed 2026-09-24)

## Cause

Group-32 SWA (`1c1c7eb`) replaced the target attention writer
(`DeepseekV4Attention._fused_qnorm_rope_kv_insert`) and every SWA reader. The
DSpark drafter writes its context KV through vLLM's module-level
`_insert_context_kv`, which calls the native group-64 op directly. Draft layers
then read group-64 pages with group-32 scales. Draft block KV and the target
were unaffected, so outputs stayed exact; only acceptance fell.

`serving/ds41/combined_dspark.py` already recompiles `_insert_context_kv`; with
group 32 it now also routes the native call through `swa_kv.load_native()`, the
same writer target attention uses.

## Why it was not noticed

Tokens/step fell 13–16% on 2026-09-22, but decode steps got about 15% cheaper
the same week (vocabulary row cache, packed projection reuse, fastcomm), so
serial tok/s looked flat (easy prose 31.0 tok/s at 2.32/step on 09-18, 30.8 at
1.99/step on 09-24). It was found when an offline PyTorch DSpark accepted
40–50% more drafts than serving at identical anchors.

## Measured (`ngram_draft/bench.py`, K3, port 8888, same kit otherwise)

T0 tokens/step:

| | garden | python | explanation | easy prose |
|---|---:|---:|---:|---:|
| 09-18, group 64 (`dspark-k3-control-v2`) | 2.01 | 2.85 | 2.79 | 2.32 |
| 09-24 control, group 32 with bug | 1.76 | 2.48 | 2.35 | 1.99 |
| group 64 workaround | 2.03 | 3.09 | 2.75 | 2.42 |
| group 32 with this fix | 2.03 | 3.00 | 2.96 | 2.35 |

Decode geomean over all 16 cases: **+11.2%** versus the 09-24 control
(DSpark-driven cases +13–23%, prompt-lookup cases about flat) and −1.4% versus
the group-64 workaround, within the ~±3% per-case run-to-run spread; easy prose
trailed group 64 by 7–8% in this single run. Evidence:
`context-fix-bench-v1.json`.

Deployed kit: `prepare_context_fix_kit.py` over `ds41-runtime-ngram-draft-v3`
(manifest `b9c163a0…`).
