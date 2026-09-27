# Draft context KV was read in the wrong SWA layout (2026-09-25)

## Symptom

At identical anchors, a clean PyTorch DSpark (DeepSeek's reference math, our
captured target features) got 40–50% more drafts accepted than serving. Serving's
draft logits correlated only 0.73 with the reference on serving's own top-32.

## Localization (debug capture kits v5–v9, probes in this directory)

- Target features, anchor tokens, EXL3 draft experts, context positions/window,
  block non-causality, attention sinks, softmax scale and TP head slicing all
  matched (`probe_*.py`).
- Layer-0 draft attention input matched serving exactly (cos 1.0000); its output
  did not (cos 0.68). Rank-local q and the `wq_a|wkv` projection matched
  (0.9998–1.0); the divergence was inside attention given correct q.
- Rewriting the context window through serving's own insert path changed nothing
  (cos 1.0 before/after): the stored bytes were reproducible, so the fault was
  in how they were read.

## Cause

`ds41/swa_kv.py` (default `swa_kv_group_size=32`) replaces
`DeepseekV4Attention._fused_qnorm_rope_kv_insert` with a group-32 FP8 writer and
decodes SWA pages as group-32. DSpark's context KV is written by the module-level
`_insert_context_kv` in `vllm/models/deepseek_v4_1/nvidia/dspark.py`, which calls
the native group-64 writer directly. The reader then applies group-32 scales to
group-64 pages, corrupting every draft layer's context. Block (query) KV goes
through the patched attention writer, so it was stored correctly; the target is
unaffected.

## Measured (port 8889, `ngram_draft/bench.py`, same-day K3 control)

`swa_kv_group_size=64` (reader and writer agree): geomean **+12.8%** over all 16
cases; DSpark-driven cases +14–26% (easy prose T0 1.99→2.42 tokens/step, python
2.48→3.09), prompt-lookup cases ~flat. Tokens/step now match the offline port's
prediction (2.33, 3.01). Evidence: `reports/swa64-bench-v1.json`.

## Fix options

1. `swa_kv_group_size=64` (config only; the pre-group-32 production default).
2. Keep group-32 for the target and route `_insert_context_kv` through the
   group-32 writer (`swa_kv.load_native()`), in `combined_dspark`'s existing
   compiled `_insert_context_kv` replacement. Needs a benchmark against (1).
