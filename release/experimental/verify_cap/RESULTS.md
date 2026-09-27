# Confidence-capped DSpark verification (2026-09-24)

Port of MiaAI Lab's `verify_cap` (from knapcio's TP4 fork) to our vLLM V2 stack:
`verify_cap.py`, installed by `prepare_kit.py` + `edits.json` on top of any
verified kit. Per request, k = leading drafts whose running product of the
DSpark confidence head stays >= 0.1 (min 1); prompt-lookup matches keep full K.
Rows after anchor + k take the anchor's experts in every target MoE layer, and
rejection sampling keeps at most 1 + k tokens (exact truncation). Rank 0's k is
broadcast. K5 kits reuse the Sep 18 36-row cooperative MoE build
(`release/experimental/dspark/prepare_candidate.py`, EMA anchor updated for the
prompt-lookup import).

Matched runs on port 8889, same day, `ngram_draft/bench.py` (8 prompts x T=0/1,
400 tokens, isolated per-request decode tok/s). Control = production kit
`b8399c02…` (K3 + prompt lookup), re-measured; it matched the Sep 23 numbers.

| Candidate | Geomean vs K3 | Code edit T0/T1 | Easy prose T0/T1 | JSON rename T0/T1 | Mean verified drafts |
|---|---:|---:|---:|---:|---:|
| K5 + cap 0.1 | -1.8 % | +24.8 / +12.7 % | -10.9 / -9.7 % | -7.4 / -10.0 % | 3.25 of 5 |
| K3 + cap 0.1 | -0.4 % | -1.5 / -0.8 % | +1.4 / +2.3 % | -1.7 / -1.1 % | — |
| EMA {3,5} + cap 0.1 | +0.7 % | +12.3 / +14.0 % | -4.4 / +5.0 % | -4.2 / -4.6 % | 3.21 of 5 |

Evidence: `reports/vcap-{control-k3,k5-t010,k3-t010,ema35-t010}-bench-v1.json`.

Findings:

- The cap works as designed: K5 + cap kept prose at K3's 1.99 tokens/step, and
  K3 + cap produced byte-identical greedy replies on all 8 T=0 cases.
- It cannot pay for a larger verify block. At equal tokens/step, K5 + cap's
  prose step was 72 ms vs K3's 64 ms: the extra cost is the 5-position drafter
  and 6-row attention/dense work, not the MoE rows the cap removes. MiaAI's
  prose-neutral result comes from a different cost profile (TP3, MXFP4 experts).
- At K3 there is little MoE to save; the effect is within noise.
- Content-adaptive block size (GLM recipe style: per-request acceptance EMA
  choosing a captured verify length from {3,5}) keeps most of the code gain
  and recovers much of prose, but loses on prompt-lookup-heavy JSON/typo cases
  (high acceptance selects length 5 at ~3.2 tokens/step). Net +0.7 %, inside
  the run-to-run spread (single cases moved up to 4-6 % between identical-text
  runs).

Decision: production stays on the K3 kit. None of the candidates is a
demonstrated general win; EMA {3,5} + cap is the option for code-heavy traffic.
Untried levers: EMA cost-table pricing (`PrefixEMA.choose(costs=...)`) so the
length choice uses measured step cost rather than acceptance alone, and capping
prompt-lookup requests at length 3 inside the EMA.
