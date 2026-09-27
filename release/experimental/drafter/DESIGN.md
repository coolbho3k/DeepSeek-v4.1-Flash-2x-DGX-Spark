# Retraining DSpark for our quantized target: design (2026-09-24)

Goal: raise draft acceptance on the model we actually serve (EXL3 3-bit experts,
FP4 KV, TP2/DCP2) without changing the target or its output distribution.
Speculative decoding is exact, so a better drafter cannot change quality; it
changes only tokens per step.

## Where we are

| Workload (bench.py, K3, T=0) | Tokens/step | Implied per-position acceptance | Decode tok/s |
|---|---:|---:|---:|
| Easy prose | 1.99 | 0.54 | 30.8 |
| Garden prose | 1.76 | 0.46 | 27.9 |
| Python | 2.48 | 0.69 | 37.6 |
| Code edit (prompt lookup) | 3.44 | — | 47.1 |

Step cost is ~64 ms at K3 and grows ~3.9 ms per extra draft position
(`../verify_cap/RESULTS.md`). Modelled payoff (tok/s, prose-like step):

| Acceptance | K3 | K5 | K7 |
|---:|---:|---:|---:|
| 0.53 (now) | 30 | 29 | 26 |
| 0.60 | 34 | 33 | 31 |
| 0.70 | 39 | 41 | 39 |
| 0.80 | 46 | 51 | 52 |

## Why we expect headroom

1. **Target mismatch.** DeepSeek trained DSpark on frozen original-backbone
   features and kept it aligned through post-training (tech report §2.4.3). We
   serve a 3-bit EXL3 requantization, and our drafter's routed experts are also
   requantized (EXL3 draft overlay, ~1 pp acceptance measured). Rough external
   evidence: MiaAI's original-weights target measured 2.17 tokens/step on prose
   at K3 versus our 1.99 (different prompts).
2. **Domain.** Our clients run thinking mode with multi-thousand-token outputs.
   The paper's recipe regenerates responses with the target; we can weight the
   data toward the model's own reasoning and code traces.
3. **Longer blocks.** The paper's gains grow with block length (math +16% at
   γ=7 to +30% at γ=15). Longer blocks only pay here if acceptance rises (table
   above), so this is Stage 2, gated on Stage 1.

The first gate measures (1) and (2) directly instead of assuming them.

## Decision: keep the DSpark architecture, retrain it

DSpark is the strongest published drafter for this family (paper, Qwen3-4B
acceptance length: chat 3.63 vs EAGLE-3 2.40 and DFlash 3.02; code 5.26 vs 4.20
and 4.44). It is also the only option with zero serving risk here: the loader,
graphs, DCP draft cache, EXL3 draft overlay, prompt lookup and the cap/EMA
experiments all assume its shapes. A new EAGLE-style autoregressive drafter
would mean K sequential drafter passes, new attention/graph integration and no
pretrained starting point.

Fixed shapes (drop-in): 3 blocks, hidden 5120, hc_mult 4, sliding window 128,
block size 5, target layers 37/38/39, Markov rank 256, confidence head,
128 routed experts top-3 + 1 shared, vocabulary/embedding/LM head shared with
the target.

## Stage 1 (most likely to succeed): fine-tune everything except routed experts

- **Initialize** from DeepSeek's DSpark weights (`mtp.0-2.*`).
- **Trainable (~0.7 B):** `main_proj`/`main_norm` (the adapter to target
  features, where most mismatch should land), attention, shared expert, routers,
  norms, hyper-connection mixers, Markov head, confidence head.
- **Frozen:** the 128×3 routed experts, held as bf16 dequantized from the
  *serving* EXL3 draft overlay (≈29 GB), so training sees serving numerics; the
  target's embedding and LM head (shared, never trained).
- **Why frozen experts:** full training is 14.6 B parameters (≈230 GB with Adam),
  which does not fit one Spark, and frozen experts mean the retrained drafter ships
  without requantizing experts. Non-expert weights are FP8 (E4M3, 32×32 E8M0
  blocks) in the checkpoint: requantize after training, with an FP8 fake-quant
  forward in the last epoch to match.

### Objective (DSpark paper, Eq. 12)

- Per draft position k = 1..5 with weight w_k = exp(−(k−1)/5):
  - 0.9 × total variation ‖p_draft − p_target‖₁ (acceptance for sampling is
    1 − TV, so this optimizes acceptance directly);
  - 0.1 × cross-entropy on the target's token;
  - 1.0 × BCE for the confidence head against c* = 1 − ½‖p_draft − p_target‖₁.
- The Markov head gets no separate loss; it learns through the factorized logits.
- Teacher forcing: positions after the anchor condition on the actual next
  tokens, which is exactly what an accepted prefix conditions on at serving time.
- Target distributions: store the top 64 of p_target per position plus tail
  mass; TV over stored support plus a tail bucket (bounded approximation).
- Recalibrate the confidence head afterwards (the paper's sequential temperature
  scaling) so the cap/adaptive policies stay usable.

## Data

### Two-pass collection

Both passes run on the unchanged serving model.

1. **Text pass (normal serving, speculation on):** generate responses to
   prompts. Store text only.
2. **Capture pass (prefill only, new capture kit):** prefill prompt + response
   and record, for every position, the exact tensor serving hands the drafter
   (aux hidden states of layers 37–39 after pooling, 3×5120 bf16) and the
   target's top-64 next-token distribution. vLLM normally computes logits only at
   sampled positions, so the capture kit runs the LM head on all positions
   (≈30 ms per 2048-token chunk). Rank 0 and rank 1 write alternate sequences to
   their own NVMe. Capture speed ≈ prefill speed, ~1,000 tok/s.

This keeps generation fast (production path) and capture simple (no decode
hooks). Record size ≈ 31 KB/token.

### Sources

| Set | Text from | Tokens | Collection time | Storage |
|---|---|---:|---|---:|
| On-policy | Open-PerfectBlend prompts (the paper's set: chat 18%, math 39%, code 39%, instruction-following 4%), thinking on and off, responses from our target | 3–5 M | ~5 h per 1 M tokens of generation (C6, ~55 tok/s) + capture | ~150 GB |
| Broad | Prefill-only over public reasoning/code/chat text; target distributions are still exact, only the conditioning text is off-policy | 10–15 M | ~3–4 h capture | ~450 GB (or ~230 GB with FP8 storage) |
| Held-out | 2% of each, split by prompt, plus the 8 bench prompts | — | — | — |

Client traffic is **not** logged or used unless you explicitly decide otherwise.
Disk: 1.3 TB free on the head, 1.6 TB on the worker.

## Training

- **Implementation:** a pure-PyTorch DSpark forward/backward ported from
  DeepSeek's reference `inference/model.py` (window attention, hyper-connections,
  MoE, Markov/confidence heads), bf16 with FP32 accumulation, blockwise training
  over whole sequences (every position is an anchor with its own 5-token query
  block, block-sparse mask).
- **Hardware:** both Sparks, data-parallel over the existing RoCE link, with no
  target model loaded (features are cached). Per Spark: frozen experts 29 GB +
  trainable state ~11 GB + activations. Gradient all-reduce ≈1.4 GB per step.
- **Budget:** ≈8 GFLOP per query token × 5 query tokens per anchor; 20 M anchors
  ≈ 8×10¹⁷ FLOP per epoch ≈ 2 h per epoch on two Sparks at an assumed
  ~60 TFLOPS each. Plan 3–5 epochs, checkpointed and resumable.
- **Optimizer:** AdamW, cosine decay, low learning rate for layers initialized
  from DeepSeek (fine-tune, not from scratch). The first training run is a
  learning-rate sweep on 1 M anchors.

## Evaluation and gates

1. **Offline evaluator = serving.** On held-out captures, the PyTorch drafter
   with *unchanged* DeepSeek weights must reproduce measured serving acceptance
   within ±2% (greedy: exact top-1 chains; sampling: expected accepted length
   from Σ min(p,q) chains). Proves features, masks and heads are right before any
   training claim.
2. **Pilot gate.** After one epoch on ~3 M anchors: held-out expected accepted
   length at K3 must improve ≥5% on prose or code, with no domain worse than −1%.
   If not, stop and report.
3. **Serving gate.** Export (same checkpoint layout, FP8 non-expert weights,
   experts untouched), load on port 8889, run `ngram_draft/bench.py` against a
   same-day K3 control. Promote only with a geomean gain beyond run-to-run noise
   (~±3% per case).
4. **Exactness:** greedy T=0 replies must match the control byte for byte, since
   the target is unchanged.

## Stage 2 (only if Stage 1 passes)

- Longer blocks (γ = 7–8) trained from the Stage 1 weights, served with
  DeepSeek-style confidence-scheduled verification: per-request length from
  confidence and measured step costs (`PrefixEMA.choose(costs=...)` machinery
  plus the cap). Needs the 36→56-row MoE capacity build and more graphs.
- Unfreezing routed experts: FSDP across both Sparks with 8-bit Adam, then
  requantize with the existing draft EXL3 pipeline
  (`scripts/run_draft_sparse_campaign.py`).

## Schedule and downtime

| Step | Needs GPUs? | Duration |
|---|---|---|
| Capture kit, PyTorch drafter, evaluator, unit tests | No | ~2–3 days of work |
| Night 1: capture pilot (1 M broad + ~0.5 M on-policy) and evaluator gate | Both, serving down | ~8 h |
| Night 2: pilot training + pilot gate | Both | ~8 h |
| Nights 3–5: full data generation, capture, training | Both | 3 × ~8 h |
| Night 6: export, 8889 benchmark, promote | Both | ~3 h |

Observed client traffic ran 04:00–20:00 PDT today, so nightly windows would be
roughly 20:00–04:00. Please confirm.

## Risks

- **Headroom smaller than hoped:** gate 2 limits the cost to two nights.
- **Off-policy broad data** could help less than on-policy data: the pilot
  compares both slices.
- **Reference port errors** (hyper-connections, window masks): gate 1 catches them.
- **FP8 requantization** of the non-expert weights costing acceptance: FP8
  fake-quant in the last epoch, and the offline check runs on the exported
  weights.
- **Confidence head miscalibration** after retraining: recalibrate; the cap/EMA
  policies depend on it.
