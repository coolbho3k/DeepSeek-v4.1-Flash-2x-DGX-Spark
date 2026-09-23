# Prefill campaign, 2026-09-23

## Where a 2048-token chunk goes

`profile_prefill.py` captured the 9th step of a 27,710-token uncached prompt
(context ≈16–18K) on both ranks, shapes recorded, no stacks. The chunk took
1.945 s (≈1,053 tok/s); the GPU was busy 1.85 s.

| Component | ms per chunk |
|---|---:|
| DCP attention exchange (305 all-gathers, FP32 32-head partials + LSE) | ~434 (per-call min, both ranks equal: transfer, not waiting) |
| Hidden-state all-reduce (81 × 2046×5120 BF16) | ~89 (min) / 116 |
| Grouped-prefill MoE (thin MUL1 + MiaAI fat gate/up + down) | ~480 |
| Attention kernels | ~300 |
| mHC (post, pre, TF32 prenorm GEMM) | ~150 |
| Dense FP8/BF16 GEMMs | ~150 |

The MoE streams every expert once per chunk (≈390 ms floor per rank at the
measured ~262 GB/s), so it runs at ~81% of that floor.

A two-Spark NCCL microbenchmark (`../nccl_small/`, `--prefill`) shows the
link delivers ~20 GB/s for 4–20 MB messages whatever the protocol setting:
serving defaults 1.04 ms per 20 MB all-reduce, Simple 0.98 ms, 4 MiB buffers no
better, forced LL 3.3 ms. The `RING_LL` kernel name does not indicate the
protocol used. Prefill communication is bound by bytes on the wire.

## Candidates measured on port 8889

Uncached, unique-nonce retrieval prompts (`bench_prefill.py`), isolated server
prefill counters, all retrievals correct, zero prefix hits. First 8K request
of each boot is a cold warm-up.

| Configuration | 8K tok/s (warm) | 32K tok/s | Head RAM min |
|---|---:|---:|---:|
| Control: 2048-token chunks | 1,165 | 1,119 / 1,154 | 3.02 GiB |
| 3072-token chunks | 1,136 | 1,100 / 1,126 | 2.57 GiB |
| Grouped-prefill FAT_MIN 16 → 2 | 1,156 | 1,103 / 1,158 | 3.01 GiB |

Neither candidate helps; 3072 chunks are ~2% slower and cost ~0.45 GiB of
host headroom. Control warm C6: 53.5 / 53.7 tok/s. No serving change.

The remaining large lever is the attention exchange payload: it already sends
only the peer's 32 heads, but in FP32 (≈33.6 MB per rank per 512-row call).
BF16 partials with an exactly preserved FP32 LSE would halve it (≈ −215 ms per
chunk, roughly +12% prefill) at the cost of rounding the peer's partial to
BF16 before the FP32 merge — a numerical change that needs explicit approval
and quality checks.

## BF16 result exchange (promoted to serving and the public recipe)

`edits-bf16-exchange.json` changes only the concurrent DCP result exchange:
the peer's 32-head partials are rounded to BF16 (round-to-nearest-even) and
the FP32 LSE travels bit-exactly in the record's last two BF16 slots (514 per
record instead of 513 FP32). Local heads and all merge arithmetic stay FP32;
the query exchange is unchanged. Result bytes per 512-row slab: 33.6 → 16.8 MB.

Prefill (port 8889, retrieval correct, zero prefix hits):

| | 8K tok/s | 32K tok/s |
|---|---:|---:|
| Control | 1,165 | 1,119 / 1,154 |
| BF16 exchange | 1,203 / 1,206 | 1,181 / 1,219 |

≈ +3.5% at 8K and +5.5% at 32K.

Quality (`quality_eval.py`, `compare_quality.py`): six 6,144-token calibration
documents scored teacher-forced with `prompt_logprobs`, plus 12 greedy replies.

| Comparison | Mean abs Δ logprob | Top-1 agreement | Perplexity change |
|---|---:|---:|---:|
| Control fresh restart vs control | 0 | 100% | 0 |
| Control, 6 documents batched vs serial | 0.038–0.120 | 93.6–98.7% | −0.5…+0.2% |
| BF16 exchange vs control (serial) | 0.042–0.117 | 93.7–98.6% | −0.9…+0.4% |

The control is bit-deterministic for identical batching; the BF16 exchange's
perturbation is the same size as ordinary batch-composition variation already
present in serving, and unbiased (mean Δ logprob +0.0016). Greedy replies: 1 of
12 identical (divergence starts after the first differing token, as with any
perturbation). Local kit `35cb463a…94d5` serves on port 8888; the public
`release/runtime` carries the same three overlay files (recipe lock updated).
`packed-supersession.json` records that `packed.py` differs from the SWA
campaign's evidenced source only by the added `to_wire` kernel; the SWA test
verifies that exactly.
