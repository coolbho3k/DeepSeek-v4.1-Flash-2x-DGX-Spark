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
