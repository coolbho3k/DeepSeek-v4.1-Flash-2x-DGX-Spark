# "Other" decode kernels: profiled, nothing worth fusing (2026-09-23)

Source: paired C1 decode traces of the model-fusion candidate (both ranks,
two steps). Excluding MoE, dense FP8/BF16 GEMMs, LM heads and NCCL, the rest
is ~8.2 ms of kernel time over ~2,400 launches per step (~56 per layer).
Largest items: sparse decode attention `_online` 1.16 ms, mHC `_prenorm`
1.12 ms (13 µs × 86), mHC pre/post TileLang 0.78 ms, DCP gather 0.46 ms,
top-k 0.33 ms, indexer mask 0.31 ms; the BF16 split-K GEMM family (44/step)
is the MoE router [384, 5120] with FP32 output plus a cuBLAS split-K reduce.

Component A/B on GB10 (L2-defeating weight copies, CUDA graphs), `bench.py`:

- mHC prenorm with each FP32 weight tile reused across rows: bit-exact, but
  not faster (native 9.0–10.6 µs at K=20480; repeated row reads hit L2).
- Single-kernel router GEMV (no split-K reduce): ≤5% faster than cuBLAS
  (~19 µs isolated), FP32 results differ at ~1e-7 relative. Not worth a
  routing-order change.

In-model durations (13 µs prenorm, 31+6 µs router) are 30–70% above isolated
times because nine streams run concurrently inside the graph. The target
layer window is ~51 ms with only ~1 ms (2%) of GPU idle between kernels, so
launch-count fusion could recover at most ~1 ms. Estimated bytes streamed
per rank per target step (~9.7 GB: routed experts ~5.3, dense FP8 ~2.8,
wo_a ~0.6, LM head ~0.66, router + mHC ~0.3) imply a ~37 ms bandwidth floor
against ~51 ms observed. Remaining decode gains must reduce bytes per
generated token (acceptance, weight format) or cross-stream contention,
not kernel count. No serving change; evidence `reports/small-kernels-v1/`.
