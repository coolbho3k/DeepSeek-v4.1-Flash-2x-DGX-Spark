# Selectable NVFP4 four-over-six main KV

The main-cache writer defaults to `nvfp4_4over6`. Use
`./start-server.sh --restart --fp4-kv-mode legacy` to select the previous
quantization, or set `DS41_FP4_KV_MODE=legacy` in `.env.ds41` before restarting.
Both ranks receive the same mode. A source update alone does not change an
already-running worker or a separately frozen deployment kit.

## Comparison with DeepSeek and the previous cache

[DeepSeek V4.1 Flash, section 2.4.4](https://arxiv.org/html/2609.19969v1#S2.SS4.SSS4)
specifies post-RoPE E2M1 main KV with an E4M3 scale per 16 values and no
second-level global scale. It retains FP8 SWA KV. Its
[reference quantizer](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/inference/kernel.py)
uses `amax / 6`. Our previous main-cache implementation already used this
format and scale rule; it was already 4.5 bits per value.

| Property | Paper/reference | Previous cache | New default |
| --- | --- | --- | --- |
| Main KV values/scales | E2M1 / E4M3, group 16 | Same | Same |
| Scale selection | Reference: rounded `amax/6` | Rounded `amax/6` | Lower error of rounded `/6` and `/4` |
| Main state, 512 values | 288 bytes | 288 bytes | 288 bytes |
| Main KV quantization | After RoPE | After RoPE | After RoPE |
| Main KV storage | Paper discusses HBM | Display memory | Display memory |

The paper's 890 global KV bytes per input token includes shared main and
indexer caches across the architecture. Our existing layout has that same
raw global budget: `(288 + 68) * (3/2 + 1) = 890`, or 445 per DCP2 rank.
This excludes bounded SWA, compressor rings, page padding, and other runtime
allocations. Four-over-six changes reconstruction quality, not capacity.
SWA separately defaults to FP8 group 32 with BF16 RoPE, with group 64 optional.
See the [SWA comparison](../swa_kv/README.md): the coarser original group 64 was
a precision disadvantage, while its BF16 RoPE tail was more precise than the
reference FP8 tail. Neither implementation is byte-identical to every paper cache.

## Scale selection and fast writers

The scale search follows the sibling GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark
implementation's `overlay-dflash2/patch_b12x_nvfp4_four_over_six.py`:

1. Keep the existing group-16 `/6` candidate, including its scale floor.
2. Encode another candidate using an E4M3-rounded `/4` scale and E2M1 values.
3. Compare squared errors of the actual reconstructions, including scale rounding.
4. Store `/4` only for strictly lower error; ties retain the exact `/6` bytes.

The `/4` scale saturates at 448, preserving finiteness at the upper E4M3
boundary. The outer scale is implicitly one. For the finite BF16 inputs
supported by the original writer, choosing between both candidates cannot
increase any group's reconstruction SSE. All-zero and exactly representable
groups can tie, so strict improvement is an aggregate property.

Both ordinary and fused RoPE insertion use the same quantizer. Native
Blackwell E2M1 conversion replaces the software threshold tree. Reciprocal
multiplication followed by FP16 rounding preserves nearest-even FP4 bins for
BF16/E4M3 ratios, including exact ties (exhaustively checked below).
The SSE comparison uses an exact integer difference, avoiding FP32 tie
ambiguity and the first implementation's expensive conditional FP64 path:
`sum((q4-q6) * (q4+q6-2*abs(x)))`, in power-of-two integer units.
Where the reconstructions differ, all terms are exactly representable in
these units and the group reduction fits int32. Equal errors keep `/6`.

Fused RoPE preserves native BF16 rounding before quantization, skips invalid
or unowned slots and incomplete CR2 groups, and writes directly into existing
pages. Decode uses four groups per program and one warp; intermediate batches
use 16 groups and one warp; large prefill uses 32 groups and one warp (two for
CR2). These choices were measured on GB10; no runtime autotuning, extra launch,
persistent GPU tensor, or temporary rotated tensor is added. Legacy mode
compiles out the second candidate and receives the native conversion speedup.
Four-over-six itself leaves the main-cache ABI and its readers unchanged.

## Display memory and speed

Each state remains 256 value bytes plus 32 scale bytes: **exactly 4.5 bpw**.
The production allocator and native library remain unchanged: **1792 MiB of
display memory per rank, zero ordinary-memory KV backing** in either mode.
Indexer MXFP4 and compressor ring formats are unchanged. The separate SWA32
change also fits the existing page size; 4.5 bpw describes main KV, not the
complete heterogeneous pool.

[Recorded timings](performance-results.json) use a separate 4 MiB real display
allocation, never the running server's KV. The probe builds the same display
allocator source with only its required size changed for this bounded check.
It unregisters and frees the allocation afterward. Every writer variant
produced its expected bytes and preserved canaries in the strided pages.

GB10 median microseconds per fused RoPE/write; 128 nodes per CUDA graph,
3 warmups and 21 timed replays, shuffled variant order each round:

| Rows / path | Previous `/6` | Initial four-over-six | Optimized four-over-six | Optimized legacy option |
| --- | ---: | ---: | ---: | ---: |
| 1, CR1, all live | 1.334 | 1.733 | 1.305 | 1.184 |
| 8, CR1, all live | 1.337 | 1.736 | 1.317 | 1.192 |
| 24, CR1, all live | 1.324 | 2.649 | 1.337 | 1.221 |
| 2048, CR1, all live | 6.457 | 13.639 | 5.947 | 3.440 |
| 2048, CR1, DCP-owned | 4.299 | 8.443 | 4.296 | 2.905 |
| 2048, CR2, DCP-owned | 3.335 | 5.768 | 3.336 | 2.425 |
| 3072, CR1, DCP-owned | 5.814 | 11.294 | 5.974 | 3.584 |

`legacy_before` measures the initial implementation with four-over-six
compiled out and its original launch geometry. Its bytes were separately
verified against the original writer. The table shows approximate parity
with the previous path at decode and DCP prefill sizes; it also includes the
3072-row case that is 2.8% slower. The new writer substantially improves on
the first four-over-six implementation, but still does more work than the
newly optimized legacy option. All measured variants have zero register spills.
These are cache-writer measurements with another server resident, not claims
of end-to-end token throughput or universal optimality.

## Accuracy validation

The [GPU probe](../../runtime/probes/check_nvfp4_four_over_six.py) compares both
modes against an independent nearest-code oracle with FP64 SSE and the exact
previous GPU writer. It checks strided pages, canaries, slot masking, ties,
signed zero, scale boundaries, native RoPE parity, CR1/CR2 and DCP ownership,
CUDA graphs, packed attention, and the public 2048-row prefill bound.

[Results](gpu-results.json): 63,936 groups, 25,432 strictly improved, **zero
regressed**. Legacy and ordinary `/6` NVFP4 match the previous GPU writer
byte for byte. Reconstruction SSE reductions are 16.64% for normal and
trained-range synthetic data, 16.62% across wide scales, and 0.59% for outliers.
The synthetic attention output SSE against original BF16 KV fell 14.11%.
All 20 native RoPE cases and 2048-row public stores passed. Peak test tensor
allocation was 42.60 MiB. The overall SSE total is dominated by a deliberately
extreme scale-boundary fixture; use the per-distribution figures above.

The separate [rounding probe](../../runtime/probes/check_nvfp4_rounding.py)
checked **4,461,660** signed BF16/E4M3 combinations, covering every finite
positive E4M3 scale and every BF16 magnitude through the format's 2688 bound,
including signed zero, subnormals, and exact midpoints. Its independent
float64 nearest-code oracle matched native conversion for every combination.
[Results](rounding-results.json) pin the same shipped codec.

These checks establish reconstruction accuracy and synthetic attention
accuracy. The paper used quantization-aware training; lower inference-time
cache SSE alone does not establish improved perplexity, retrieval accuracy,
or model benchmark scores. A full-model quality A/B remains unperformed.

## Restarted serving check

On 2026-09-22, the existing two-rank deployment was restarted with this codec
and `nvfp4_4over6` on both ranks. Loaded-backend checks and CUDA graph capture
passed. Both workers registered 1792 MiB of display KV with zero ordinary KV.
The API is on port 8888. [Serving results](serving-results.json) record nine
successful checks: arithmetic, uncached 6036-token retrieval spanning multiple
prefill chunks, a 63-token explanation, and six concurrent arithmetic requests.

The immutable deployment is a copy of the previous kit with these cache/source
updates. Its old host helper also received the public launcher's existing
`MemAvailable` accounting, retaining the same startup reserve and idle-GPU
requirement. Reclaimable file cache had tripped its older `MemFree` check.
Existing images, weights, native libraries, 2048-token prefill budget, C6,
and memory limits were retained. These are local serving checks, not a fresh
public installation or a controlled model-quality/throughput A/B.

## Reproduction

Reproduce inside the installed serving image with this runtime at `/work`:

```bash
python3 -B /work/probes/check_nvfp4_four_over_six.py --output /results/accuracy.json
python3 -B /work/probes/check_nvfp4_rounding.py --output /results/rounding.json
python3 -B /work/probes/bench_nvfp4_kv.py --selected --output /results/speed.json
```

The accuracy probe accepts `--baseline-codec` for the original writer (recorded
SHA256 `d06ce6d2431c44c562e0fe85e7094c6db9aacd2185153473b98050ec327deea4`).
The speed probe accepts `--baseline-runtime` for the initial four-over-six
runtime and `--display-library` for the bounded allocator built by
[build_nvfp4_display_probe.py](../../runtime/probes/build_nvfp4_display_probe.py).
The probes load no model weights.
