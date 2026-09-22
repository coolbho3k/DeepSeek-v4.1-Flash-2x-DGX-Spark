# Group-32 FP8 sliding-window KV with BF16 RoPE

The default is now **32/BF16**: an independent UE8M0 power-of-two scale per
32 of the 448 non-RoPE FP8 values; all 64 RoPE values remain BF16. Select the
original layout with `./start-server.sh --restart --swa-kv-group-size 64`, or
set `DS41_SWA_KV_GROUP_SIZE=64` in `.env.ds41`. Select `32` to switch back.
The independent main-cache default is `nvfp4_4over6`; `--fp4-kv-mode legacy`
restores its old quantizer. Both choices reach both workers before allocation
and graph capture, and require a restart.

## What differs from DeepSeek's reference

[The paper, section 2.4.4](https://arxiv.org/html/2609.19969v1#S2.SS4.SSS4)
keeps FP8 sliding-window KV because it is sensitive to quantization.
The [reference model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/inference/model.py)
uses groups of 32 across all 512 post-RoPE values. Our previous vLLM cache
used groups of 64 for 448 non-RoPE values, plus an unquantized BF16 RoPE tail.
The coarser group was the remaining precision disadvantage: a larger nearby
value can force a scale that loses small values to FP8 subnormal rounding.
Keeping BF16 RoPE is more precise than quantizing those values to FP8, so the
old cache was not uniformly less precise than the reference.

| Property | DeepSeek reference | Previous / optional 64 | New default 32 |
| --- | --- | --- | --- |
| Non-RoPE values | FP8, group 32 | FP8, group 64 | FP8, group 32 |
| RoPE values | FP8, group 32 | BF16 | Same BF16 bytes |
| SWA scale | Power-of-two UE8M0 | Same | Same, exact ceiling |
| Main cache | E2M1/E4M3 group 16, `/6` | Same | Four-over-six selection |

Ordinary normalized values often reconstruct identically under both power-of-two
scales: changing the exponent alone does not add FP8 mantissa bits. The smaller
group helps values near the subnormal/zero boundary, particularly when adjacent
halves have very different magnitudes. This is an error improvement, not a claim
of measured full-model perplexity or benchmark gains.

## Native fused writer and packed readers

The shipped native library adapts the pinned Apache-2.0 vLLM kernel, preserving
its Q normalization, RoPE arithmetic, single launch, PDL support and existing
decode/prefill dispatch. Group 32 reduces over two lanes instead of four and
writes 14 scale bytes plus two padding bytes. Integer exponent extraction
computes `ceil(log2(amax/448))` exactly; approximate `log2f` can round an exact
boundary upward before `ceil`. No extra quantization launch or persistent
ordinary-memory KV copy is introduced. Group 64 keeps the original native writer.

Two-pass attention, online prefill, online decode, DCP packed-output attention,
eager selected-row gather and prefix-retention validation accept both layouts.
Scale addressing specializes at compile time. Q, attention arithmetic, sparse
selection and image visibility are preserved. New Triton variants compile on
first use; full-model graph capture then uses the selected format.

## Memory and padding

The **1792 MiB display-memory pool per rank and zero ordinary KV backing** are
unchanged. Four-over-six main KV remains exactly **4.5 bits per value**. SWA has
higher precision and is a separate, bounded cache.

| Per 32-token SWA page | 64/BF16 | 32/BF16 |
| --- | ---: | ---: |
| State bytes, including scale padding | 584 | 592 |
| Unaligned page bytes | 18,688 | 18,944 |
| Allocated page bytes, native 576-byte alignment | 19,008 | 19,008 |
| Trailing page padding | 320 | 64 |

The extra scales fit inside the old alignment padding, so this change adds no
page allocation. Removing the remaining trailing padding would save only 0.34%
of these SWA pages. The two unused scale bytes per state add another 64 bytes per
page; removing both would save 0.67%. These are SWA-page percentages, not total
KV-capacity gains. Native block-outer packing shares a stride determined by the
widest cache group; a smaller SWA page does not necessarily reduce that stride.
Changing packing would require separate allocator/reader and capacity validation.

## Validation

[GPU results](gpu-results.json) and [actual display-memory results](display-results.json)
check the native writer against an independent CPU float64 quantizer and the
original group-64 native operator. Coverage includes negative/DP-padded slots,
non-contiguous page strides, padded Q heads, zero/tie inputs, mixed magnitudes,
1–3072 rows, CUDA graph replay, complete-byte canaries, bit-exact Q/BF16 RoPE,
and all four packed attention paths against float64 dense attention.

All 11 cases and 12 reader comparisons passed. No tested 32-value group regressed.
The deliberately heterogeneous fixture improved 903 groups and reduced non-RoPE
SSE from `9.0475804e-6` to `2.0249607e-7` (97.76%); ordinary random inputs mostly
tied. The fixture is not an estimate of typical model-quality improvement.

Median native Q/RoPE/KV writer times on GB10, actual display backing, microseconds:

| Rows | Original 64/BF16 | New 32/BF16 |
| ---: | ---: | ---: |
| 1 | 1.514 | 1.497 |
| 8 | 1.701 | 1.647 |
| 24 | 3.087 | 3.073 |
| 128 | 11.279 | 11.309 |
| 512 | 143.090 | 145.746 |
| 2048 | 601.721 | 610.838 |
| 3072 | 903.827 | 909.193 |

These interleaved CUDA-graph microbenchmarks include Q processing. They are not
end-to-end throughput measurements. Decode is essentially unchanged; observed
prefill changes are under 2%. No global claim of optimality is made.

The native [build receipt](native-build.json), [builder](build_native.py),
[corresponding source](../../runtime/sources/swa32.cu), and
[upstream source/license](../../runtime/vendor/vllm-swa32-apache/UPSTREAM.json)
are included. Build inside the pinned serving image with:

```sh
python3 release/experimental/swa_kv/build_native.py --runtime release/runtime --output /tmp/swa32-build
```

The CPU suite verifies both defaults and alternatives, both-worker propagation,
source/binary hashes, source transformation, and the binary verifier's size bound.

The [local serving check](serving-results.json) passed after restarting both ranks
with four-over-six and 32/BF16 on 2026-09-22. Loaded-backend verification and graph
capture passed, with the same 16311 blocks / 3,313,955 reported KV tokens and
1792 MiB display-only pool per rank. All 11 requests passed: uncached 6036-token
retrieval, decode, six simultaneous arithmetic requests, a repeated prefix
(5888 cached tokens), and an earlier branch (4096 cached tokens). The staged
Git export passed all 212 included CPU tests. This reused existing verified
images/weights; a fresh public two-host installation and broad model-quality
A/B were not tested.
