# Indexer key format probe: MXFP4 vs NVFP4 four-over-six

A CPU-only probe run on 2026-09-26. Question: would storing the sparse-attention indexer's queries
and keys as NVFP4 with four-over-six scale selection (the main KV's writer) change which past
positions the indexer keeps, compared with the reference MXFP4?

## Method

- `project.py` follows the reference model (`source/<rev>/inference/model.py`) from captured
  block inputs (`capture-source-v1/states/LL`, 40 text records of 2048 tokens) to indexer
  queries, head weights and keys. It covers all eight index-source layers: 2, 8, 14, 20, 24, 28,
  32 and 36. Layers 24–36 use layer 20's keys.
- `analyze.py` scores `sum_h w_h * relu(q_h . k)` with the reference RoPE. It then applies each format
  to q and k after RoPE and compares each format's top-512 with the BF16 top-512.
  - **native**: each 2048-token record on its own, queries at positions 1024–2047.
  - **stitched**: the 8 records of one corpus concatenated to 16,384 tokens, queries in the last record.
    Records were captured independently, so this approximates long-context key statistics, not true
    long-range relevance.
- `depth.py` finds where in the BF16 ranking the swaps happen.
- `run.sh` reproduces everything, using the calibrator image with no GPU.

The quantizers were checked against brute-force nearest-value search. MXFP4 differs only on exact
ties, which it breaks nearest-even like the hardware. Four-over-six is never worse than `/6` on any group.

## Results

Recall of the BF16 top-512 set (higher is better). "Recovered" is the share of MXFP4's misses
that four-over-six removes.

| Layer | Stitched candidates | MXFP4 | NVFP4 /6 | NVFP4 4/6 | Recovered | Key-only NVFP4 4/6 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 7,680 | 0.868 | 0.888 | 0.898 | 23% | 0.912 |
| 8 | 7,680 | 0.906 | 0.924 | 0.929 | 24% | 0.938 |
| 14 | 7,680 | 0.903 | 0.922 | 0.927 | 25% | 0.934 |
| 20 | 15,361 | 0.908 | 0.925 | 0.930 | 23% | 0.942 |
| 24 | 15,361 | 0.861 | 0.884 | 0.894 | 24% | 0.907 |
| 28 | 15,361 | 0.910 | 0.926 | 0.931 | 23% | 0.939 |
| 32 | 15,361 | 0.917 | 0.932 | 0.937 | 25% | 0.947 |
| 36 | 15,361 | 0.921 | 0.936 | 0.941 | 25% | 0.947 |

- Native 2048-token contexts show the same pattern at higher recall (0.90–0.97), with 24–26% recovered.
- **No format dropped any BF16 top-64 or top-16 position, in any layer or context.**
- The swaps sit at the edge of the kept set. With MXFP4, the best-ranked dropped position has a
  median BF16 rank of 246–356; even in the worst 1% of queries it is rank 128–248. The deepest
  replacement kept instead sits at median BF16 rank 826–1,327.
- Four-over-six moves the best dropped rank about 35–50 places deeper and pulls the replacements closer.

### Query precision and group size

A follow-up run added FP8 queries (E4M3, one scale per 128 values, the format vLLM's FP8 indexer
query uses) and E4M3 key scales per 32 values:

| Layer 24, stitched | MXFP4 q+k | FP8 q, E4M3/32 keys | FP8 q, NVFP4 keys | BF16 q, NVFP4 keys |
| --- | ---: | ---: | ---: | ---: |
| Recall of BF16 top-512 | 0.861 | 0.896 | 0.906 | 0.907 |

FP8 queries are within 0.001 of BF16 queries in every layer. Most of the remaining gap comes
from 4-bit queries, not keys.

## Reading

- Four-over-six consistently removes about a quarter of the format-induced swaps when both
  queries and keys are 4-bit. With FP8 queries and NVFP4 keys, about a third are removed.
- Every swap is a borderline position traded for another borderline position; the strongest
  positions are unaffected.
- DeepSeek trained with MXFP4 queries and keys, so the model already tolerates this tail noise.
- Capacity is unchanged: vLLM pads each 64- or 128-state indexer page to a 512-byte multiple, and
  64 x 68 = 4,352 bytes already rounds up to 4,608 = 64 x 72.
- The expected quality effect is small at 16K tokens. It could grow at much longer contexts, where
  the candidate stage and far more competitors come into play. No end-to-end quality test was run.
- The serving implementation below goes one step further than this probe: instead of choosing
  between `/6` and `/4`, it searches every E4M3 scale in `[amax/6.5, amax/2.5]`. On real layer 2 and
  20 activations that lowers index-key reconstruction error from 0.747% (four-over-six with an MXFP4
  candidate) to 0.669% NMSE, against 1.37% for MXFP4. Recall was not re-measured with the search.

## Serving implementation (default for new deployments)

`DS41_INDEXER_K_FORMAT=nvfp4` (launcher flag `--indexer-k-format nvfp4`) selects NVFP4 keys.
Decode scores them against FP8 queries; prefill uses NVFP4 queries on the FP4 tensor cores. It is
the default for new deployments; `--indexer-k-format mxfp4` selects the previous keys. The mode requires the full-FP4 DCP route: FP4 main KV, the MXFP4 indexer
route and index-key parity. (An earlier build called this format `nvfp4_4over6`; that name is gone.)

**Scale selection (keys and NVFP4 queries)** is the main-KV writer's search
(`fp4_main_kv._search_scales`, also the main-KV default `nvfp4_search`). For each 16-value group:
- start from `E4M3(amax/6)`, DeepSeek's reference rule;
- try every E4M3 scale in `[amax/6.5, amax/2.5]` in ascending order (at most 12);
- a candidate replaces the current best only if its reconstruction SSE is strictly lower, compared
  exactly in integer units.

Every group is therefore no worse than `/6` or four-over-six, and ties keep the `/6` bytes. No bound
against DeepSeek's MXFP4 quantizer is claimed; the earlier MXFP4-scale candidate was dropped. On
the probes' synthetic data, key and query groups have 0.68–0.73x the SSE of `/6`, 0.85–0.89x that
of four-over-six and about 0.5x that of MXFP4. A few key groups (421 of 4,128) are worse than
MXFP4's, where its power-of-two block scale happens to fit better.

Prefill queries first take a power-of-two scale per token and head, folded into that head's
weight, which brings the head maximum to 1,024 or below. Every searched scale (at most
`amax/2.5`) is then a finite E4M3 value.

**Code:**
- `serving/ds41/nvfp4_indexer.py`:
  - the writer and the gathers;
  - the FP8 x NVFP4 decode scorers (eager and capture-safe);
  - the NVFP4 query quantizer and `QueryPackage`;
  - `prefill_logits`;
  - the native loader, which checks the binary hash against the receipt;
  - torch references.
- `kernels/nvfp4_indexer.cu` and `serving/nvfp4-indexer-native/` hold the prefill scorer.
  - It uses `mma.sync.m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X`, E2M1 x E2M1 with UE4M3
    scales, and follows DeepGEMM's SM120 MXFP4 thread mapping.
  - ReLU, head weights and the 32-head sum are fused.
  - Two query rows per CTA (128 registers, no spills).
  - Built GPU-free by `build_native.py`.
- `spark_indexer_k_math.py` sends FP4 index-key writes to the NVFP4 store after native RMSNorm.
- `vllm_dcp.py` does the following:
  - sets the 72-byte record;
  - switches indexer queries to vLLM's FP8 path;
  - returns a `QueryPackage` (pre-RoPE queries, quantized lazily per prefill chunk) in the unused
    `q_scale` slot;
  - recompiles `sparse_attn_indexer` with the NVFP4 gather, workspace, cache view, eager decode
    scorer and `prefill_logits`.
- `spark_combined_miaai.py` selects the capture-safe decode scorer.
- Pins in `overlay-manifest.json`, `spark_backend_attestation.py`, `spark_combined_miaai.py` and
  `runtime-requirements.json` were refreshed. `release/freeze.py` was re-run.

**Validation**, on one GB10 in the serving image, without the engine:
- `probes/check_nvfp4_indexer_gpu.py`: writer, RoPE, compress ratio 2, gathers and the FP8 decode
  scorer are exact or within 4e-6 of the float64 references.
- `probes/check_nvfp4_indexer_integration_gpu.py`:
  - vLLM's store entry point through the parity hook is byte-exact;
  - the eager and graph decode scorers over block-table pages match float64.
- `probes/check_nvfp4_prefill_gpu.py`:
  - key writer and query quantizer bytes are exact against the oracle;
  - no key group (0 of 4,128) and no query group (0 of 65,792) has error above `/6` or
    four-over-six;
  - FP4 tensor-core logits are within 1.9e-6 of float64;
  - the `QueryPackage` path matches the reference pipeline.
- Every query-quantizer launch geometry (1–16 heads per program, 1–4 warps) matches the reference
  byte for byte. An earlier join/permute construction of the query vector miscompiled with one
  head per program and two warps; the kernel now loads each head row and splits it into RoPE
  pairs, as the key writer does.
- Building `vllm_dcp.make_probe_patches()` against the image's hash-pinned vLLM applies every text
  anchor. The query package and prefill routing are bound. With the flag unset, the patch set is
  the original one.
- 264 CPU tests pass (`cd tests && PYTHONPATH=.. python3 -B -m unittest discover -s . -p 'test_*.py'`).

**Decode queries** (`DS41_INDEXER_DECODE_QUERY`, default `fp8`):
- `nvfp4` quantizes the `QueryPackage`'s decode rows exactly as in prefill and scores them with the
  same kernel.
- The kernel also splits keys across CTAs (`blockIdx.y`), so a few decode rows still fill the GPU.
  A `clean` mode writes -inf to the rest of each row, as the graph contract requires.
- Batches that need ragged padding keep the FP8 scorer.
- `probes/check_nvfp4_decode_gpu.py`:
  - the split kernel matches float64 under both output contracts;
  - the eager and graph NVFP4 decode scorers match float64;
  - the dispatcher routes and falls back correctly;
  - a real CUDA-graph capture under `GraphOwner`, replayed after pages and lengths changed in place,
    matches the reference.

**Speed**, measured while a quantization job shared the GPU. Prefill scoring, including query
quantization, as the median ratio against vLLM's MXFP4 route:

| Prefill shape | 256 x 32K | 1,024 x 64K | 2,048 x 128K |
| --- | ---: | ---: | ---: |
| NVFP4 route vs MXFP4, first measurement | 0.95 | 1.13 | 1.00 |
| Later, heavier contention: previous build (4/6 + MX) | 0.93–0.97 | 1.32–1.35 | 1.32–1.36 |
| Same session: search build | 1.01–1.05 | 1.33–1.48 | 1.43–1.49 |

Under contention the baseline itself varied 2x between runs, so these ratios bound the effect
rather than measure it; idle-GPU timing is part of the A/B gates. The search adds work only in
quantization: 8.7 vs 3.8 us to write 2,048 index keys, and 212 vs 183 us to quantize 2,048 x 32
prefill query heads (one call per indexer layer and prefill chunk).

Decode scoring, gather included (ms):

| Requests x rows, keys | FP8 queries | NVFP4 queries | MXFP4 DeepGEMM |
| --- | ---: | ---: | ---: |
| 1x4, 64K | 1.12 | 0.31 | 2.72 |
| 1x4, 512K | 3.11 | 3.03 | 4.90 |
| 6x4, 128K | 0.99 | 0.53 | 4.89 |
| 6x4, 512K | 11.9 | 9.4 | 43.3 |

A re-run with the search build, under heavier contention, kept the ordering at every shape with all
three paths up to 3x slower at the small shapes (1x4 64K: 2.45 / 2.41 / 2.74; 6x4 128K: 3.23 /
2.80 / 11.9; 6x4 512K: 11.8 / 9.4 / 43.7). The search changes no decode scorer; it adds about
1 us per decode query quantization and 0.5 us per index-key write.

At 512K both new paths are dominated by the per-request gather into a contiguous workspace. A
scorer that reads pages directly is the next decode optimization.

**Kit and A/B:**
- `prepare_kit.py` derives a runtime kit from a verified parent kit:
  - it replaces files the parent carries unchanged from git HEAD;
  - it applies reviewed edits where the parent differs;
  - it recomputes every pin, the overlay and bundle manifests, and the runtime requirements.
- `ds41-runtime-nvfp4-indexer-v2` (bundle `2f5c11fb...`, parent `b9c163a0...`) is verified on both
  hosts. It also carries the searched main-KV writer (`DS41_FP4_KV_MODE=nvfp4_search`). Importing
  its `serve.py` with the production worker environment runs the full registration chain:
  - the base deployment's settings (`nvfp4_4over6`, MXFP4 indexer): the parent's 13 DCP hooks;
  - `nvfp4_search` main KV: 13 hooks, search writer;
  - NVFP4 keys: 15 hooks, the NVFP4 writer and the NVFP4 graph scorer, with FP8 or NVFP4 decode queries.
- v1 (bundle `b61271ca...`) predates the search and the `nvfp4` name; do not use it.
- `model_fusion/launch.py` takes `--fp4-kv-mode`, `--indexer-k-format` and
  `--indexer-decode-query`. A dry run of its configuration path validates all four A/B modes
  against the kit.
- `ab_gates.py` runs control, kv-search, nvfp4-fp8 and nvfp4-nvfp4 on one deployment with the
  existing benchmark and probe harnesses. See [RUNBOOK.md](RUNBOOK.md).

**Served on two DGX Sparks** (2026-09-27, full-pool refit weights, kit v3, `nvfp4_search` main KV,
NVFP4 keys with FP8 decode queries):
- boot and CUDA-graph capture passed; text 3/3, diagnostics 2/2, held-out images 19/24;
- synthetic retrieval passed at 131,062, 524,279 and 1,039,993 prompt tokens (1M prefill 25.6 min),
  with at least 2.15 GiB host memory available on the tighter host during the 1M request;
- short-context decode matched control within noise (serial 28–46 tok/s, six requests 60–65 tok/s).

**Not yet done:**
- a KL A/B against an unquantized-indexer reference on long prompts;
- idle-GPU timing of the prefill route (measured only under a concurrent GPU job);
- a paged-direct decode scorer (the per-request gather dominates at 512K).
