# Credits and provenance

## MiaAI Lab / Wesley Young — AGPLv3 runtime foundation

This recipe owes its optimized EXL3 serving foundation to **MiaAI Lab
(Mia'a AI Lab) and Wesley Young**, and the contributors to
[DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks).
Their native Engram row store, grouped-prefill MoE implementation, cooperative
MoE work, EXL3 integration, performance investigation and two-Spark recipe work
are substantial upstream contributions, not original work of this repository.

All incorporated MiaAI code and our adaptations of it are explicitly
**AGPL-3.0-only (GNU Affero General Public License version 3)**. Original license
texts, copyright notices, MIT dependency notices and file-level source hashes
are retained under `release/runtime/vendor/`. Our recipe integration is also
distributed under AGPL-3.0-only; this is not an MIT relicensing of MiaAI's work.

Pinned source lineage:

| Component | Upstream revision / location |
| --- | --- |
| Kernel snapshot (unchanged) | `8404ac7d389c418300d0bee960d52313247930e1` |
| Responses content compatibility; mrexodia / MiaAI PR12 | `a2c8d28a2355f193c4008430e07061293fb47a6b` |
| Agent cache reporting and periodic retention; mrexodia / MiaAI PR21 | `14ebc3937d4cef76c3f7369607df817f703e21a1`, `0b5654dfbbfb8aa88c7c4d70e21104403e1fb236` |
| Cooperative-MoE merge | `b9c49e90bdcc6f1e0192feb57214df11b67d36aa` |
| Earlier Engram / grouped-prefill integration | `979e68a` (full pins in the corresponding `UPSTREAM.json`) |
| Cooperative ExLlamaV3 dependency | `02aef45cd681b960a00afcd0749a4ab99e6c1bfe`, original MIT notices retained |

The `UPSTREAM.json` inventories distinguish pristine upstream files from local
adaptations. The native cooperative source adapted for up to 24 rows is in
`release/runtime/sources/cooperative24.{cu,cuh}`. Grouped-prefill and Engram
sources and build instructions accompany their native libraries. SPDX notices
are retained in the derived sources. The corresponding source is shipped with
the recipe/runtime, not available only in a private campaign directory.

The PR12/PR21 adaptations are **AGPL-3.0-only**, with credit to mrexodia,
MiaAI Lab and Wesley Young. We map Responses `input_text`/`output_text`
through a pinned in-memory adapter instead of editing installed vLLM files.
We adopt cache-hit reporting and configurable 4096-token periodic retention,
preserving our replay-tail/image correctness patches and 2048-token prefill.
We do not adopt upstream's launch scripts or text-only serving profile.

## Other foundations

- **DeepSeek-AI:** DeepSeek V4.1 Flash architecture, original model, native
  vision, Engram, and DSpark research/weights. Model revision
  `df42c109f1defefcbfcedbe7d905718a12266e40`; weight licenses are separate.
- **turboderp / ExLlamaV3:** EXL3 and the MUL1 quantization/kernel foundations.
  Original ExLlamaV3 MIT notices remain with the vendored source.
- **vLLM and DSpark contributors:** model serving, scheduling, distributed
  execution, speculative decoding and the OpenAI-compatible API.
- **FlashInfer, CUTLASS, TileLang, Triton and NVIDIA CUDA/NCCL:** native kernel,
  compilation and communication foundations; their licenses remain applicable.
- **NVIDIA CUB/CCCL:** block radix-sort primitives used by the exact
  length-aware top-k kernel; the original CCCL license is retained separately.

## Recipe inspiration

The user-facing configuration and quick-start organization are informed by:

- [MiaAI's DeepSeek V4 DSpark recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark).
- [The GLM-5.3 two-Spark recipe](https://github.com/coolbho3k/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark).
- The neighboring DeepSeek V4 NVFP4/DSpark recipes and their retained contributor
  credits, including Keys, Fraser Price, Anemll, and TonyD2Wild where applicable.

Our additions include this separate 3bpw target/draft quantization campaign,
FP4/DCP2/image-safe integration, display-reserve KV experiments, six-session
adaptation and release packaging. Credit for upstream techniques stays upstream.
These statements do not imply endorsement by MiaAI or other upstream authors.

The concurrent DCP overlap scheduler and packed-output address adaptations are
this recipe's additions to that foundation, also **AGPL-3.0-only**. They retain
the existing attention/merge arithmetic and use vLLM's existing communicator;
no NCCL implementation is vendored. Their measured gains are this recipe's
results, not an upstream MiaAI performance claim.

The page15 Engram layout, asynchronous retrieval schedule and GPU-readable host
staging adapt MiaAI's original native reader/cache/callback and packed dense
row format (`979e68a62c90b24d928f5638596e0ceed90e9f34`). Full credit for that
foundation remains with MiaAI Lab / Wesley Young and upstream contributors.
The adaptations, native reader source and portable download integration are
**AGPL-3.0-only**. Their limited performance results are reported separately;
the packed data does not change or relicense DeepSeek's original weight bytes.

## DSpark draft-length and kernel experiments

The experiments under `release/experimental/dspark/` extend MiaAI's cooperative
target-MoE geometry and specialize the existing MiaAI/ExLlamaV3-derived staged
path for the drafter's top-three experts. Full credit for those kernel
foundations remains with **MiaAI Lab / Wesley Young and contributors**, and
**turboderp / ExLlamaV3**. These adaptations are **AGPL-3.0-only**; the generated
corresponding-source bundles retain the original AGPL, MIT and ExLlamaV3 notices.

Native confidence-based verification and rejection sampling come from vLLM's
DSpark implementation and retain its Apache-2.0 notices. The local SM121/DCP
compatibility hooks do not replace that sampler. The local acceptance-EMA
policy is informed by the attributed GLM recipe's warmup, prefix-verification
and recovery lessons. KV-only projection and Markov-addition integration are
local adaptations around the existing native operations. Experimental results
are documented separately; inclusion of their source is not a claim that every
experiment improved performance or is enabled in the released default.

## SWA group-32 native writer

The fused Q/RoPE/sliding-window writer adapts **vLLM contributors' Apache-2.0**
implementation. The original source, license and provenance are retained under
`release/runtime/vendor/vllm-swa32-apache/`; the corresponding adapted source
and build script are shipped with the SWA results. The local adaptation changes
FP8 scale grouping and exact scale-exponent selection while retaining BF16 RoPE.
