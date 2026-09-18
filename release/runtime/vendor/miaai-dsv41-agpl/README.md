# MiaAI-Lab grouped EXL3 kernels — AGPLv3

Upstream: https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks
Pinned commit: `979e68a62c90b24d928f5638596e0ceed90e9f34`.
Copyright/attribution: Mia's AI Lab and the upstream contributors.

These vendored sources are GNU Affero General Public License version 3
(AGPLv3), **not part of the surrounding project's permissive-license grant**.
`LICENSE` is the unmodified upstream AGPLv3 text; `LICENSE.MIT` preserves
upstream's historical MIT contribution notices. Do not remove those notices.
All imported code files carry explicit SPDX AGPL-3.0-only markers.
`UPSTREAM.json` records the exact upstream Git blobs and both byte digests.

The grouped prefill kernel/header, native row store, graph-compatible Engram
adapter, and upstream row/dequant tests are imported. No
launcher, model weights, vision-disable patch or memory-watchdog setting is
imported. The row store is compiled through `serving/miaai_row_store.cpp` and
CPU-tested against both canonical Engram tables. It is not registered in
serving, and its CUDA callback has not yet been exercised. The grouped GPU
kernels remain unbuilt and disabled.
The original include paths expect the corresponding ExLlamaV3 build headers;
this directory is an attributed source snapshot, not a standalone extension.

## Required adaptation before serving

Preserve image support and the vision weights, FP4 KV, TP2/DCP2, and the user's
0.90 maximum GPU utilization. The upstream header describes routing weights
being applied after the down projection. Our current canonical expert path
applies routing in FP32 before the down projection's FP16 input boundary.
Direct substitution would change rounding. Review/adapt that arithmetic and
qualify captured real activations, full outputs, tail tiles, stream/graph
ordering, memory bounds and multimodal serving before enabling these kernels.

The original bodies are unchanged. Only the license/provenance comment prefix
has been added locally. The separately AGPL-marked local C++ adapter enforces
O_DIRECT, no resident scales/table, a 64 MiB/table cache ceiling, and at most
96 I/O workers. Current CPU tests use zero cache. Keep further adaptations
separately attributable.
