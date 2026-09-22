# Third-party notices

The top-level recipe license is **AGPL-3.0-only**. The full text is in `LICENSE`.
MiaAI-derived source, including Engram, grouped-prefill and cooperative-MoE
integrations, remains under AGPLv3. See [CREDITS.md](CREDITS.md) for attribution.

Preserved notices and provenance are shipped in:

- `release/runtime/vendor/miaai-*/LICENSE*` and `UPSTREAM.json`.
- `release/runtime/native-source/LICENSE*`.
- `release/runtime/vendor/vllm-swa32-apache/LICENSE` and `UPSTREAM.json` for the
  Apache-2.0 fused Q/RoPE/SWA writer and its group-32 adaptation.
- `release/runtime/kernel-rebuild/vendor/exllamav3/LICENSE`.
- `release/runtime/notices/FLASHINFER-LICENSE` and CUTLASS notices.
- `release/runtime/kernel-generated/LICENSE`.
- `release/runtime/notices/CCCL-LICENSE` for the NVIDIA CUB/CCCL templates
  used by the native exact top-k implementation. The same upstream license is
  retained in the image at `/usr/local/lib/python3.12/dist-packages/nvidia/cu13/cccl/LICENSE`.

Separately licensed components retain their original terms: ExLlamaV3's MIT
source, vLLM/FlashInfer and other upstream open-source packages, and proprietary
NVIDIA driver/CUDA components are not relicensed by the top-level license.
The prebuilt runtime image also contains upstream packages and their notices.

Model and drafter weights are separate Hugging Face artifacts retaining their
DeepSeek/model-repository licenses. Advertising the weights as EXL3 3bpw does
not change the license of serving code.

When distributing a modified runtime, provide the matching corresponding source,
build/installation scripts and retained notices. When users interact remotely
with a modified AGPL-covered program, retain the required prominent source
offer; the recipe README/source link must remain available to your users.
See the actual license, particularly sections 5, 6 and 13, for its requirements.

The default source payload is `release/runtime/`; its file hashes are recorded
in `bundle-manifest.json`. The prebuilt image/cache and recipe pins belong
together. Do not redistribute a changed binary under an unchanged source pin.
