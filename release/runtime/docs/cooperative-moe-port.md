# MiaAI cooperative MoE port (AGPL-3.0-only)

Pinned upstream: MiaAI Lab / Wesley Young,
[`b9c49e90bdcc6f1e0192feb57214df11b67d36aa`](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks/commit/b9c49e90bdcc6f1e0192feb57214df11b67d36aa).
Native code derives from Turboderp's ExLlamaV3. Original MIT notices and AGPL
license are retained under `vendor/miaai-cooperative-moe-agpl`; the pinned
ExLlama header subset is separate from the checkout used by draft calibration.

This is an opt-in candidate derived from frozen runtime53. It replaces only
K3/MUL1, TP2, top-six, five-to-eight-row target-expert calls with the two-stage
ordinary CUDA-launch implementation. It does not change model weights, vision,
KV layout, DCP, native FP4 draft experts, shared experts or grouped prefill.
Unsupported calls keep the existing dispatch path. The existing dispatcher's
lock, CUDA event, poison state, bank pointers and graph owner are retained.
The native implementation supports one-to-eight rows; our registered hybrid
keeps the existing faster staged kernels at one-to-four rows. Both-rank v3
tests showed no small-batch speed gain from replacing our tuned kernels.

The new scratch consists of nonoverlapping views inside the existing four
serialized temporary tensors: **zero new persistent GPU scratch allocation**.
The original completion-lock array is never borrowed. One fused Triton launch
converts input/route weights to FP16, maps sparse expert IDs and zeros all851
cooperative counters on every call, including graph replay. Counter reset is
essential when fallback and cooperative kernels alternate over aliased temps.
There remain transient input/output tensors and CUDA module/graph overhead;
zero additional scratch does not promise an exactly identical total RAM peak.

Upstream arithmetic is NOT bit-identical to our previous FP32 MMA/FP32 routing
path: route weights round to FP16 and MMA fragments fold FP16 into FP32.
The port preserves upstream arithmetic as an explicitly separate candidate.
Never infer model-quality or speed qualification from its compile success.
The GPU probe keeps the existing2e-5 NMSE gate against previous and canonical
expert paths, tests changing routes, duplicates, missing IDs, poisoned scratch,
graph replay, strided inputs, both input/ID dtypes, and fallback interleaving.
Full-model accuracy/acceptance and end-to-end latency still require matched runs.

Build with `python3 -B probes/run_cooperative_moe_build.py` after importing the
pinned sources. The isolated container has no GPU devices,2CPUs,8GiB RAM,
read-only source and no network, so compilation can overlap draft calibration.
Prepare the additive kit with `python3 -B scripts/prepare_cooperative_runtime.py`.
Selection is `DS41_ENABLE_COOPERATIVE_MOE=1` at startup; zero retains parent
dispatch for A/B testing. A selection change after registration is rejected.
The kit is marked unqualified and does not create a deployment or stop/start
serving. Never modify frozen runtime53 or canonical weights to enable this.

CPU tests: `python3 -B probes/check_cooperative_moe_cpu.py`; its optional
`--torch-alias` mode checks real CPU tensor storage aliasing with no GPU.
GPU tests: `probes/check_cooperative_moe_gpu.py` inside the prepared-kit test
environment, on idle GPUs only. Six actual experts aliased across384 route IDs
measure routing/kernel behavior, NOT full-layer bandwidth or serving tok/s.
