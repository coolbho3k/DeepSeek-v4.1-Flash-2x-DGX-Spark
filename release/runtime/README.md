# Private packed-projection / native B12X candidate

Full FP4 main/indexer KV, unchanged FP8 SWA, plus component-qualified packed
wo_a and native B12X MXFP8 linears. B12X sums all FP32 K partials before one
output rounding; BF16 atomic split-K is disabled. wo_a retains BF16 activation
arithmetic and native inverse RoPE. Decode through six rows uses grouped GEMV;
prefill over16 rows reconstructs one temporary32MiB BF16 weight, then native
BMM. Avoiding persistent expansion projects640MiB/rank saved across40 layers.
B12X's extra compact scales reduce the net saving; full-model storage and
speed are not measured yet. This is NOT an admitted or public runtime.

Canonical weights, BF16 vision, SSD engrams, TP2/DCP2,0.90 utilization, KV cap,
8GiB/no-swap serving cgroups, host-memory/ownership guards, one-session/1M
profile and all controller code are unchanged. No drafter is loaded.
Combined frozen startup, full-model stability/quality/performance, full-FP4
1M/six-session serving, speculation and30tokens/s remain unqualified.


## Index-key arithmetic parity

This v5 candidate uses native RMSNorm and an explicit sine-term fused
multiply-add in the native rotary/MXFP4 paged store. Both captured BF16
rounding regressions and256 seeded stress cases passed on both Sparks.
The normalized tensor is call-local and bounded to270,336B at1056 rows;
FP8 calls retain their original writer. Component overhead was about6us.
Combined frozen qualification and all loaded-model results remain pending.


## Loaded B12X backend verification

The v6 candidate fixes v17's stale CUTLASS-only mapping requirement. After
model loading, a read-only observer records actual native B12X selections and
all40 packed grouped projections, qualified source hashes and worker PID/start
time. The controller requires that live receipt. Any mapped CUTLASS library
must still be the qualified one; loading an unused library is not required.
Inference code, weights,0.90, KV budget and all memory/ownership guards remain
unchanged. This is not a full-model performance or quality qualification.


## v7 startup memory

Set the installed Humming package's supported HUMMING_DISABLE_PARALLEL_BUILD=1
before imports in parent and spawn interpreters. This skips two optional build
children, preserving native on-demand initialization. The two-GPU startup-only
A/B measured about593-600MiB less cgroup memory at the matching post-entry stage.
This is not full-model headroom, KV capacity or speed qualification. All kernels,
weights, controllers, allocator ceilings, memory guards and KV budget are unchanged.


## v8 stricter .89 allocation ceiling

The actual Torch allocator cap and native vLLM profile both use0.89, below the
user's0.90 maximum. Worker/preflight requirements derive from that lower maximum;
the2GiB available reserve,2GiB worker cache allowance,1GiB preflight allowance,
8GiB/no-swap CPU cap and768MiB continuous stop floor are unchanged. Native KV
profiling must still admit the unchanged1,009,612,800B/rank downward cache cap.
No KV override, weight/kernel change or speculative shortcut is introduced.
The backend observer pins the new worker hash; its logic is unchanged.
This candidate still requires two-GPU and actual loaded-model qualification.


## v9 native dense decode graph candidate

One-token BF16 native B12X operations are captured during weight post-processing,
before native KV profiling, with shared serialized scratch and independent input/
output ownership. Native packing, corrected FP32 reduction, prefill/fallback math,
BF16 vision, FP4 main/MXFP4 indexer/FP8 SWA, SSD Engrams and TP2/DCP2 are unchanged.
The controller requires all210 loaded dense graph bindings and40 packed wo_a
layers, qualified sources and live-worker identity. No first-request capture.
Actual utilization remains0.89, hard maximum0.90, and all memory reserves/cache
caps remain unchanged. Component qualification is not a loaded-model speed,
quality or memory result. Combined and full-model qualification remain required.


## v10 image-safe fused sparse attention candidate

The source-pinned packed FP4-main/FP8-SWA attention kernel is selected before
native startup. Small batches use bounded key parallelism and a global FP32
normalizer; BF16 queries, two-term BF16 probabilities and FP32 partial outputs
are preserved. Every visible image entry is retained. The observer requires
the actual native attention binding, all210 dense graph bindings and40 packed
projections, qualified source hashes and live worker identity.
Actual utilization stays0.89, user maximum0.90; reserves, KV cap, vision weights,
SSD Engrams and TP2/DCP2 are unchanged. Component and native-fixture results
do not qualify the combined frozen entry or full-model speed/quality/memory.
This kit is not admitted for a full-model launch until separate combined proof.


## v11 image-prefix bootstrap repair

v10 accidentally omitted the image-prefix installer from its optimized entry.
This candidate restores early parent/spawn installation, resolves the original
baked source through the thin package path with the same strict source hash,
and verifies actual native resize/scheduler/prefix hooks before admitting a
loaded model. No cached prefix may end inside an image; complete-image prefix
reuse remains enabled. Vision pixels and weights are unchanged.
Every compute kernel, FP4/MXFP4 codec, model weight, memory reserve, utilization
limit and worker/pair ownership guard is byte-identical to v10. Existing v10
kernel proofs are reused; focused native startup and image checks are required
before launch. No new full-model qualification is claimed by this builder.


## v12 exact fused sparse mapping candidate

Only compressed candidate compaction/address calculation changes. Stable
candidate order, duplicates, image key membership, physical DCP2 ownership
and synchronous invalid-request/page exceptions are preserved. One kernel
replaces the eager sort/arithmetic sequence, with a bounded error transfer.
No quantization, cache payload, persistent GPU workspace, vision, memory
limit or worker/pair ownership changes. v11 image-prefix repair is retained.
Existing kernel proofs are inherited; focused frozen registration and native
attention/image integration are still required before full-model admission.


## v13 native parallel Engram callback (AGPLv3)

Native parallel row reads replace serial Python staging. The original hasher,
image masking, native BF16 vision, FP4/MXFP4/FP8 caches and TP2/DCP2 stay intact.
Pinned and GPU staging is bounded to256 tokens per table; no full table or
resident-scale allocation. Full-model CUDA graphs and DSpark remain disabled.
The new miaai_engram.py, spark_native_engram.py, miaai_row_store.cpp and binary
are AGPLv3, not covered by any surrounding permissive grant. Corresponding
source, upstream provenance and both retained licenses are in vendor/.
Build the CPU binary with g++ -std=c++17 -O2 -shared -fPIC -pthread
serving/miaai_row_store.cpp -o serving/miaai-row-store-v1.so.
The callback and actual embedding hook passed both GPUs. Frozen parent/spawn
startup, loaded-worker verification and full-model performance remain required.


## v15 grouped expert prefill (AGPLv3)

The new spark_grouped_prefill.py and ds41_miaai_fat_moe_v1.so are AGPL-3.0-only,
not covered by surrounding permissive grants. Their corresponding generated
source, build scripts, pinned EXL3 headers and original license notices are in
vendor/miaai-grouped-prefill-ds41-v2/. MiaAI-Lab's original source/provenance is
also retained under vendor/miaai-dsv41-agpl/. The additive build uses CUDA
-O3 -lineinfo --fmad=false and the included PyTorch extension build script.
The bounded shared scratch is144,466,596 bytes, allocated on the first large
forward so native profiling counts it. Small decode retains the prior math.
Device-only routing splits thin experts from grouped experts with at least16
rows. All FP16 boundaries and FP32 pre-down routing are preserved. Original
images, FP4/MXFP4/FP8 KV, TP2/DCP2 and0.89memory utilization stay unchanged.
Do not claim full-model gains from the aliased-weight component benchmarks.

v15 includes the GPU-tested asynchronous-routing Python dependency missing from the preserved v14 candidate. Kernel and dispatcher math are unchanged.


## v16 fused DCP communication (AGPLv3)

spark_dcp_communication.py and ds41/dcp_communication.py are AGPL-3.0-only.
They are the corresponding source (Triton JIT); no opaque binary is added.
The pre-start hook extends the original atomic FP4 installation with exact
stable SWA partitioning, FP32 normalization/merge, and a single packed output
and LSE collective. Original query/sink exchange, image visibility, FP4/MXFP4
cache code, synchronous bounds checks and all memory/ownership guards remain.
No persistent GPU scratch is added. The largest packed send is 8,404,992 bytes.
Component kernels support graphs; full-model capture and DSpark remain off.
Full-model gains require loaded measurements, not the component benchmark.


## Combined MiaAI/DSpark candidate (AGPL-3.0-only)

This variant supersedes the older eager/1056/32-thread/no-draft selection.
Native V2 full decode/verification graphs,2048-token grouped prefill,96 SSD I/O
threads and native FP8/FP4 DSpark are selected together. Full original BF16
vision, FP4 main/MXFP4 indexer/FP8 SWA and TP2/DCP2 remain. The target EXL3
weights and tokenizer are unchanged. Original MIT/Apache notices and complete
corresponding native sources are preserved. Explicit utilization is recorded below.
The outer startup RAM thresholds are retained; the worker uses the same
RAM/allocator formulas. Context and KV caps
are unchanged; actual post-profile capacity still requires measurement.
Actual model memory fit, graph capture, image behavior and speed are NOT yet
qualified. The unchanged owned pair controller and all RAM/no-swap/allocator
guards govern the attempt; successful-only post-warmup evidence is required.

Current frozen selection: utilization=0.925, DSpark enabled=True, KV cap bytes/rank=738197504. This supersedes the preceding profile summary. The1M request limit is unchanged; native admission remains mandatory. Successful post-capture CPU heap trimming is enabled.
