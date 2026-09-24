# Display-memory driver investigation (local experiment)

These probes do not qualify a driver for the public recipe. The public driver
guard remains unchanged. Do not run a probe concurrently with serving or any
other GPU job. No module reloads, firmware changes, privileged containers,
physical-address access, or driver installations are performed by these tools.

## Observations, 2026-09-21 PDT

The NVIDIA-branded dgx1 was upgraded to driver **595.84**, keeping kernel
**6.17.0-1029-nvidia**. Its BIOS is `5.36_0ACUM018`, dated 2025-08-06.
dgx0 remains on driver **580.173.02**, with the same kernel, and is an ASUS GX10
with BIOS `GX10DGX.0105.2026.0505.1153`, dated 2026-05-05.

Immediately after reboot, dgx1 had `modeset=N, fbdev=Y`. A 16 MiB dumb-buffer
creation failed with `Function not implemented`, before any CUDA registration.
The operator live-reloaded `nvidia_drm` with `modeset=1 fbdev=0`; the resulting
flags were independently observed as `Y/N`.

With those flags, the **unchanged released** `libds41_display_kv.so`
(`bf60fcdf13126ed74363d2a7f0eff208a7318667d75cde323ba11b6b07f8556c`)
successfully allocated and registered all **1,879,048,192 bytes (1.75 GiB)**,
with zero ordinary prefix. No prefaulting or registration workaround was used.

- The small no-touch mapping and full-size released allocator both passed.
- GPU writes followed by sampled CPU reads had zero mismatches.
- Two full-buffer GPU pattern checks on display backing had zero mismatches.
- Twelve display-backed CUDA graph gather replays, changing the backing pattern
  before each replay, had zero mismatches. Ordinary CUDA control checks passed.
- Registration reduced `MemAvailable` by about **18 MiB**, not 1.75 GiB. This
  supports continued use of the display reserve on this configuration.
- Synthetic full-buffer reads measured about **169 GB/s** on display backing
  versus **262–263 GB/s** on ordinary CUDA allocations. These are not model
  throughput measurements; the repeated small gather is cache-resident and is
  not a DRAM-bandwidth claim.

The reported issue's failure is *later*, at `cuMemHostRegister` with
`CUDA_ERROR_INVALID_VALUE`, on ASUS GX10/firmware0105 with reportedly correct
Y/N flags. We have **not reproduced that failure**. Restoring the flags fixes
our observed post-reboot prerequisite failure, not necessarily issue #3.
Kernel, firmware/platform, memory pressure, and mapping differences remain
unisolated. Do not claim that 595.84 is universally broken or qualified.

Private raw evidence is in `reports/display-driver-595/`:
`small-unfaulted.local.json`, `yn-small-unfaulted.local.json`,
`yn-native-1792.local.json`, and `yn-native-full-gpu.local.json`.

## Probe components

`probe.c` offers a small DRM mapping, anonymous control, or the exact released
allocator loaded from `/original.so`. Its arguments are:

```text
native|drm|anon MiB none|read|write|first|populate io|normal|portable fixed|direct register-MiB
```

Allocation is bounded to 1..1792 MiB; native mode requires1792. A zero registration
length means the entire allocation. Optional variants are diagnostic hypotheses,
not validated workarounds. It requires at least8GiB host `MemAvailable`.

`gpu.c` and `kernels.cu` reuse the campaign's full-pattern and CUDA graph checks.
Compile on an ARM64 CUDA13 host into a dedicated temporary directory:

```bash
gcc -O2 -Wall -Wextra -I/usr/local/cuda/include probe.c gpu.c -o probe -Wl,-l:libcuda.so.1 -ldl
/usr/local/cuda/bin/nvcc -cubin -arch=sm_121 kernels.cu -o kernels.cubin
```

`run.py` requires an explicit SSH host, dedicated `/tmp/ds41-display-595.*`
directory, installed immutable image ID and unique report tag. Stage `probe`,
`kernels.cubin`, and the released library as `original.so` there first.
It runs non-root with the DRM device group, no network, a read-only rootfs,
6GiB cgroup limit, no additional swap, and a120second timeout. `--full-gpu`
enables the GPU kernels/graphs. `--ordinary-mib` is optional for native mode;
the serving recipe uses0. Exited probe containers are retained as evidence.

## Mixed-driver serving experiment

`launch_mixed.py` is a separate, explicit `--execute` path. It requires the
saved deployment's SHA256, verifies loaded and NVML-reported driver580.173.02
on the head and595.84 on the worker, and uses existing verified assets through
the public controller lifecycle. It never reads an environment file or stops
an existing server. API port, weights, kernels, memory utilization, six-sequence
profile, KV allocation, and512MiB emergency watchdog are inherited unchanged.

This local test reuses the previous private runtime kit; it is **not** a
fresh-clone/GHCR installation qualification. The published guard is not relaxed.

### Serving result

Run `ds41-release-v1790043320824738334` started at2026-09-22T02:15:32Z and
the API completed startup at02:25:08Z (about9minutes36seconds). Both ranks
successfully registered their full1.75GiB display-backed pools **after model
loading**, including the595worker. The planner reported **3,313,955 aggregate
KV tokens**, six sequence slots and a1,048,576-token per-request limit.
Target and draft graphs captured on both ranks; the public lifecycle's
health/AOT check completed. No display allocator patch was applied.

Two short HTTP canaries passed: `7 times8` returned `56`, then a coherent
236-token prose completion finished normally in8.92seconds, with first text
at0.52seconds. The streamed portion was8.40seconds (roughly28tokens/s on this
single short prompt; **not** a matched performance comparison). The report is
`reports/display-driver-595/mixed-580-595-http.local.json`.

Afterwards both workers and the controller remained running on port8888.
Observed host `MemAvailable` was about2.46GiB on dgx0 and5.16GiB on dgx1.
The old recipe's0.92utilization, display-only KV budget and512MiB watchdog
were unchanged. No long-context capacity, sustained load, image generation
request, or broad quality evaluation was repeated in this test.

Conclusion:595.84 is **not universally incompatible** with this technique.
This does not reproduce or invalidate the ASUS595report; our595host is a
different platform/firmware, and the reporter's kernel still needs comparison.
The historical pinned worker image was present under its immutable ID despite
not appearing in the default tagged-image listing; no image download was needed.
