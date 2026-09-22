# Headless setup and experimental display-reserve KV

**Strongly recommended: run both Sparks headless, without an active desktop,
monitor, remote graphical desktop or display workload.** Preserve SSH access
before changing display settings. Do not change or reload NVIDIA modules while
the model, CUDA jobs, or a graphical session are running.

The complete [README setup](../README.md#2-set-up-headless-display-memory-on-both-sparks)
includes persistent boot configuration, the guarded no-reboot module reload,
DRM card discovery and recovery. Keep the UEFI/BIOS display-memory reservation
at **2 GB**, as in the tested configuration; do not set it to zero.

This recipe uses **1.75 GiB per Spark** from the platform's approximately 2 GiB
display-reserved allocation region. It does not increase physical RAM or turn
2 GiB into ordinary CUDA memory. A DRM display buffer is allocated, mapped and
registered for GPU access; the existing FP4/MXFP4/FP8 KV layouts use that backing.
The rest of the model still uses normal CUDA allocations.

This is an experimental NVIDIA/GB10-specific path, not a general Vulkan trick
or a guarantee that every driver/firmware combination exposes the same region.
Its raw read bandwidth measured below ordinary CUDA-backed RAM in our probes.
The launcher fails if the expected path is unavailable; it does not silently
take another 1.75 GiB from ordinary RAM or reduce image support.

## Driver compatibility

The launcher has **no driver-version allowlist**. It allows `595.84` and newer
versions; versions without local serving evidence produce a warning, not a
version-based rejection. The two hosts need not use identical driver versions.
`doctor`, `prepare`, and `start` check `/sys/module/nvidia/version` against
`nvidia-smi` before downloads or stopping an existing pair for `--restart`.
Node preflight and start check again. Missing observations or a kernel/NVML
mismatch on a host still fail. These are read-only consistency checks, not a
CUDA allocation probe or proof that every firmware/kernel combination works.

```bash
cat /sys/module/nvidia/version
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

[Capicua25x reported in issue #3](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark/issues/3)
that `595.84` loads the model and plans the KV pool, then fails with
`register display IO: CUDA_ERROR_INVALID_VALUE`; the same ASUS GX10 hosts and
profile boot successfully on `580.173.02`. The failure is specifically at
`cuMemHostRegister(..., DEVICEMAP | IOMEMORY)`, after DRM allocation and `mmap`,
before obtaining a CUDA device pointer. It is not evidence of a quantization
problem or a request to raise memory utilization.

On September21,2026 (PDT), an NVIDIA-branded Spark on `595.84` and kernel
`6.17.0-1029-nvidia` passed the unchanged1.75GiB allocator's GPU pattern and
CUDA graph checks. A mixed pair with an ASUS GX10 on `580.173.02` then completed
model loading, display-KV registration on both ranks, graph capture and two
short generation checks, with a3,313,955-token planned KV pool. Both used the
same kernel, `modeset=1 fbdev=0`, and unchanged memory limits. This was a local
existing-assets canary, not a fresh-clone or long-context qualification.

We have **not reproduced the reporter's registration failure**, and its cause
is not established. These results rule out a blanket claim that595.84 breaks
this technique everywhere; platform/firmware, kernel and mapping differences
remain possible. CUDA's I/O registration depends on mapping attributes and
physical-page layout, not just free memory.
Keep a supported OS/kernel/driver combination and use your platform's official
driver installation/recovery procedure while idle. With Secure Boot, ensure
the chosen modules are signed and trusted. The recipe does not automate driver
downgrades, edit DKMS policy, or change Secure Boot. A package install without
loading the matching module does not establish kernel/userspace consistency.

## Required state on both hosts

```bash
sudo cat /sys/module/nvidia_drm/parameters/modeset  # must report Y
sudo cat /sys/module/nvidia_drm/parameters/fbdev    # must report N
ls -l /dev/dri/card*
```

Choose the NVIDIA DRM card in `HEAD_DRM_CARD` and `WORKER_DRM_CARD` if it is not
`card0`. The container maps that selected card to its internal `/dev/dri/card0`
and gets only its device group; it is not a privileged container.

## One-time administrator setup (only while idle)

These are instructions, **not actions performed by the launcher**:

1. Arrange a headless boot using your OS's documented method. Stop any desktop
   and GPU jobs, and confirm SSH still works. Disabling a desktop does not mean
   disabling `nvidia_drm` modesetting: this allocator needs modesetting enabled.
2. Inspect existing NVIDIA module options under `/etc/modprobe.d/` and remove
   conflicts deliberately. Using your editor, create a dedicated file such as
   `/etc/modprobe.d/ds41-display-kv.conf` containing:

   ```text
   options nvidia_drm modeset=1 fbdev=0
   ```

3. On Ubuntu/DGX OS, run `sudo update-initramfs -u -k all`, then reboot both
   idle machines when convenient. On other distributions use their
   supported initramfs procedure. Do not blindly unload a live NVIDIA module.
4. Recheck the two parameter files above, then run `./start-server.sh doctor`.

Do not set the firmware's display-memory reservation to zero. Do not disable
CUDA, change zram or globally drop OS caches to make this allocator work.
Normal startup checks memory separately.

## Restoring your prior display setup

First stop this recipe and all GPU/display jobs. Remove only the configuration
file you created for this recipe, restore any prior settings you deliberately
changed, rebuild the initramfs as above, and reboot. Do not remove unrelated
NVIDIA configuration. The launcher itself does not install a systemd unit, alter the
default boot target, edit a firewall, or modify networking.

## Capacity and safety

The tested configuration has zero ordinary KV backing plus 1.75 GiB display
backing per GPU, TP2/DCP2, six sequence slots and a 1,048,576-token per-request
limit. On the preceding v103 runtime, it completed six simultaneous independent
contexts containing 3,146,968 input tokens (3,161,062 including generated tokens
at the final observation).
This is aggregate capacity, **not six full million-token contexts**.
The long-capacity test has not been repeated with rc2's attention/top-k kernels;
see [release validation](release-validation.md).

The background Python controller samples available host memory and stops only
its own two containers below 512 MiB or on a failed worker. This is an emergency
guard, not a guarantee against hard lockups on a unified-memory platform. Keep
both systems otherwise quiet; a desktop or another large process can use the
remaining headroom. The memory-utilization setting is not a whole-system cap.
