# Release validation

## Current kernel update: completed serving trial

GHCR rc2 adds the one-pass decode attention and exact length-aware radix top-k
tested together in deployment v106. The serving overlay and native kernel
bytes match that tested payload. The 12-request suite improved pooled decode
28.27 → 29.51 tok/s (+4.38%); the published cooperative C1 prompt improved
29.98 → 35.67 tok/s at T=0. Acceptance/output changes contribute to these rates;
the 32K prefill observation was not faster. Images, tools and six short requests
passed. See [full results and caveats](kernel-batch-performance.md).

The large-capacity test below belongs to the preceding runtime. KV allocation,
weights and safety settings are unchanged, but that capacity run was not
repeated with the new attention/top-k implementation.

## Preceding serving payload: completed capacity test

Observed hardware/driver: NVIDIA GB10, NVIDIA driver 580.173.02. Other driver
versions are not established by these measurements.

The preserved payload comes from recipe v21 / serving release 103. On September
17, 2026, the existing two-Spark server completed the six-session capacity run:

| Measurement | Result |
| --- | --- |
| Aggregate independent prompt tokens | 3,146,968 |
| Total resident tokens at final observation | 3,161,062 |
| Largest prompt | 1,031,997 tokens |
| Simultaneous decoding sessions | 6 |
| Simultaneous decode observation | 15.03 seconds |
| Aggregate decode in that short window | 50.76 tokens/s, repetitive synthetic output |
| Expected retrieval codes | Found in all six sessions |
| Preemptions / request errors | 0 / 0 |
| Capacity run wall time | 4,837.49 seconds (80.62 minutes) |
| Peak reported KV occupancy | 83.35% |
| Lowest sampled head MemAvailable in test log | 0.887 GiB |

The same runtime passed short six-session generation, an image fixture and
automatic tool calling. Single-session short prose decode measured 23.16
tokens/s. These are limited functional checks, not broad vision/language quality
benchmarks. The capacity test ended only its own requests and left serving up.

Do not convert unused reported KV percentage directly into additional supported
tokens: cache-group lifetimes and encoder/decoder accounting differ during
prefill and decode. The demonstrated capacity is the actual independent request
usage above, not an extrapolation from allocation or occupancy counters.

## Public distribution: published and anonymously accessible

The runtime is uploaded to `ghcr.io/coolbho3k/deepseek-v4.1-flash-exl3-3bpw-2x-dgx-spark:20260917-rc2`.
The immutable digest in `recipe-lock.json` is
`sha256:30557154f0d56b613a95867d41918c50d61f94107fa46861343d08f28c54bce3`.

The package is **15.40 GiB compressed**, with 39 layers (largest 696.70 MiB),
including the compiled cache and corresponding source. It was repackaged from
a pristine image, never committed from a live serving container. Framework
versions and installed duplicates were retained; obsolete lower-layer content
was removed. No new framework compilation was performed.

- rc1's 37 layers are unchanged and reused. The prior imported-image check
  compared 244,379 regular files and 1,577 symlinks to the donor, excluding
  container-generated device/network files. rc2 adds cache/source layers;
  all final image blobs and the remote registry digest were verified.
- The new pinned cache contains 15,341 files. Both asset hashes are pinned.
- All 39 layers, superseded content and 14 nested archives were scanned.
  No current HF/GHCR credential representation matched. The 141 prior findings
  and 19 additional provider/key-pattern matches were reviewed as upstream
  fixtures, parser markers, binary-data coincidences or the empty SSH directory.
  New release layers had no findings. This is not a security certification;
  see [security review](../release/security-review.json).
- The HF target lock now uses its recorded **post-upload** manifest hash,
  `6d79a9ae5cfd121df7c559b76cde87b50551adbe68ae0dc94c97973e85e8e1d1`.
  Only `.gitattributes` changed relative to the pre-upload inventory; all
  55 model/Engram/original-draft shard hashes are unchanged.

Anonymous access to the exact rc2 manifest and both HF manifests passed
`python3 -B release/validate_public.py --online`. Fresh preparation requires
that anonymous check; it never uses publisher credentials as a fallback.
The public wrapper now versions small runtime asset/cache directories while
retaining the shared verified target/drafter downloads across upgrades.

## Public packaging: offline checks, live boot deferred

The public wrapper replaces systemd/campaign-specific launch assumptions with
plain Docker/SSH and explicit configuration. It uses a separate state directory
and does not adopt the currently running campaign server.

All 66 offline tests pass in the workspace. They cover configuration, lifecycle,
GHCR transport, OCI layer round-trips, public-access errors and the published
HF manifest/verifier dispatch, the exact tested kernel pins, secret-audit
redaction/archives, non-destructive runtime upgrades, and Git/export allowlist
equivalence with private-file exclusions. They do not load model weights or
compile/execute CUDA kernels. A clean exported checkout is checked separately.

CPU-only tests cover configuration validation, alternate host paths/UIDs/DRM
devices, one/two-rail command generation, explicit ownership, stale PID rejection,
busy-GPU refusal, side-effect-free dry runs, download integrity, archive traversal
and link rejection, bundled-source inventory and Python syntax.

**Not yet performed:** starting this newly packaged wrapper from a clean clone
and fresh asset cache on two GPUs. The operator explicitly prohibited stopping
the current server for this test. No restart, live kernel replacement, module
reload or new GPU benchmark was performed as part of release preparation.

Before marking a public release fully clean-install-qualified, obtain that
permission and complete the checklist in [release maintenance](release-maintenance.md).
Existing serving success does not prove every new download/launch path.
