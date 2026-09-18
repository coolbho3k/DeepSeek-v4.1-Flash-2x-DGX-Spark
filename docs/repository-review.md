# Initial repository security and cleanliness review

Reviewed on September 17, 2026, before the first GitHub push. The repository
remains a release candidate: a clean two-Spark GPU launch is still pending.
The working server was not restarted for repository preparation.

## Publication boundary

Only the inventory produced by `release/export.py` belongs in Git. The index
must match that inventory byte-for-byte, including the pinned native binaries.
The checkout is approximately 18 MiB; model weights and the container image
remain separately pinned HF/GHCR downloads, not Git objects.

Campaign reports, calibration data, downloaded models, OCI images, local
configuration, caches, shell history and credentials are excluded. Git's
allowlist includes the performance comparison helper needed by the tests;
regression tests check both that inclusion and private-file exclusions.
Historical author build paths in corresponding source and compiled metadata
are provenance, not required user launch paths. Original upstream licenses
and MiaAI attribution are retained.

## Security checks and boundaries

- Scan the exact staged blobs, including native libraries, against currently
  available publisher credential values and common token/private-key patterns.
  Check sensitive filenames and archive contents; never log matching values.
- Keep SSH credentials outside Git and leave the remote repository private
  until the owner elects to launch publicly. Do not embed tokens in remotes.
- CI uses commit-pinned actions and runs CPU-only tests with read-only repository
  permissions. It does not retain the checkout credential and has no deployment
  or publishing job.
- The launcher uses pinned, verified public downloads. It does not require
  HF/GHCR publishing credentials or install networking, SSH or system services.
- The API is unauthenticated by default and uses host networking. Use only a
  trusted LAN, or loopback plus your own authenticated gateway. Docker and GPU/
  DRM device access are privileged capabilities even without `--privileged`.
- `.env.ds41` is trusted shell configuration, not an untrusted data file.
  Only source configuration you wrote or reviewed. Use user-owned cache paths.

The prior all-layer GHCR credential review is recorded separately in
[`release/security-review.json`](../release/security-review.json). A secret
scan is not a vulnerability audit of every third-party runtime component, nor
a guarantee that no unknown secret or security issue exists.

## Verification before release

Run CPU tests, the launcher dry-run and the bundle hash verifier from an actual
clean Git clone. Check anonymous HF/GHCR metadata against `recipe-lock.json`.
These checks do not allocate GPU memory, download all weights, or demonstrate
a fresh full-model start. That final qualification remains the explicit next
step in [release maintenance](release-maintenance.md).
