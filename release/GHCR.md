# GHCR publication

Uploaded package: [coolbho3k/deepseek-v4.1-flash-exl3-3bpw-2x-dgx-spark](https://github.com/users/coolbho3k/packages/container/package/deepseek-v4.1-flash-exl3-3bpw-2x-dgx-spark).

- Published tag: `20260917-rc2`.
- Digest: `sha256:30557154f0d56b613a95867d41918c50d61f94107fa46861343d08f28c54bce3`.
- Download: 15.40 GiB compressed, 39 layers; about 164 MiB added over rc1.
- Native source, MiaAI attribution and AGPLv3 notices are included.
- Anonymous access to this exact manifest is verified. No login is required
  for consumers. `python3 -B release/validate_public.py --online` checks it.
- Full fresh-clone two-node GPU startup remains untested by operator instruction.

rc2 includes the v106-tested one-pass attention and native exact radix top-k,
plus 2,384 additional compiler-cache files (15,341 total). Framework layers
are reused unchanged. Target and drafter HF pins are unchanged. The complete
39-layer image, including superseded content and 14 nested archives, was
scanned before publication. None of the five known credential representations
matched; all 160 pattern/path findings were reviewed as upstream examples,
parser markers, binary-data coincidences or the empty SSH directory. The new
release layers had no findings. See [security review](security-review.json).
This is a bounded credential audit, not a comprehensive security certification.

## Packaging design

The release reuses the exact compiled runtime, preserving framework versions,
binary bytes and paths. It is **not** a framework rebase, and does not remove
installed duplicate PyTorch/vLLM packages. The merged pristine filesystem
eliminates obsolete content from historical Docker layers. No live-container
commit, host home directory, Docker credentials, model weights or serving state
was packaged.

`repack_oci.py` exports a new, never-started, mount-free container from the
immutable donor image. It streams that filesystem into roughly 1-GiB raw
layers, scanning credential patterns and recording file hashes. This avoids
GHCR's 10-GB-per-layer limit. Its guard stops its own export, not the server, if
available host RAM drops below 1.5 GiB.

`finalize_oci.py` adds the public recipe's corresponding-source snapshot and
asset inventory. The image contains `/opt/ds41-release/kernel-cache.tar` and
`runtime-source.tar.gz`. The latter is a build-time source snapshot, not an
alternative installer; use this repository's final external image/weight pins.
A digest-pinned image cannot embed its own final digest in its source snapshot.
The repository's launcher and final pins are authoritative; maintainer metadata
and installer-only fixes may postdate that build-time snapshot. Runtime serving
modules and the native kernel are byte-identical to the tested v106 payload.
The legacy input-bundle qualification flags remain false, as required by its
verifier; they are not an account-publication receipt. Actual registry
publication and public access are recorded in `publication-status.json`.

`push_ghcr.py` verifies the final blobs and a manifest-specific content-review
receipt, then uses a checksum-verified upstream `crane` binary with the existing
Docker login. Uploading does not require a daemon import, GPU access or server
restart. The verification here additionally imported the image locally and
compared its files in a read-only, no-GPU container capped at 256 MiB.

## Future maintainer releases

1. Authenticate to GHCR under the intended owner. Never use an HF token for
   GitHub, embed a token, or replace an existing different release tag.
2. Use `package_ghcr.py` to prepare the bounded asset context from a pristine
   immutable image. Its emitted Dockerfile is a local packaging fallback; do
   **not** use its single merged filesystem layer for GHCR.
3. Use `repack_oci.py` with a new output directory and candidate tag. Include
   all corresponding sources and licenses using `finalize_oci.py`.
4. Review the new redacted scan and source inventory. Do not blindly reuse an
   earlier audit. Use `push_ghcr.py` only with a review tied to the final digest.
5. Verify imported image identity, unchanged runtime files, packaged cache and
   source, and registry digest. Set the package public and test anonymously.
6. Update only the staged public kit's image identity and controller hash.
   Pin actual registry and HF manifest digests in `recipe-lock.json`, then run
   `python3 -B release/freeze.py`, offline tests, online metadata validation and
   a clean export. Do not modify an existing live kit.
7. Obtain explicit permission before stopping the server for a clean GPU boot.

End users only run `./start-server.sh`. The launcher pulls by digest, extracts
verified assets from a temporary container that is **never started**, and
downloads the pinned HF target/drafter. No publisher token, local Docker build,
systemd service, tunnel or network reconfiguration is required.

GitHub references: [authentication](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
and [package visibility](https://docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility).
