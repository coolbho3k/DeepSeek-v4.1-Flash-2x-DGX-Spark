# Release implementation

`launch.py` is the public, standard-library-only Docker/SSH launcher.
`config.py` validates the documented configuration without importing CUDA.
`bootstrap.py` downloads pinned public assets, verifies hashes, and imports the
prebuilt ARM64 image. It never builds kernels or quantizes weights.

`runtime/` contains the v106 attention/top-k kernel update published in GHCR
rc2. That update passed short six-session, 32K prefill, image and tool checks.
The Git overlay additionally ships the tested concurrent DCP schedule and the
page15 Engram reader, GPU-readable staging and deferred retrieval. The small
native reader is prebuilt here, with its corresponding C++ source; the image
itself is unchanged. `bootstrap.py --rank` downloads and verifies only the
corresponding rank's packed tables from a separately pinned HF revision.
Old canonical model/draft pins and original Engram files remain unchanged.
See `../docs/engram-io-performance.md` for measured results and atomic upgrades.
The 3.15M-token capacity qualification used the preceding runtime; it has not
been repeated with these kernels. Host-side launch tools have been generalized
for public configuration. Historical qualification flags inside the archived
payload are not release status: see the top-level README and
`docs/release-validation.md`.

MiaAI-derived code is **AGPL-3.0-only**, with the original upstream license and
copyright notices retained. See `../CREDITS.md` and `../THIRD_PARTY_NOTICES.md`.
The runtime is not stock vLLM. Do not substitute a different image or compile
cache and expect the same behavior.
