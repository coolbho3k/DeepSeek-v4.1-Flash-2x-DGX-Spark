# Standalone verification of downloaded release files

`tools/verify_downloaded_release.py` in the runtime kit (or
`scripts/verify_downloaded_release.py` in the workspace) uses only Python's
standard library. It requires an independently supplied SHA256 for the public
`release-manifest.json`; it never trusts a manifest simply because it is next
to the weights. Obtain both the tool and digest from a trusted release source.

The current supported manifest is the unapproved metadata-preview schema.
Its manifest-only check covers the 129-file public map, including all 55
shards / 426,055,648,464 weight bytes. It does not claim those files have been
downloaded, nor that their payloads have been checked:

```bash
python3 /path/to/kit/tools/verify_downloaded_release.py \
  --manifest /path/to/release-manifest.json \
  --manifest-sha256 TRUSTED_PUBLIC_MANIFEST_SHA256
```

After a release has been materialized, run one full verification on each host
BEFORE loading models. This streams every public file with a 1MiB buffer and
verifies size, SHA256 and unchanged file identity. The command refuses live
GPU compute processes when `nvidia-smi` is available. It never stops workers
or flushes caches. Reading roughly 426GB is substantial I/O even with bounded
process memory; do not run this pass beside serving or calibration.

```bash
python3 /path/to/kit/tools/verify_downloaded_release.py \
  --manifest /path/to/model/release-manifest.json \
  --manifest-sha256 TRUSTED_PUBLIC_MANIFEST_SHA256 \
  --directory /path/to/model --full \
  --output /path/to/private-receipts/host0-full-v1.json
```

Use a fresh receipt path outside the model directory. Receipts include local
paths/inodes and are PRIVATE; do not upload them. A failed or interrupted pass
does not write a successful receipt. Source files are never modified.

The model directory must contain materialized regular files. Symlinks and
unexpected public files are refused. Normal `.cache/huggingface/` download
bookkeeping is ignored, but not followed through symlinks, and is never treated
as public model data. A symlink-based HF snapshot is not accepted as a
materialized release directory. Do not add private plans/credentials to it.

Before a later launch, the following checks that a previous full verification
still refers to the same local files. It reads only file metadata and the
manifest, not the weight payloads:

```bash
python3 /path/to/kit/tools/verify_downloaded_release.py \
  --manifest /path/to/model/release-manifest.json \
  --manifest-sha256 TRUSTED_PUBLIC_MANIFEST_SHA256 \
  --directory /path/to/model \
  --check-receipt /path/to/private-receipts/host0-full-v1.json
```

Moving/replacing/changing files or changing the verifier invalidates the
receipt and requires another full pass. This is a local integrity optimization,
not a defense against an administrator who can rewrite both files and receipts.
It does not establish model accuracy, publisher authenticity, source-payload
equivalence, or publication approval. The original full source verification
and quality evaluation are separate evidence.

The tool passed a synthetic complete-shard full hash and fingerprint-reuse
test, 38 corruption/schema/path/race/resource refusals, and an isolated
manifest-only check outside the workspace. No actual model payload was read
by those tests. The frozen preview is a virtual map and cannot itself be used
as the model directory for the `--full` command.
