# Maintainer release procedure

End users do not run these packaging commands. Their entry point is
`./start-server.sh`; no local image/kernel/quantization build is required by the
intended prebuilt distribution path.

## Before pushing the repository

1. Keep the exact public weight and drafter pins in `recipe-lock.json`.
2. Follow [GHCR packaging](../release/GHCR.md) to publish the audited runtime
   with its cache and corresponding source. Pin the actual immutable registry
   digest and every payload hash. Confirm anonymous pulls; no guessed image
   URLs, mutable tags or private-package dependencies in the public lock.
3. After an intentional host-tool/source change, refresh the staged bundle with
   `python3 -B release/freeze.py`, then run:

   ```bash
   python3 -B release/validate_public.py --online
   bash -n start-server.sh
   python3 -B -m unittest discover -s tests -v
   python3 -B release/export.py --output /absolute/new/empty/release-directory
   ```

4. Inspect the export. It contains only the public allowlist, not `.env.ds41`,
   `.state`, tokens, weights, campaign reports, calibration inputs or old launch
   scripts. The exporter refuses an existing destination and does not delete
   anything. Before committing, compare the Git index against the export
   inventory and scan the staged blobs for secrets. Never force-add ignored
   campaign files. Test the committed tree in a fresh clone before pushing.
   See [repository review](repository-review.md) for the initial review scope.
   Runtime/cache versions are stored separately under the configured asset
   cache; a new runtime pin must not overwrite old archives or redownload
   unchanged target/drafter weights.
5. Do not claim fresh-clone GPU qualification until the remaining check below
   has actually completed. Keep source, binary pins, licenses and measured
   claims in sync. Never relabel MiaAI-derived code as MIT.

## Pending authorized clean-install test

**Do not stop the existing server until the operator explicitly authorizes it.**
CPU tests and HTTP metadata checks are allowed without restarting it.

Once authorized: use the exported checkout, public Hub and GHCR pins, with the GPUs
idle. Run `doctor`, `prepare`, and the ordinary default start command. Confirm
no image/kernel build or quantization is needed, both workers load the expected
native backends, and the API works directly on the head's configured LAN port.
Check image requests, automatic tool choice, reasoning, six short sessions,
then a long-context capacity workload while watching both hosts' RAM.

Confirm repeated starts reuse verified downloads, an interrupted download
resumes, an explicitly stopped pair can restart, and Ctrl+C during the readiness
wait or log viewing leaves serving alive. Leave the final server running unless
the operator asks otherwise. Record exact artifact digests and results here.

## Corresponding source

The frozen runtime includes native sources, generation/build helpers, vendor
snapshots, `UPSTREAM.json` records and original notices. The image also retains
its installed open-source implementation and package notices. Archive build
receipts may contain the original author's build paths; these are provenance,
not user launch dependencies. The public launcher does not execute historical
`launch_node.py` or `prepare_dual_rail.py` campaign tools in the source bundle.

The included source/build context is for inspection, modification and license
compliance. A bit-for-bit source rebuild of the entire third-party base image
has not been qualified; public reproduction uses the pinned prebuilt image.
