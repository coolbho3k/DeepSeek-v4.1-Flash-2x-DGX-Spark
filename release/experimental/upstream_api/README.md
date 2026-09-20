# MiaAI Responses/cache integration

AGPL-3.0-only; credit MiaAI Lab / Wesley Young and mrexodia.
See [scope and configuration](../../../docs/upstream-api-cache.md).

`prepare.py` is a maintainer-only CPU packaging tool, not the end-user path.
It updates reviewed source hashes transitively (failing on cycles), preserves
the private test kit's existing Engram bindings, and creates a fresh immutable
candidate. It never changes a mounted kit, downloads weights or starts/stops
the server. Public users receive the source overlay in `release/runtime` and
use the ordinary launcher; no new native runtime build or weight upload.

The public refresh intentionally requires exactly the reviewed five edits
against its previous manifest and fails if invoked again after freezing.
Private candidates require an independently pinned verified parent and a
nonexistent output directory. Qualification claims are not inherited as proof
that a candidate has been tested.
