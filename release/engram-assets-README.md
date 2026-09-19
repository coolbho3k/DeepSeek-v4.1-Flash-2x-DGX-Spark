# Lossless page15 Engrams for the two-Spark EXL3 3bpw recipe

These are **storage-layout caches, not a new quantization**. Main/draft weights,
original Engram weight/scale bytes and canonical `engrams/*.safetensors` are
unchanged. Keep the originals for older runners and other inference engines.

Use the [companion recipe](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark).
Its matching reader and manifest are pinned together. It downloads only the
two tables owned by each TP rank (about **97.66 GiB per Spark**), verifies each
transport part and the assembled file, and mounts the tables read-only. Parts
are streamed straight into a resumable assembly file; no second on-disk copy
of the downloaded parts is required. No user-side packing, quantization or C++
build is needed. Do not rename these files to replace the original Safetensors.

Layout: 4 KiB header, fifteen complete 264-byte rows per 4 KiB page. Each row
contains 256 original FP8 weight bytes plus eight original E8M0 scale bytes.
The final page is zero-padded. `manifest.json` pins each table's complete SHA256,
size, layer, TP ownership, source-model snapshot, and ordered download parts.
`*.bin.part-NNNNN` are transport pieces, not directly usable tables. Concatenate
in manifest order; the recipe performs and verifies this automatically.

Full credit to **MiaAI Lab / Wesley Young** and contributors for the native
Engram reader, bounded cache, callback path and dense packed-row foundation.
This recipe adds page15 packing, deferred retrieval and GPU-readable staging;
the derived implementation is **AGPL-3.0-only**, with corresponding source and
notices in Git. Model/data licenses remain those of the original weights.

Measured cold bulk retrieval was roughly twice as fast, but the full-model
gain was modest: 30.35 → 30.78 tok/s pooled decode; warmed uncached 32K prefill
1,082.7 → about 1,077.6 tok/s. Images, tools and six short concurrent requests
passed. Eleven of twelve baseline replies matched exactly; one coding reply
changed wording reproducibly by seed. Row/dequantization component tests were
exact, but this is **not** broad quality or fresh-clone GPU qualification.
KV capacity and serving memory settings are unchanged.

Compatibility: old recipes pin the previous HF revision and original format.
New recipes pin this complete inventory and the matching ABI-2 reader. The
Hub assets are published before the Git recipe points to them; no mutable
`main` branch is used for runtime downloads. Updating Git does not modify a
running frozen server. An explicit restart is required to use a new recipe.
