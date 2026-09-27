# Serving after the quant campaign: searched NVFP4 scales and the indexer key format

The quant quality campaign (the 64K full-pool refit, its held-out evaluation and packaging) comes
first. It decides which deployment is the base: the current one, or a new one with the refit
weights. The runtime work here (searched NVFP4 main-KV scales, opt-in NVFP4 index keys) only
changes the runtime kit, and that kit is derived from whichever kit the base deployment uses.

Run everything on the head node (dgx0) from `/home/emi/code/ds41`. Before any GPU job, run
`python3 artifacts/restore/evict_cache.py` (page cache counts against GPU memory on GB10).

## 1. Base deployment

### Refit rejected

The base is the current deployment,
`.state/public/ds41-release-v1790407264701421380/deployment.json` (sha
`02c6357f2d68822b9bdc0f427c6e36a853fd3446982f63aef670bee9a3e095ea`), with kit
`artifacts/ds41-runtime-swa32-context-fix-v1` (bundle `b9c163a0...`).

### Refit accepted

`scripts/package_fullpool_candidate.py` and `scripts/prepare_fullpool_release.py` build the new
deployment. Each step refuses to overwrite and prints the SHA the next step needs (`M`, `P`, `S`,
`R`, `K`, `D`). Everything v1 stays untouched; the rollback is the current deployment above.

0. **Source shards 47/48** (both engram tables, 203 GB) must be present: the held-out evaluation
   and the packer read them. They were deleted on 2026-09-26 to free disk and are being restored
   from the NAS backup with `package_fullpool_candidate.py restore-source` (hash-checked while
   copying; receipt `reports/source-restore-v1.json`). If a run is interrupted, delete the
   `*.restore-partial` file it leaves and run it again.
1. After dgx1's layers are rsynced into dgx0's bank:
   `python3 -B scripts/package_fullpool_candidate.py check --hash-bank` (only `manifest_absent`
   should be pending).
2. `python3 -B scripts/build_fullpool_manifest.py --drop-page-cache` gives `M`.
3. Held-out evaluation: `python3 scripts/run_fullpool_evaluation.py plan`, then its capture and
   score commands. Compare `reports/heldout-quantized-score-fullpool64k-v1/summary.json` with the
   published v1 numbers (ppl 3.8201, KL 0.04475, top-1 0.9324).
4. **Disk** (decided 2026-09-27: delete both, once safe):
   - dgx0's NAS-backed fit-only inputs (`calibration/capture-{topup-v1,code-topup-v2,
     domain-topup-v3-part-0,part-1}/expert-inputs`, 662 GB, all 45,120 files matched to unchanged
     archive copies) are deleted by `artifacts/restore/retire_after_fits.sh`, which waits for
     dgx0's fit worker to finish and then runs `retire_fit_inputs.py --apply` (receipt and log in
     `artifacts/restore/`). Capture metadata and `capture-source-v1` stay. The `.inputs-LL.verified`
     stamps move to `calibration/.retired-inputs-<ts>/`, so any later refit re-restores from the NAS.
   - After scoring, delete the evaluation's state tensors (`reports/heldout-quantized-states-
     fullpool64k-v1/**/*.safetensors`, about 172 GB); keep its JSON receipts. Re-capturing takes
     about 79 minutes if they are ever needed.
5. With no GPU job running:
   `python3 -B scripts/package_fullpool_candidate.py pack --mode reuse --manifest-sha256 $M`
   gives `P` and `S`.
6. `python3 -B scripts/prepare_fullpool_release.py metadata --package-sha256 $P` gives `R`.
7. `python3 -B scripts/prepare_fullpool_release.py kit --selection-sha256 $S --package-sha256 $P
   --release-manifest artifacts/ds41-release-preview-v3/public/release-manifest.json
   --release-manifest-sha256 $R` writes `artifacts/ds41-runtime-fullpool64k-v1` and gives `K`.
8. Copy `artifacts/ds41-exl3-3bpw-candidate-v2`, `artifacts/ds41-release-preview-v3` and the new
   kit to dgx1 (`rsync -a`, no `--delete`), then verify the kit there with `python3 -B verify.py`.
9. On each host, with the GPU idle (`h=0` on dgx0, `h=1` on dgx1), build and fully hash the model
   view, then copy dgx1's bindings file back to dgx0's `reports/`:

   ```bash
   V=artifacts/ds41-runtime-fullpool64k-v1/tools/verify_mapped_release.py
   MF=artifacts/ds41-release-preview-v3/public/release-manifest.json
   VIEW=/home/emi/code/ds41/artifacts/ds41-runtime-model-view-v3
   python3 -B $V --manifest $MF --manifest-sha256 $R --directory $VIEW \
     --prepare-sources artifacts/ds41-release-preview-v3/PRIVATE-sources-host$h.json \
     --bindings reports/ds41-runtime-model-view-v3-bindings-host$h.json \
     --output reports/ds41-runtime-model-view-v3-prepared-host$h.json
   python3 -B $V --manifest $MF --manifest-sha256 $R --directory $VIEW --full \
     --bindings reports/ds41-runtime-model-view-v3-bindings-host$h.json \
     --output reports/ds41-runtime-model-view-v3-full-host$h.json
   ```
10. Build the deployment, which validates everything with the new kit and prints `D`, the launch
    command and the rollback command:

    ```bash
    python3 -B scripts/prepare_fullpool_release.py deployment --kit-sha256 $K --release-manifest-sha256 $R \
      --bindings reports/ds41-runtime-model-view-v3-bindings-host0.json \
      --bindings-host1 reports/ds41-runtime-model-view-v3-bindings-host1.json \
      --receipt-host0 /home/emi/code/ds41/reports/ds41-runtime-model-view-v3-full-host0.json \
      --receipt-host1 /home/emi/code/ds41/reports/ds41-runtime-model-view-v3-full-host1.json \
      --output /home/emi/code/ds41/reports/ds41-fullpool64k-deployment-v1.json
    ```

The DSpark drafter was trained on v1 hidden states; measure draft acceptance after the switch.

```bash
BASE=<deployment.json>; BASE_SHA=$(sha256sum "$BASE" | cut -c1-64)
PARENT_KIT=$(python3 -c "import json;print(json.load(open('$BASE'))['nodes'][0]['kit'])")
PARENT_SHA=$(python3 -c "import json;print(json.load(open('$BASE'))['kit_manifest_sha256'])")
```

## 2. Search/NVFP4 kit

For the current deployment this is done: `artifacts/ds41-runtime-nvfp4-indexer-v2`, bundle
`2f5c11fba3515f80ddfcacb04f0123bafce782158f27eafe86e906e9f6ab1edb`, parent `b9c163a0...`,
present and verified on both hosts. (v1, `b61271ca...`, predates the scale search and the `nvfp4`
format name. Do not use it.)

If the refit is accepted, the base kit becomes `ds41-runtime-fullpool64k-v1`, so re-derive on top
of it. It takes seconds and refuses a parent whose reviewed files differ:

```bash
python3 -B release/experimental/indexer_fp4/prepare_kit.py --parent-kit "$PARENT_KIT" \
  --parent-sha256 "$PARENT_SHA" --output artifacts/ds41-runtime-nvfp4-indexer-v3 \
  --receipt artifacts/ds41-runtime-nvfp4-indexer-v3.receipt.json
rsync -a artifacts/ds41-runtime-nvfp4-indexer-v3 emi@dgx1.lan:/home/emi/code/ds41/artifacts/
ssh emi@dgx1.lan "cd /home/emi/code/ds41 && python3 -B artifacts/ds41-runtime-nvfp4-indexer-v3/verify.py \
  artifacts/ds41-runtime-nvfp4-indexer-v3 --manifest-sha256 <printed sha>"
```

Verify with `python3 -B` so no bytecode lands in the kit. A `__pycache__` directory fails bundle
verification.

## 3. A/B gates

Four modes on the same deployment, weights and serving settings:

| Mode | Kit | Main KV | Index keys | Decode queries | Prefill queries |
| --- | --- | --- | --- | --- | --- |
| control | deployment's own | as recorded (`nvfp4_4over6`) | MXFP4 | MXFP4 | MXFP4 |
| kv-search | new | `nvfp4_search` | MXFP4 | MXFP4 | MXFP4 |
| nvfp4-fp8 | new | `nvfp4_search` | NVFP4 | FP8 | NVFP4 |
| nvfp4-nvfp4 | new | `nvfp4_search` | NVFP4 | NVFP4 | NVFP4 |

This takes about 2–3 h: four boots of about 10 minutes each, plus the benchmarks and retrieval runs.

```bash
KIT=artifacts/ds41-runtime-nvfp4-indexer-v2
KIT_SHA=2f5c11fba3515f80ddfcacb04f0123bafce782158f27eafe86e906e9f6ab1edb
python3 -B release/experimental/indexer_fp4/ab_gates.py --base-deployment "$BASE" \
  --base-sha256 "$BASE_SHA" --kit "$KIT" --kit-sha256 "$KIT_SHA" --tag post-campaign-v1 \
  --retrieval 131072 524288 --leave-running control
```

Reports go to `reports/indexer-ab/post-campaign-v1/<mode>/`, and `summary.json` records every step.

Gates, each against control:

1. **Boot and function:** the mode boots, and the generation diagnostics pass (text, images,
   tools).
2. **Speed:** serial decode, C6 concurrency and prefill tok/s are no worse than control, within
   noise. The search adds a few microseconds per decode step and well under a millisecond per
   2,048-token prefill chunk, so kv-search should be indistinguishable from control.
3. **Retrieval:** the 128K and 512K long-context retrievals pass wherever they pass in control.
4. **Quality:** KL and top-1 against a reference on long prompts. This is a separate offline job
   and is not run by `ab_gates.py`.

Choosing:
- **Main KV:** `nvfp4_search` if kv-search passes 1–3. It is byte-for-byte never worse than
  four-over-six per group (23.5% lower SSE on the probe's fixtures, 12% on Gaussian data).
- **Index keys:** NVFP4 if both NVFP4 modes pass 1–3.
- **Decode queries:** `nvfp4` if nvfp4-nvfp4 passes 1–3 with a clear long-context decode win;
  otherwise `fp8`.

## 4. Serve the chosen modes

```bash
./stop-server.sh
python3 -B release/experimental/model_fusion/launch.py --base-deployment "$BASE" --base-sha256 "$BASE_SHA" \
  --kit "$KIT" --kit-sha256 "$KIT_SHA" --port 8888 \
  --fp4-kv-mode nvfp4_search [--indexer-k-format nvfp4 [--indexer-decode-query nvfp4]]
```

**Rollback:** relaunch without `--kit` and the flags. That runs the deployment's own kit exactly as
before. Cached KV pages are never reused across modes, because every mode starts fresh processes.

The repo defaults for new deployments are `nvfp4_search` main KV and NVFP4 index keys with FP8
decode queries (`launch_profile.py` keeps MXFP4 as the meaning of a descriptor without the
field, so existing deployments are unchanged; `model_fusion/launch.py` passes that explicitly).

## 5. Next: solve 1M decode (owner asked to be reminded)

1M-context decode runs at about 2.3 tok/s, against 25–31 tok/s at short context. A short prompt
right after a 1M request still runs at only about 7 tok/s. Indexer scoring explains only about
20–50 ms of the ~420 ms per token, so most of the slowdown is unexplained. Once serving is back,
profile a 1M decode step on both ranks (per-stage CUDA events and host time, or nsys) before
optimizing anything. Suspects: the per-request key gather, top-k over 1M logits, DCP
communication, per-step host work that scales with block tables or allocated cache, and memory
effects that persist after the long request.
