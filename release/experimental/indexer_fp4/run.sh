#!/bin/bash
# CPU-only indexer FP4 probe: stage each index-source layer's captured text states from the NAS,
# project them to indexer q/w/k in the calibrator image (no GPU), then compare top-512 selections.
# Page cache is dropped as it goes: on GB10 it counts against the running refit's GPU headroom.
set -euo pipefail
ROOT=/home/emi/code/ds41
NAS=/mnt/synology/spark_backup/ds41-20260918/dgx0/data/calibration/capture-source-v1/states
OUT=$ROOT/artifacts/indexer-fp4-probe
IMG=ds41-calibrator:base
RUN=(docker run --rm --network none --cpus 6 --memory 16g --memory-swap 16g --user 1000:1000
     -v "$ROOT:/work:ro" -v "$OUT:/out" "$IMG")
mkdir -p "$OUT/stage" "$OUT/proj"
for L in 02 08 14 20 24 28 32 36; do
  [ -f "$OUT/proj/$L/.done" ] && continue
  rm -rf "$OUT/stage/$L"; mkdir -p "$OUT/stage/$L"
  rsync -t "$NAS/$L"/text-*.safetensors "$OUT/stage/$L/"
  python3 "$ROOT/artifacts/restore/evict_cache.py" "$NAS/$L" > /dev/null
  "${RUN[@]}" /work/release/experimental/indexer_fp4/project.py "$((10#$L))" "/out/stage/$L" "/out/proj/$L" \
    > "$OUT/project-$L.log" 2>&1
  touch "$OUT/proj/$L/.done"
  rm -rf "$OUT/stage/$L"
  python3 "$ROOT/artifacts/restore/evict_cache.py" "$OUT" > /dev/null
  echo "$(date -u +%FT%TZ) layer $L projected ($(ls "$OUT/proj/$L" | grep -c safetensors) records)"
done
"${RUN[@]}" /work/release/experimental/indexer_fp4/analyze.py /out/proj /out/report.json > "$OUT/analyze.log" 2>&1
python3 "$ROOT/artifacts/restore/evict_cache.py" "$OUT" > /dev/null
echo "$(date -u +%FT%TZ) report written: $OUT/report.json"
