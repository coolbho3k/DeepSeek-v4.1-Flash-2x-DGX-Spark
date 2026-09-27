#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Run a drafter script on this Spark's GPU inside the pinned serving image.
# Usage: run_trainer.sh <script.py> [args...]   (paths inside: /drafter /data /draft /target /runs)
# Serving must be stopped on this host: training uses the unified memory the target occupies.
set -euo pipefail
ROOT=/home/emi/code/ds41
IMAGE=${DRAFTER_IMAGE:-sha256:a5ef1cecb16259d16e49578c334b05f60dc58873eeac94cc4a29c5c246d0bcbf}
mkdir -p "$ROOT/artifacts/drafter-runs"
exec docker run --rm --gpus=all --network=host --ipc=host --shm-size=8g \
  --ulimit memlock=-1 --device=/dev/infiniband \
  -e PYTHONUNBUFFERED=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ${DRAFTER_ENV:-} \
  -v "$ROOT/release/experimental/drafter:/drafter:ro" \
  -v "$ROOT/artifacts/drafter-data/capture:/data:ro" \
  -v "$ROOT/artifacts/ds41-exl3-3bpw-candidate-v1/draft:/draft:ro" \
  -v "$ROOT/artifacts/ds41-exl3-3bpw-candidate-v1:/target:ro" \
  -v "$ROOT/artifacts/ds41-draft-exl3-3bpw-sparse-v3:/draft-exl3:ro" \
  -e PYTHONPATH=/opt/exllamav3 \
  -v "$ROOT/artifacts/drafter-runs:/runs" \
  -w /drafter --entrypoint /opt/ds41-venv/bin/python "$IMAGE" "$@"
