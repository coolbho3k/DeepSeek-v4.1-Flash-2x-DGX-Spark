#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# SciCode as Artificial Analysis runs it (test split, 288 scored subproblems,
# scientist background, pass@1, 300 s executor timeout, 3 repeats), against the
# ds41 server at reasoning_effort=max. Runs inside a memory-capped container.
set -u
E=/home/emi/code/ds41/artifacts/evals/scicode
export PATH=$E/.venv/bin:$PATH HOME=/tmp HF_HOME=$E/.hf OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export SCICODE_PYTHON=$E/.venv-score/bin/python
export DS41_API_KEY=none DS41_BASE_URL=${DS41_BASE_URL:-http://10.100.32.1:8888/v1}
MIN_AVAILABLE_KIB=$((1024*1024))   # stop at 1 GiB, twice serving's 512 MiB safety stop
STATE=$E/runs/aa-state
mkdir -p "$STATE"
echo "running $(date -u +%FT%TZ)" > "$STATE/status"

# Host-memory watchdog: /proc/meminfo in the container reports the host.
(
  while sleep 5; do
    avail=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
    echo "$(date -u +%FT%TZ) $avail" > "$STATE/host-mem"
    if [ "$avail" -lt "$MIN_AVAILABLE_KIB" ]; then
      echo "stopped_low_host_memory $(date -u +%FT%TZ) MemAvailable=${avail}kB" > "$STATE/status"
      pkill -TERM -f 'inspect eval' ; sleep 20; pkill -KILL -f 'inspect eval'
      kill -TERM $$ ; exit 0
    fi
  done
) &
WATCHDOG=$!

cd $E/eval/inspect_ai
for pass in 1 2 3; do
  if [ -f "$STATE/pass$pass.done" ]; then continue; fi
  echo "pass $pass started $(date -u +%FT%TZ)" >> "$STATE/history"
  inspect eval scicode.py \
    --model openai-api/ds41/deepseek-v41-flash-exl3 -M client_timeout=86400 \
    --reasoning-effort max --temperature 1.0 --top-p 1.0 --max-tokens 393216 \
    --max-connections 4 --max-samples 4 --log-buffer 1 \
    --timeout 86400 --max-retries 30 --retry-on-error 2 --no-fail-on-error \
    -T split=test -T with_background=True \
    -T output_dir=$E/runs/aa-pass$pass -T h5py_file=$E/eval/data/gdrive/test_data.h5 \
    --log-dir $E/runs/aa-logs/pass$pass --display plain \
    > "$STATE/pass$pass.out" 2>&1
  code=$?
  echo "pass $pass exited $code $(date -u +%FT%TZ)" >> "$STATE/history"
  if [ $code -ne 0 ]; then echo "failed_pass$pass exit=$code $(date -u +%FT%TZ)" > "$STATE/status"; kill $WATCHDOG; exit $code; fi
  touch "$STATE/pass$pass.done"
done
echo "complete $(date -u +%FT%TZ)" > "$STATE/status"
kill $WATCHDOG
