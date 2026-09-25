# SciCode, Artificial Analysis method, against this recipe

Compares the served EXL3 3bpw model with the unquantized DeepSeek V4.1 Flash on
a third-party benchmark, run as closely as possible to how the third party ran
it.

**Reference:** Artificial Analysis (AA) reports **51.85%** SciCode (149.33 of
288 subproblems) for DeepSeek V4.1 Flash (Reasoning, Max Effort), evaluated
2026-09-14 (AA comparison page; mirrored in Epoch AI's `benchmark_data.zip`,
`scicode_external.csv`: 0.518518...). AA's harness is private; its published
method is: test split, scientist background included, subproblem-level pass@1,
step scripts graded on isolated executors with a 300 s timeout (dataset
v1.0.1), 3 repeats, temperature per the model lab's recommendation, and the
maximum output tokens disclosed by the model creator.

No lighter benchmark has a trusted third-party result for this model: neither
AA nor Epoch AI has published GPQA Diamond or Mock AIME for V4.1 Flash.

## Settings

| Setting | Value | Source |
|---|---|---|
| Harness | Official SciCode `inspect_ai` task, upstream `e3158ea` | scicode-bench/SciCode |
| Split / background | test (65 problems, 288 scored subproblems) / included | AA |
| Reasoning effort | `max` (DeepSeek native 100) | AA "Max Effort" |
| Temperature / top_p | 1.0 / 1.0 | DeepSeek model card recommendation |
| Max output tokens | 393,216 (384K) | DeepSeek's documented maximum output |
| Step executor timeout | 300 s (upstream default 1800 s) | AA |
| Repeats | 3 passes, separate output directories | AA |
| Concurrency | C4 (`--max-connections 4 --max-samples 4`) | chosen for this hardware |
| HTTP client timeout | `-M client_timeout=86400` | a 384K-token response takes ~7 h at ~16 tok/s |

**The HTTP timeout must be set with `-M client_timeout`.** `inspect eval
--timeout` does not configure the OpenAI-compatible provider's HTTP client,
which otherwise keeps the OpenAI SDK default of 600 s. Without it, any step
still reasoning after 10 minutes is cut off and retried from scratch (up to
`--max-retries`), silently capping reasoning at ~10 minutes (~10k tokens here)
per step. Two earlier attempts were invalidated this way. Check with
`show_sample.py <problem> <step>`: timed-out calls show `error=Request timed out`,
and with the cap every successful call finishes in under 600 s.

## Harness changes (`scicode-harness.patch`)

1. **Executor timeout 300 s**, as AA states.
2. **SciCode issue #59 fixed.** For the supplied class-based steps 13.6 and
   62.1, upstream extracts only `__init__` from the reference file and drops
   `class Maxwell` / `class Block` / `class EnlargedBlock`, so later steps in
   problems 13 and 62 fail regardless of model output (verified on the real
   files). The fix passes the full reference file forward. Because AA's harness
   is private, `status.py` reports both the fixed score and an unfixed-harness
   bound that counts steps 13.7-13.15 and 62.2-62.6 as failed.
3. **Pinned scoring environment.** Step scripts run under `SCICODE_PYTHON`, a
   separate venv with numpy 1.26.4, scipy 1.13.1, sympy 1.12.1,
   matplotlib 3.9.0, datasets 2.20.0 (the mid-2024 era of dataset v1.0.1).
   Current scipy removed `integrate.simps`, which model code uses (e.g. step
   12.3), so an unpinned environment fails correct code.
4. **Memory kills recorded separately.** Step scripts raise their own
   `oom_score_adj` so the container's OOM killer picks them before the harness;
   a SIGKILL is logged as `oom`, not `fail`. Any `oom` step must be rescored on
   a machine with more memory before a final number is quoted.

Upstream's denominator is 291 (it counts the 3 supplied steps); scores here use
AA's 288.

## Validation before the run

- Validation-split gold solutions under the same container and memory cap:
  48/50 pass. The two failures (70.8, 78.3) are known broken references
  (SciCode issue #43), are not in the test split, and fail identically in the
  pinned environment.
- One validation gold step (10.11, Ewald summation) peaks at 2.43 GiB, which
  set the 3 GiB container cap.
- Dummy-mode run on test problems 13 and 62 confirmed the #59 fix: later-step
  prompts contain the class definitions.

## Running

The harness runs on the worker, which has more free memory, against the head's
server over the fabric (`http://10.100.32.1:8888/v1`), inside a container:

- `--memory 3072m --memory-swap 3072m` hard cap;
- a watchdog in `aa_run.sh` stops the run if worker `MemAvailable` drops below
  1 GiB (serving stops itself at 512 MiB);
- the container outlives SSH sessions (the worker's user manager has no linger).

Setup (paths are absolute; the eval tree lives in ignored `artifacts/evals/`):

```bash
E=/home/emi/code/ds41/artifacts/evals/scicode
git clone https://github.com/scicode-bench/SciCode.git $E && cd $E && git checkout e3158ea
git apply /home/emi/code/ds41/release/experimental/scicode_aa/scicode-harness.patch
python3 -m venv .venv && .venv/bin/pip install -e . inspect_ai "openai>=3.1" gdown
python3 -m venv .venv-score && .venv-score/bin/pip install numpy==1.26.4 scipy==1.13.1 \
    sympy==1.12.1 matplotlib==3.9.0 h5py datasets==2.20.0 rich && .venv-score/bin/pip install --no-deps -e .
.venv/bin/gdown --folder https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR -O eval/data/gdrive
cp /home/emi/code/ds41/release/experimental/scicode_aa/{aa_run.sh,status.py,status.sh,peak_step.py} $E/
# test_data.h5 sha256 48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890
rsync -a --exclude /runs/ $E/ emi@10.100.32.2:$E/   # anchored exclude: openai ships a 'runs' package
ssh emi@10.100.32.2 "docker run -d --name scicode-aa --restart no --user 1000:1000 --network host \
    --memory 3072m --memory-swap 3072m -v $E:$E --entrypoint /bin/bash ds41-base:20260910 $E/aa_run.sh"
```

Progress and scores, from the head: `artifacts/evals/scicode/status.sh`.

`peak_step.py <step> <generated_code_dir> <split>` rebuilds one step's test
script and reports its exit code and peak RSS.

`show_sample.py <problem> <step> [chars]` prints, for a scored problem, every
model call's output tokens, stop reason, errors and retries, plus the start and
end of one step's reasoning.

## Run history

1. 262,144 max tokens (DeepSeek's recommended minimum, not its documented
   384K maximum). Restarted.
2. 384K max tokens, but the 600 s HTTP client default was still in effect.
   Problem 11 scored 8/12 with every call under 600 s and two timeouts on step
   11.4. Restarted. Apparent 100k+-token "long steps" in runs 1-2 were
   repeated 10-minute timeouts and retries, not single long generations.
3. 384K max tokens with `-M client_timeout=86400` (current).

Throughput is ~69 tok/s shared across 4-5 streams (~15 tok/s each), so long
max-effort steps take hours and a pass can take more than a day.
