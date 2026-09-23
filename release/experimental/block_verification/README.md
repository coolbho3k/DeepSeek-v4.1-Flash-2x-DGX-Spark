# Native block verification experiment

See [measured results and limitations](RESULTS.md). Standard verification
remains the default.

Compare `rejection_sample_method="block"` against the saved probabilistic K3
standard-verification runs. Keep weights, images, KV formats, display memory,
serving limits and all other speculative settings fixed. No new serving
baseline is measured.

`reuse_baseline.py` verifies matching saved requests and writes immutable
benchmark inputs containing the arithmetic mean of the two saved serial runs
and the two saved warm C6 waves. It retains the slower first-use C6 result
separately and hashes the source evidence. These are derived inputs, not new
measurements.

Prepare a candidate with `../draft_sampling/prepare.py` and
`--rejection-sample-method block`, supplying the existing deployment and its
hash. The preparer verifies the parent, changes both launch-profile copies,
refreshes manifests, and verifies the new kit. Copy and verify that kit on both
hosts before using `../model_fusion/launch.py`.

Before serving, run the installed native rejection-sampler tests selected by
`block_verification or placeholder or greedy_rejection_sample` in isolated
runtime containers. `probe.py` adds BF16 draft-probability distribution checks
and full-vocabulary CUDA graph checks, including changed inputs, reversed
request-state mappings, greedy requests, and trailing placeholders. Neither
these tests nor the performance trial prove correctness for every possible
model distribution.

Replay the saved workloads with `../draft_sampling/benchmark.py`, using the
derived `baseline-serial.json` and `baseline-c6.json`. Run two complete candidate
trials and one additional C6 wave, matching the available earlier evidence.
The benchmark admits requests only with the independent memory watchdog and
checks worker identities and isolated request counters. First-use and warm C6
results must both be reported. Candidate text can differ at the same seed.

Local evidence: `reports/block-verification-v1/`. The benchmark depends on the
maintainer's local `probes` package, as do the earlier serving experiments.
