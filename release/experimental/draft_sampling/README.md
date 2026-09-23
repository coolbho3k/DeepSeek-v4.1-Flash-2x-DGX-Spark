# Probabilistic DSpark draft sampling

See [measured results and limitations](RESULTS.md).

This experiment changes the native `draft_sample_method` from `greedy` to
`probabilistic`, preserving the selected model-fusion runtime, fixed K3,
standard rejection sampling, weights, cache settings, memory limits and graph
capacities. The installed API's default temperature is 1.0. The comparison
retains top-p 0.95 and the exact saved requests from the previous serving run.
No new baseline requests are made.

`prepare.py` clones an independently verified runtime, changes the sampling
selection in both pinned launch-profile copies, and refreshes the affected
manifests. It does not edit the parent or stop/start any server. Verify and copy
the resulting immutable kit to both nodes before using the model-fusion
campaign launcher.

`benchmark.py` uses the maintainer's local `probes` library. It replays the four
400-token temperature-one serial cases and the six-request 256-token wave from
completed baseline reports. One 64-token candidate warm-up is excluded from
measurements. Worker identities, the independent RAM watchdog, request
admission margins and isolated request counters are checked throughout.
Reports include useful throughput, draft acceptance and verification-step
costs. The script never changes or restarts the deployment.

Local evidence lives under `reports/draft-sampling-v1/`; the existing baseline
is `reports/model-fusion-v1/final-serial.json` and `final-c6.json`. The first
preparation was rejected before container startup because its two launch-profile
copies disagreed; v2 updates both copies and retains that check.

Probabilistic proposals can change the text for a fixed request seed. Native
exact rejection sampling is intended to retain the target distribution; this
small serving performance trial does not independently establish distributional
correctness or model accuracy. Changing generated text also changes routing
and subsequent draft difficulty, so the comparison is descriptive.
