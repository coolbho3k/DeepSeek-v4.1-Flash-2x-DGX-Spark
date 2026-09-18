# DeepSeek V4.1 Flash EXL3 3bpw on two DGX Sparks

An OpenAI-compatible server for **DeepSeek V4.1 Flash**, using our published
**EXL3 3bpw MUL1 target and DSpark drafter**, TP2/DCP2, compact FP4 KV, native
image support, tool calling and up to six concurrent sessions.

**Built on the substantial work of [MiaAI Lab / Wesley Young](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks).**
Their Engram, grouped-prefill and cooperative-MoE implementations are central
to this recipe. All incorporated MiaAI code and adaptations remain explicitly
**AGPLv3**, with original notices and corresponding source included.
See [credits](CREDITS.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

> **Run both Sparks headless.** An active desktop/display workload consumes
> memory this configuration needs. The default KV allocator uses **1.75 GiB per
> GPU from the approximately 2 GiB display-reserved region**. Both hosts need
> `nvidia_drm modeset=1 fbdev=0` and an accessible NVIDIA DRM card. Read the
> [one-time setup and recovery instructions](docs/display-memory.md) **before
> starting**. The launcher never changes drivers, boot settings or networking.
> This is experimental and driver-dependent; a headless setup is strongly
> recommended, not a guarantee against unified-memory exhaustion.

> **Public prebuilt runtime:** `20260917-rc2` is available on
> [GHCR](https://github.com/users/coolbho3k/packages/container/package/deepseek-v4.1-flash-exl3-3bpw-2x-dgx-spark),
> pinned by digest in `recipe-lock.json` (**15.40 GiB compressed**, including
> kernel caches and corresponding source). Anonymous access is verified. This
> version includes the tested one-pass decode attention and exact length-aware
> top-k kernels. `prepare` never falls back to a local image build.

> **Release candidate, not yet clean-install-qualified:** the current runtime
> passed short six-session, image, tool and 32K prefill checks. The 3.15M-token
> capacity test below used the preceding runtime and has not been repeated with
> the new kernels. The fresh-clone launcher has CPU-only validation; its clean
> two-node GPU boot is deferred until explicitly authorized.
> See [validation status](docs/release-validation.md).

## What you need

- Two **128 GB NVIDIA DGX Sparks / GB10**, ARM64 Linux, preferably otherwise idle.
- Working NVIDIA drivers, NVIDIA Container Toolkit and non-root Docker access
  on both. Python 3.11+, `ssh`, `rsync`, `ip` and `ss` on both; `git` on the head.
- Existing passwordless SSH from the head to the worker, with the host key
  already verified. Neither root SSH nor matching usernames/home directories
  are required. The launcher does not install SSH keys or change SSH policy.
- A working ConnectX RoCE connection; the measured setup used two 200-Gbit/s
  links. One or two links can be configured; single-link performance has not
  been qualified by the reported run.
- **At least 550 GiB free disk space on each host** for a fresh install. The
  target download is about 397 GiB, the separate drafter 4.76 GiB, plus the
  prebuilt runtime, extracted image, kernel caches and working-space reserve.
- Internet access for the initial public Hugging Face and GHCR downloads on each host.
  **No HF token is needed.** Never give the launcher a write token.

Only the head needs this checkout. It copies the small runtime recipe to the
worker; each host downloads verified weights locally. No NFS mounts, SSH
tunnels, port-forwarding, firewall rules, systemd services or startup daemons
are installed. Existing fabric networking must already work.

## Quick start

Clone this repository, enter its directory, and run everything below on the
Spark that should host the API:

```bash
git clone https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark.git
cd DeepSeek-v4.1-Flash-2x-DGX-Spark
cp .env.ds41.example .env.ds41
```

Edit `.env.ds41`: set your worker SSH address, both fabric IPs, interface/HCA
names and storage paths. The addresses in the example are **examples**, not
settings the launcher applies to your network. For one RoCE link, clear all
six `*_SECONDARY_*` fields.

```bash
./start-server.sh --dry-run   # resolved configuration; no network or GPU work
./start-server.sh doctor     # read-only checks on both configured hosts
./start-server.sh prepare    # download pinned assets, verify, pull prebuilt image
./start-server.sh            # start both nodes and wait for readiness
```

The first preparation is a large download. Interrupted downloads resume, and
completed payloads are SHA-256 verified. Repeat launches reuse verified files.
Runtime/cache versions are stored separately, so recipe upgrades preserve old
runtime assets and reuse the unchanged model and drafter downloads.
The distribution is designed to require **no manual quantization, Docker build
or kernel preparation**: native binaries and a pinned prebuilt kernel cache
are supplied with the runtime assets. Normal CUDA initialization and graph
capture still happen at startup. Complete cold-start cache coverage—and the
absence of any first-use JIT compilation—must still be checked in the pending
clean-install test. There is no automatic image-build or quantization fallback.

The public API is available directly at `http://<head-LAN-IP>:8888/v1`.
The default model name is `deepseek-v41-flash-exl3`.

```bash
curl http://<head-LAN-IP>:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v41-flash-exl3","messages":[{"role":"user","content":"Hello!"}],"max_tokens":128}'
```

The API has **no authentication by default**. Keep it on a trusted private LAN;
use your own authenticated gateway before exposing it beyond that network.
For local-only access, set `API_HOST=127.0.0.1`.

## Everyday use

```bash
./start-server.sh status
./start-server.sh logs                    # head inference/throughput logs
./start-server.sh logs --node 1           # worker logs
./start-server.sh logs --controller       # startup checks and RAM watchdog
./start-server.sh stop                    # only this recipe's recorded pair
./start-server.sh --restart               # explicit stop + fresh launch
```

Startup detaches a small **plain Python controller**, not a system service.
It watches the two exact containers and available host RAM. The server stays
up when startup returns or your terminal closes; Ctrl+C while waiting for
readiness or viewing logs stops only the wait/viewer. To stop serving, use the
explicit `stop` command. Nothing is configured to start automatically on boot.
Logs and ownership records live in `.state/public/` and the configured cache's
`runs/` directory. No unrelated containers or historical campaign are adopted.

## Configuration

Edit `.env.ds41`, or use `DS41_ENV_FILE=/absolute/path/to/config`.
CLI options override the configuration file for that invocation:

```bash
./start-server.sh --port 8888 --max-num-seqs 4 --max-model-len 1048576
```

| Setting | Default | Supported choices |
| --- | --- | --- |
| API port / bind address | `8888` / `0.0.0.0` | Configurable |
| Per-request context limit | `1048576` | `4096..1048576` |
| Concurrent sequences | `6` | `1..6` |
| GPU memory utilization | `0.92` | `0.85..0.925`; only 0.92 is the reported profile |
| Prefill batch tokens | `2048` | `2048` or `3072`; 2048 is the reported profile |
| Long-prefill threshold | `2048` | `0`, or `1056..batch` |
| Parallelism / speculation | TP2, DCP2, DSpark K=3 | Fixed in this pinned release |
| KV backing | 1792 MiB display / GPU | Fixed; zero ordinary KV allocation |
| KV formats | FP4 main, MXFP4 indexer, FP8 sliding window | Fixed; original image visibility retained |

Defaults explicitly enable the tested 0.92 startup-admission exception. Actual
profiling, allocation checks and the **512 MiB host-memory watchdog** remain.
For another utilization, set `ALLOW_STARTUP_MEMORY_SHORTFALL=0`. Raising
utilization does not enlarge the fixed display KV pool. Six sequence slots do
not reserve six independent 1M contexts. See [configuration](docs/configuration.md).

Full native image visibility is retained; images are not disabled or restricted
to a text-sized attention window. Automatic tool choice and DeepSeek V4.1 tool
and reasoning parsers are enabled. The drafter download is an expert overlay
loaded by this runtime, **not a replacement for the target's original draft
configuration and non-expert weights**.

## Weights and reproducibility

- [EXL3 3bpw target](https://huggingface.co/coolbho3k/DeepSeek-V4.1-Flash-EXL3-3bpw)
- [EXL3 3bpw DSpark draft experts](https://huggingface.co/coolbho3k/DeepSeek-V4.1-Flash-DSpark-EXL3-3bpw)

`recipe-lock.json` pins model, draft, runtime image and cache revisions/hashes.
The native serving payload and corresponding source are in `release/runtime/`.
This is a specialized runtime; stock vLLM cannot load the complete recipe by
simply pointing at the checkpoint. "3bpw" describes the EXL3 expert
quantization, not a claim that every tensor—including vision—is three-bit.

The release checkout intentionally excludes calibration intermediates, private
receipts, downloaded weights, credentials and the old campaign launch scripts.
The original campaign files may still exist in the author's working directory,
but `.gitignore` allowlists only the public recipe. Maintainers should use the
[release checks and export procedure](docs/release-maintenance.md) before pushing.

## Measured results and limits

Current **GHCR rc2 / v106** measurements, September 17, 2026:

- Matched 12-request suite: **29.51 decode tok/s** pooled, up from 28.27 on
  the baseline. T=0 content medians ranged from **25.88 to 35.14 tok/s**;
  the published cooperative C1 prompt reached **35.67 tok/s** median.
- Uncached 32,766-token prefill: **978.57 input tok/s**, 33.54 seconds to first
  token. This observation was not an improvement over the preceding runtime.
- Six short concurrent requests, native images and automatic tool calling
  passed. Acceptance/output changes affect the decode comparison; these small
  samples do not establish general quality parity or statistical significance.

See [measurement details](docs/kernel-batch-performance.md).

**Historical capacity evidence — preceding runtime v103**, September 17, 2026.
Weights, KV allocation and safety settings are unchanged, but the following
long-context test has **not** been repeated with rc2's attention/top-k kernels:

- **3,146,968 independent input tokens across six simultaneous sessions**, with
  **3,161,062 total resident tokens** at the final observation. The largest
  prompt contained 1,031,997 tokens. All six retrieved the three test codes;
  there were zero request errors or preemptions.
- The capacity run took **80.6 minutes**. Once all prefills finished, a short
  15-second window measured **50.8 tokens/s aggregate**, on highly repetitive
  synthetic output. This is not a normal-chat throughput guarantee or a long
  decode endurance test.
- A short single-session prose sample measured **23.2 decode tokens/s**. An
  earlier varied target+quantized-draft suite averaged about **30.5 tokens/s**;
  it used an earlier memory configuration and is not an apples-to-apples claim.
- First content for the 1.032M-token prompt took **31.5 minutes** (about 546
  input tokens/s including admission). Large concurrent prefills substantially
  delay decoding. Do not infer million-token prefill speed from 32K benchmarks.

These are functional/capacity and workload-specific performance measurements,
not broad model-quality certification. The Hugging Face cards describe the
small calibration/evaluation campaigns and their limitations. Different
firmware, drivers, background processes, fabrics and prompts can change results.

## License

Recipe integration and MiaAI-derived code: **AGPL-3.0-only**. Preserve attribution,
source availability and upstream notices when modifying or redistributing.
Model weights and separately licensed upstream components retain their own
terms. See [LICENSE](LICENSE), [CREDITS.md](CREDITS.md), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
