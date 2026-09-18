# Configuration and operating notes

Copy `.env.ds41.example` to `.env.ds41`; it is sourced as trusted shell code.
Keep secrets out of it. `DS41_ENV_FILE` selects a different configuration file.
The file overrides inherited environment values; explicit CLI arguments override
the file. Run `./start-server.sh --help` and `--dry-run` to inspect the interface.

## Fabric and host identity

`WORKER_HOST` is an existing SSH endpoint, optionally `user@hostname`. It can be
the management LAN address; the `*_FABRIC_IP` values are the separate GPU
communication addresses. The head always hosts the API. Its SSH username,
checkout path and UID/GID need not match the worker's.

Set each host's `*_IFNAME`, `*_HCA`, and `ROCE_GID_INDEX` to the actual active
RoCEv2 link. Existing addresses must lie in `FABRIC_NETWORK`. For two links,
configure the six `*_SECONDARY_*` fields and a subnet covering both rails. For
one link, leave all six blank. The launcher checks addresses/GIDs; it never runs
network setup commands or rewrites Netplan, routes, firewall rules or NFS exports.

The measured configuration used two 200-Gbit/s links. The single-link command
path has CPU coverage only; do not assume identical bandwidth or performance.
TCP-over-ordinary-Ethernet fallback is not provided by this pinned recipe.

## Storage and downloads

`DS41_CACHE_DIR` defaults to `.assets` inside the head checkout.
`REMOTE_CACHE_DIR` defaults to the worker SSH user's `~/.cache/ds41`.
`REMOTE_DIR` defaults to a versioned recipe under that worker cache. Overrides
must be absolute dedicated paths. Different local and remote paths are supported.

The recipe downloads the original target's `draft/` non-expert weights **and**
the separate EXL3 expert overlay. Do not replace one directory with the other.
Each host has its own model verification receipt. The runtime/image/kernel-cache
identities are pinned; edits or missing files fail checks rather than being
silently repaired or compiled.

`prepare` requires idle GPUs on both hosts. This deliberately avoids a second
large download/hash job beside a resident model. It does not stop existing GPU
jobs. No global page-cache clearing, swap tuning or automatic disk cleanup occurs.

## Serving knobs

CLI overrides: `--port`, `--host`, `--worker`, `--gpu-memory-utilization`,
`--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens`, and
`--long-prefill-token-threshold`. These are launch-time settings: editing the
file does not mutate a live server. Use an explicit `--restart` when appropriate.

The tested defaults are 0.92 utilization, six sequence slots, a 1,048,576-token
per-request limit, and 2048-token prefill chunks. Both 2048 and 3072 chunks are
supported by the implementation, but larger chunks need more temporary memory.
The image-safe minimum is retained; arbitrary tiny prefill batches are refused.

`LONG_PREFILL_TOKEN_THRESHOLD=0` disables that fairness threshold; otherwise use
1056 through the configured batch size. Scheduler changes alter the tradeoff
between prompt admission and interactive decode. They are not free speedups.

The startup memory exception is explicitly opted into with
`ALLOW_STARTUP_MEMORY_SHORTFALL=1` and is scoped to 0.92. Set it to 0 for other
utilizations. This is not an unlimited allocator override: native profiling,
KV bounds and the 512-MiB host watchdog remain. This release's fixed 1.75-GiB
display pool does not grow when utilization is raised.

TP2/DCP2, DSpark K=3, target/draft quantization, graph shapes, CPU cgroup limits,
indexer workspace and display allocation size are pinned runtime choices, not
advertised arbitrary knobs. Supporting a materially different profile needs
its own runtime validation. Full native vision and automatic tools remain on.

## Ownership, logs and shutdown

The launcher records an immutable per-run deployment and exact container
identities. A detached Python controller supervises that pair; it is not a
system service. The controller stops its own pair on a worker failure, startup
failure or low-memory condition, and does not automatically restart it.

`status`, `logs`, and `stop` use the saved configuration, so they still refer to
the right pair after you edit `.env.ds41`. If no public-launcher state exists,
`stop` does nothing to other servers. A recycled process ID cannot be signalled
as the old controller. Keep `.state/public/` and the run directory until the
owned pair is stopped. Do not delete them as a way to bypass a failed launch.

`logs` follows inference output; add `--node 1` for the worker,
`--controller` for startup/watchdog output, or `--no-follow` for a snapshot.
`--no-wait` starts in the background without waiting for readiness. Interrupting
the readiness wait or a log viewer does not stop the server.

After a failed startup, inspect controller and both worker logs. Resolve the
reported issue, then use `--restart` for a fresh owned pair. No broad
`docker stop $(docker ps ...)`, container pruning, driver reload, or automatic
network repair is part of this recipe.
