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
> GPU from the approximately 2 GiB display-reserved region**. Keep the UEFI/BIOS
> display-memory reservation at **2 GB**, as in the tested setup—not zero. Both
> hosts need `nvidia_drm modeset=1 fbdev=0` and an accessible NVIDIA DRM card.
> Follow [step 2 below](#2-set-up-headless-display-memory-on-both-sparks) **before
> starting**; it includes persistent and no-reboot instructions. The launcher
> never changes drivers, boot settings or networking.
> This is experimental and driver-dependent; a headless setup is strongly
> recommended, not a guarantee against unified-memory exhaustion.
> **There is no driver-version allowlist.** Our baseline uses `580.173.02`;
> `595.84` also passed a local mixed-driver startup and generation check.
> Another ASUS setup reported a failure on `595.84`, so compatibility still
> depends on the kernel, firmware and DRM mapping. The launcher checks that
> each host's loaded driver and NVML agree; it never installs/downgrades a driver. See
> [driver compatibility](docs/display-memory.md#driver-compatibility).

> **Public prebuilt runtime:** `20260917-rc2` is available on
> [GHCR](https://github.com/users/coolbho3k/packages/container/package/deepseek-v4.1-flash-exl3-3bpw-2x-dgx-spark),
> pinned by digest in `recipe-lock.json` (**15.40 GiB compressed**, including
> kernel caches and corresponding source). Anonymous access is verified. This
> version includes the tested one-pass decode attention and exact length-aware
> top-k kernels. The Git-shipped serving overlay additionally enables tested
> **concurrent DCP attention** and **lossless packed Engram retrieval** using
> this same image. The tested native Engram reader and corresponding source
> ship in the Git overlay; no new Docker/C++ build or image download is needed.
> New Triton specializations may
> JIT-compile during warmup/first use because rc2's cache predates the overlay.
> `prepare` never falls back to a local image build.

> **Release candidate, not yet clean-install-qualified:** the current runtime
> passed short six-session, image, tool and 32K prefill checks. The 3.15M-token
> capacity test below used the preceding runtime and has not been repeated with
> the new kernels. The fresh-clone launcher has CPU-only validation; its clean
> two-node GPU boot is deferred until explicitly authorized.
> See [validation status](docs/release-validation.md).

The latest Engram trial measured **30.78 tok/s pooled serial decode** and
**about 1,078 tok/s on warmed, uncached 32K prefill**. Versus its fresh baseline,
decode improved about 1.4% and prefill was essentially unchanged, despite much
faster cold SSD retrieval. Images, tools and six short concurrent sessions
passed; KV allocation and memory limits are unchanged. Eleven of twelve
baseline replies matched exactly; one coding reply differed. See
[measurements, limitations and atomic upgrades](docs/engram-io-performance.md)
and the [preceding DCP attention results](docs/dcp-overlap-performance.md).

## What this recipe adds

- Our target quantizes **routed experts to EXL3 3bpw MUL1**, retaining the
  original formats for other weights, including native vision. “Full precision”
  here means **not further quantized by this recipe**, not that every original
  tensor is BF16 or FP32.
- Our DSpark draft-expert quant saves **0.975 GiB per GPU** relative to the
  original FP4 experts. In the 160-request comparison, draft acceptance was
  **53.17% versus 54.22%**: a **1.05 percentage-point** reduction, not a broad
  quality guarantee. See the [drafter model card](https://huggingface.co/coolbho3k/DeepSeek-V4.1-Flash-DSpark-EXL3-3bpw).
- **FP4 main KV, MXFP4 indexer and FP8 sliding-window KV**, with TP2/DCP2 and
  DSpark K=3. The six-session profile supports up to 1,048,576 tokens per request;
  historical testing reached about **3.15M input tokens in aggregate**, not 6M.
  Main KV uses the same 4.5-bit E2M1/E4M3 layout with **NVFP4 four-over-six**
  scale selection by default; `--fp4-kv-mode legacy` restores the previous writer.
  See the [accuracy checks](release/experimental/nvfp4_kv/README.md). Sliding-window KV
  defaults to **FP8 per 32 non-RoPE values with BF16 RoPE preserved**;
  `--swa-kv-group-size 64` restores the old grouping. Both layouts allocate the
  same page size. See [SWA accuracy and memory](release/experimental/swa_kv/README.md).
- An experimental display-buffer allocator supplies **1.75 GiB (1792 MiB) of KV
  backing per GPU** from the display reservation. The nominal remaining 256 MiB
  is not extra host-RAM safety margin. Images, reasoning and tool calling remain
  enabled.

Background: [the author's NVIDIA forum post](https://forums.developer.nvidia.com/t/deepseek-v4-1-flash-for-2x-dgx-spark-exl3-3bpw-3m-kv-cache-c6-new-2gb-free-ram-unlock/383583).
The setup instructions are all below; you do not need the forum thread or the
author's original workspace to run the recipe.

## What you need

- Two **128 GB NVIDIA DGX Sparks / GB10**, headless and otherwise idle. These
  instructions target Ubuntu-based DGX OS, with cgroup v2. The measured setup
  used NVIDIA driver **580.173.02**; a mixed **580.173.02 / 595.84** pair also
  passed a short serving canary. Other versions are allowed with a warning,
  not rejected solely by version. Do not blindly downgrade a working system.
- Working NVIDIA drivers, NVIDIA Container Toolkit and non-root Docker access
  on both. Python 3.11+, `ssh`, `rsync`, `ip` and `ss` on both; `git`, `curl` and
  `rdma` for the setup/checks below.
- Existing passwordless SSH from the head to the worker, with the host key
  already verified. Neither root SSH nor matching usernames/home directories
  are required. The launcher does not install SSH keys or change SSH policy.
- A working ConnectX RoCE connection; the measured setup used two logical
  200-Gbit/s rails. This does not require two physical cables: a Spark QSFP
  connector exposes two network interfaces. One or two rails can be configured;
  single-rail performance has not been qualified by the reported run.
- **At least 650 GiB free disk space on each host** for a fresh install. The
  target download is about 397 GiB, the separate drafter 4.76 GiB, plus the
  prebuilt runtime, extracted image, kernel caches and working-space reserve.
  The lossless packed Engram layout adds **97.66 GiB per host**; only each
  host's own rank is downloaded, with no local repacking or duplicate part files.
  Original Engrams are retained so old runners keep working.
- Internet access for the initial public Hugging Face and GHCR downloads on each host.
  **No HF token is needed.** Never give the launcher a write token.

Only the head needs this checkout. It copies the small runtime recipe to the
worker; each host downloads verified weights locally. No NFS mounts, SSH
tunnels, port-forwarding, firewall rules, systemd services or startup daemons
are installed. Existing fabric networking must already work.

## Setup from a fresh clone

The **head** is the Spark you will launch from and connect API clients to.
The **worker** is the other Spark. Only the head needs a clone. Follow these
steps in order; the administrator commands are manual, one-time preparation,
not changes the launcher makes for you.

1. [Check host access and tools](#1-check-host-access-and-tools).
2. [Set up headless display memory on both Sparks](#2-set-up-headless-display-memory-on-both-sparks).
3. [Identify and check the RoCE fabric](#3-identify-and-check-the-roce-fabric).
4. [Clone and configure on the head](#4-clone-and-configure-on-the-head).
5. [Check, download and start](#5-check-download-and-start).
6. [Connect a client](#6-connect-a-client).

### 1. Check host access and tools

On **both Sparks**, as the ordinary user that will run the recipe:

```bash
uname -m                                  # aarch64
python3 --version                         # 3.11 or newer
nvidia-smi
stat -fc %T /sys/fs/cgroup                 # cgroup2fs
docker info --format '{{.ServerVersion}}'  # must work WITHOUT sudo
nvidia-ctk --version
```

Stock DGX Spark software includes the NVIDIA Container Toolkit. If Docker
works only with `sudo`, an administrator can add the intended user with
`sudo usermod -aG docker "$USER"`; then log out and reconnect, including SSH
sessions, before checking again. Docker-group membership grants root-equivalent
access. **Do not run `start-server.sh` with sudo.** For missing/broken GPU
container support, use [NVIDIA's Docker setup instructions](https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html).

If the ordinary command-line tools are missing, install them on both hosts:

```bash
sudo apt-get update
sudo apt-get install git python3 rsync openssh-client openssh-server iproute2 curl
```

Before disabling either desktop, confirm that you can SSH into **both** machines
from another computer. Then set up head-to-worker SSH using your real worker
username and LAN hostname below. Reuse an existing key; only if you have none,
run `ssh-keygen -t ed25519` interactively and do not overwrite an existing key.
Verify the worker's host-key fingerprint through a trusted channel when first
connecting; do not disable host-key checking.

```bash
ssh-copy-id user@spark-worker.lan
ssh -o BatchMode=yes user@spark-worker.lan 'hostname; docker info --format "{{.ServerVersion}}"'
```

The second command must succeed without a password/passphrase prompt. For an
encrypted SSH key, unlock it in an agent on the head that remains available
after your terminal closes; an agent forwarded from a departing laptop is not
a durable credential for the controller. Reverse SSH and root SSH are not
needed, and the two users do not need matching names or home directories.

### 2. Set up headless display memory on both Sparks

**Do this on both hosts over a verified SSH connection, with all model servers
and other CUDA jobs stopped.** Save any desktop work first. This profile leaves
very little room for other workloads: disconnect monitors and avoid local or
remote graphical desktops. If the required state below is already present,
skip the changes and just verify it.

Keep the firmware's **display-reserved memory at 2 GB**, as used in the tested
setup. If you previously changed that reservation, restore it in UEFI/BIOS;
firmware changes require a reboot. Setting it to zero removes the reservation
this technique is intended to use. The commands below do not change firmware.

Record your original boot target and module settings for recovery:

```bash
systemctl get-default
sudo cat /sys/module/nvidia_drm/parameters/modeset
sudo cat /sys/module/nvidia_drm/parameters/fbdev
```

**Recommended persistent setup (reboot required):**

```bash
sudo systemctl set-default multi-user.target
sudoedit /etc/modprobe.d/ds41-display-kv.conf
```

Put this single line in that dedicated file:

```text
options nvidia_drm modeset=1 fbdev=0
```

Check for conflicting `nvidia_drm` options in existing `/etc/modprobe.d/` files
or boot configuration, preserving unrelated settings. Do **not** blacklist
`nvidia_drm` or add `nomodeset`: the allocator needs DRM modesetting enabled,
with only framebuffer-console support disabled. Then, on each idle Spark:

```bash
sudo update-initramfs -u -k all
sudo reboot
```

`set-default` changes future boots; it does not stop the current desktop.
After reconnecting, perform the verification below.

**Optional no-reboot activation (the method used during development):**

With CUDA jobs stopped, stop the graphical login/desktop if active:

```bash
if systemctl is-active --quiet display-manager; then
    sudo systemctl stop display-manager
fi
```

If the loaded module is not already `modeset=Y`, `fbdev=N`, use this guarded
reload. It requires both an empty compute-process list and a zero DRM module
reference count; if either check fails, **stop and investigate—do not force
unload or bypass the checks**.

```bash
(
    set -eu
    ds41_gpu_jobs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    test -z "$ds41_gpu_jobs"
    test "$(sudo cat /sys/module/nvidia_drm/refcnt)" = 0
    sudo rmmod nvidia_drm
    sudo modprobe nvidia_drm modeset=1 fbdev=0
)
```

If `nvidia_drm` is not loaded at all, only the final `modprobe` is needed, under
the same idle/headless prerequisites. Do not unload `nvidia` or `nvidia_uvm`.
If the reload fails, keep serving stopped; restore your recorded module settings
or reboot into a known-good boot configuration. This changes the running module
only: use the persistent setup above to retain the settings after a reboot.

**Verify the required state on both hosts:**

```bash
sudo cat /sys/module/nvidia_drm/parameters/modeset  # Y
sudo cat /sys/module/nvidia_drm/parameters/fbdev    # N
systemctl is-active display-manager          # inactive
nvidia-smi --query-compute-apps=pid --format=csv,noheader  # empty before startup
```

Identify each host's NVIDIA DRM card; NVIDIA's vendor ID is `0x10de`:

```bash
for ds41_card in /sys/class/drm/card[0-9]*; do
    if [ -r "$ds41_card/device/vendor" ]; then
        printf '%s vendor=' "${ds41_card##*/}"
        cat "$ds41_card/device/vendor"
    fi
done
```

For example, NVIDIA `card1` means `/dev/dri/card1`. Set `HEAD_DRM_CARD` and
`WORKER_DRM_CARD` accordingly in step 4; the values can differ. No Vulkan setup,
zram change or global cache flush is needed.

To restore a desktop later, first stop serving, restore the boot target you
recorded (usually `sudo systemctl set-default graphical.target`), remove only
your recipe-specific module option, restore any other settings you changed,
rebuild the initramfs and reboot. See also [display-memory details](docs/display-memory.md).

### 3. Identify and check the RoCE fabric

Connect the Sparks' ConnectX QSFP ports directly, retaining ordinary LAN access
for administration. On **both hosts**, inspect the active interfaces and their
HCA mappings:

```bash
ip -br address
rdma link show
```

Use the actual active interface/HCA names, including capitalization; your cable
may use different ports from the examples. A QSFP connector exposes two logical
network interfaces; two rails do not necessarily mean two cables. See
[NVIDIA's connection guide](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks)
for cabling and OS-specific persistent network configuration.

Keep an already working fabric configuration. If the **dedicated, directly
cabled fabric interfaces** have no IPv4 addresses yet, the following is an
optional temporary setup. Substitute your actual interfaces and first confirm
these subnets are unused; **do not apply it to your LAN/SSH interface**.

On the **head**:

```bash
sudo ip link set enp1s0f1np1 up
sudo ip address add 10.100.32.1/24 dev enp1s0f1np1
sudo ip link set enP2p1s0f1np1 up
sudo ip address add 10.100.33.1/24 dev enP2p1s0f1np1
```

On the **worker**:

```bash
sudo ip link set enp1s0f1np1 up
sudo ip address add 10.100.32.2/24 dev enp1s0f1np1
sudo ip link set enP2p1s0f1np1 up
sudo ip address add 10.100.33.2/24 dev enP2p1s0f1np1
```

Do not add addresses that already exist. For one rail, omit the secondary
interface's two commands. These manual addresses are not persistent across
reboots and may be replaced by your network manager; use your OS's normal
network configuration to persist them. The launcher never configures them.

From the head, test each configured rail using its matching interface:

```bash
ping -c 3 -I enp1s0f1np1 10.100.32.2
ping -c 3 -I enP2p1s0f1np1 10.100.33.2  # omit for one rail
```

IP ping alone does not verify RDMA. Run this read-only Python snippet on **both
hosts** to find the RoCE v2 IPv4 GID index for each interface/IP:

```bash
python3 - <<'PY'
import ipaddress
from pathlib import Path
for port in sorted(Path('/sys/class/infiniband').glob('*/ports/*')):
    for gid in sorted((port / 'gids').iterdir(), key=lambda p: int(p.name)):
        address = ipaddress.IPv6Address(gid.read_text().strip()).ipv4_mapped
        kind = (port / 'gid_attrs/types' / gid.name).read_text().strip()
        if address is not None and kind == 'RoCE v2':
            netdev = (port / 'gid_attrs/ndevs' / gid.name).read_text().strip()
            print(f'HCA={port.parent.parent.name} port={port.name} '
                  f'interface={netdev} GID_INDEX={gid.name} IPv4={address}')
PY
```

Match the output to each configured fabric IP, not the LAN IP. The current
launcher requires port **1**, RoCE **v2**, IPv4-mapped GIDs, and a **common GID
index on all selected rails on both hosts** (often `3`; do not assume it).
Resolve missing/mismatched entries before proceeding. There is no automatic
TCP fallback.

### 4. Clone and configure on the head

Clone this repository, enter its directory, and run everything below on the
Spark that should host the API:

```bash
git clone https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark.git
cd DeepSeek-v4.1-Flash-2x-DGX-Spark
cp .env.ds41.example .env.ds41
```

Edit `.env.ds41` with your editor. This is trusted shell configuration, so do
not source someone else's unreviewed file or put credentials in it. Set your
worker's SSH address, the fabric values you verified above and both DRM cards.
For the two-rail address example above, the network portion would be:

```bash
WORKER_HOST=user@spark-worker.lan
HEAD_FABRIC_IP=10.100.32.1
WORKER_FABRIC_IP=10.100.32.2
FABRIC_NETWORK=10.100.32.0/23
HEAD_IFNAME=enp1s0f1np1
WORKER_IFNAME=enp1s0f1np1
HEAD_HCA=rocep1s0f1
WORKER_HCA=rocep1s0f1
ROCE_GID_INDEX=3
HEAD_SECONDARY_IP=10.100.33.1
WORKER_SECONDARY_IP=10.100.33.2
HEAD_SECONDARY_IFNAME=enP2p1s0f1np1
WORKER_SECONDARY_IFNAME=enP2p1s0f1np1
HEAD_SECONDARY_HCA=roceP2p1s0f1
WORKER_SECONDARY_HCA=roceP2p1s0f1
HEAD_DRM_CARD=/dev/dri/card0
WORKER_DRM_CARD=/dev/dri/card0
```

**Replace the username, hostname, devices and addresses with your own.**
`WORKER_HOST` can use the worker's ordinary LAN address; it need not be its
fabric address. `FABRIC_NETWORK` is a filter containing every selected fabric
IP: the example `/23` covers the two `/24` rail subnets. Setting these variables
does not create interfaces, assign addresses or change routing.

For **one rail**, clear all six secondary fields in `.env.ds41`:

```bash
HEAD_SECONDARY_IP=
WORKER_SECONDARY_IP=
HEAD_SECONDARY_IFNAME=
WORKER_SECONDARY_IFNAME=
HEAD_SECONDARY_HCA=
WORKER_SECONDARY_HCA=
```

Start with the supplied serving defaults: utilization `0.92`, six sequence
slots, 1,048,576-token context limit, 2048-token prefill batches/threshold and
`ALLOW_STARTUP_MEMORY_SHORTFALL=1`. The default API listens on the head's LAN
port **8888**. Keep both hosts otherwise idle; the memory margin is small.

Fixed K3 now uses **probabilistic draft sampling** by default. The selected
shared-row projection and fused prefill gather are also enabled, alongside
four-over-six NVFP4 main KV and group-32 FP8 sliding KV with BF16 RoPE. See
[fusion measurements](release/experimental/model_fusion/RESULTS.md) and
[sampling measurements and limits](release/experimental/draft_sampling/RESULTS.md).

**Choose storage before downloading.** By default, the head uses this checkout's
`.assets/`, and the worker uses its SSH user's `~/.cache/ds41/`. If you want a
different disk, set dedicated, absolute paths in `.env.ds41`, for example:

```bash
DS41_CACHE_DIR=/mnt/models/ds41
REMOTE_CACHE_DIR=/mnt/models/ds41
REMOTE_DIR=/mnt/models/ds41/recipe
```

Only use paths that actually reside on your intended mounted disk and are
writable by the respective ordinary user; the paths may differ between hosts.
Do not point these at an unrelated checkpoint directory. Docker's image storage
is separate: `docker info --format '{{.DockerRootDir}}'` shows its location.
Use `df -h` on both hosts to check the filesystems holding **both** the asset
cache and Docker storage. Budget at least 550 GiB free per host for a new
installation, with at least 32 GiB remaining on the runs filesystem at startup.
Changing `DS41_CACHE_DIR` alone does not move Docker's images off the root disk.

### 5. Check, download and start

On the head, from the checkout:

```bash
./start-server.sh --dry-run   # resolved configuration; no network or GPU work
./start-server.sh doctor     # read-only checks on both configured hosts
```

**Read the doctor's output, not just its exit code.** It reports host properties;
it does not reject every unsupported value. Check `aarch64`, Python 3.11+,
non-root users, all required tools present, DRM `modeset=Y`/`fbdev=N`, the chosen
card present, working Docker and the expected fabric addresses. Complete the
GID checks in step 3 as well. If the ordinary-user probe reports permission
denied for a module parameter, verify it with the `sudo cat` checks in step 2.
The stricter startup preflight happens after asset
preparation, so resolve these issues before downloading hundreds of GiB.

With both GPUs idle, continue:

```bash
./start-server.sh prepare    # download pinned assets, verify, pull prebuilt image
./start-server.sh            # start both nodes and wait for readiness
```

No Hugging Face token, GHCR login or host Python package installation is needed.
The launcher downloads the target **and** separate draft-expert overlay; keep
the target's original draft configuration/non-expert weights as well. Preparation
refuses busy GPUs; it does not stop another server to make room.

The first preparation is a large download. Ordinary partial downloads can
resume, and completed payloads are SHA-256 verified. Some interrupted staging
states need manual inspection; see [troubleshooting](#troubleshooting).
Repeat launches reuse verified files.
Runtime/cache versions are stored separately, so recipe upgrades preserve old
runtime assets and reuse the unchanged model and drafter downloads.
The distribution is designed to require **no manual quantization, Docker build
or kernel preparation**: native binaries and a pinned prebuilt kernel cache
are supplied with the runtime assets. Normal CUDA initialization and graph
capture still happen at startup. Complete cold-start cache coverage—and the
absence of any first-use JIT compilation—must still be checked in the pending
clean-install test. There is no automatic image-build or quantization fallback.

Wait for the **Ready** message. Model loading and CUDA graph capture take time;
there is no qualified fresh-install elapsed-time guarantee yet. In another
terminal use `./start-server.sh logs --controller` and `./start-server.sh logs`
to watch progress. If startup's observation window expires, inspect the same
controller rather than starting a second copy. Editing `.env.ds41` does not
change an already running server; applying it requires an explicit restart.

### 6. Connect a client

The public API is available directly at `http://<head-LAN-IP>:8888/v1`.
The default model name is `deepseek-v41-flash-exl3`. This is an **API server, not
a web chat UI**; `/` is not the health endpoint. On the head, check:

```bash
curl --fail http://127.0.0.1:8888/health
curl --fail http://127.0.0.1:8888/v1/models
```

From a LAN client, replace `<head-LAN-IP>` with the head's actual LAN IP or
hostname (and the port if you changed it):

```bash
curl --fail "http://<head-LAN-IP>:8888/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v41-flash-exl3","messages":[{"role":"user","content":"Hello!"}],"max_tokens":128}'
```

The API has **no authentication by default**. Keep it on a trusted private LAN;
use your own authenticated gateway before exposing it beyond that network.
For local-only access, set `API_HOST=127.0.0.1`.
In an OpenAI-compatible client, use the `/v1` base URL and model name above.
If the client insists on an API key, a dummy value such as `not-needed` satisfies
the client but **does not add authentication**. Automatic tool choice and the
DeepSeek V4.1 tool/reasoning parsers are already enabled. No tunnel is needed.

### Reasoning effort

Reasoning is **enabled by default at `high` (75/100)**. Set the request's
top-level `reasoning_effort` to one of these names:

| Setting | Numeric effort / behavior |
| --- | --- |
| `none` | Thinking off |
| `minimal` | 25 |
| `low` | 50 (DeepSeek native) |
| `medium` | 60 |
| `high` | 75 (DeepSeek native; default) |
| `xhigh` | 90 |
| `max` | 100 (DeepSeek native) |

`minimal`, `medium`, and `xhigh` are this recipe's compatibility aliases;
DeepSeek's native `low`, `high`, and `max` values are unchanged. For a custom
integer from **1 through 100**, put it in `chat_template_kwargs` instead:

```json
{
  "model": "deepseek-v41-flash-exl3",
  "messages": [{"role": "user", "content": "Explain your approach."}],
  "chat_template_kwargs": {"reasoning_effort": 60}
}
```

Use either the named top-level setting or the numeric template setting.
The number is a prompt instruction encouraging more thorough reasoning, **not
a hard reasoning-token limit or percentage of compute**. `none` disables
thinking; numeric 1 is still thinking mode. The mapping ships in the recipe's
Python overlay, so it needs no image rebuild or local compilation. Updating
the checkout does not change an already running server; the new mapping takes
effect on a subsequent launch from the updated recipe.

## Everyday use

```bash
./start-server.sh status
./start-server.sh logs                    # head inference/throughput logs
./start-server.sh logs --node 1           # worker logs
./start-server.sh logs --controller       # startup checks and RAM watchdog
./stop-server.sh                          # only this recipe's recorded pair
./start-server.sh --restart               # explicit stop + fresh launch
```

Startup detaches a small **plain Python controller**, not a system service.
It watches the two exact containers and available host RAM. The server stays
up when startup returns or your terminal closes; Ctrl+C while waiting for
readiness or viewing logs stops only the wait/viewer. To stop serving, use the
explicit `stop` command. Nothing is configured to start automatically on boot.
`./stop-server.sh` is a convenience alias for `./start-server.sh stop`; it stops
the recorded pair on both hosts and preserves models, caches and logs. Use
`./stop-server.sh --dry-run` to check the action without stopping anything.
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
| Prefix-cache retention interval | `4096` | `0` (semantic-only), or multiples of `256` through `1048576` |
| Parallelism / speculation | TP2, DCP2, DSpark K=3 | Fixed in this pinned release |
| KV backing | 1792 MiB display / GPU | Fixed; zero ordinary KV allocation |
| KV formats | 4.5-bit NVFP4 main, MXFP4 indexer, FP8 sliding window | Original image visibility retained |
| Main KV writer | `nvfp4_4over6` | `--fp4-kv-mode legacy` restores the previous writer |
| Sliding-window KV | FP8 group 32 / BF16 RoPE | `--swa-kv-group-size 64` restores group 64 / BF16 |

Defaults explicitly enable the tested 0.92 startup-admission exception. Actual
profiling, allocation checks and the **512 MiB host-memory watchdog** remain.
For another utilization, set `ALLOW_STARTUP_MEMORY_SHORTFALL=0`. Raising
utilization does not enlarge the fixed display KV pool. Six sequence slots do
not reserve six independent 1M contexts. See [configuration](docs/configuration.md).

Agent cache reuse: `PREFIX_CACHE_RETENTION_INTERVAL=4096` retains periodic
sliding-window checkpoints inside the same evictable KV pool. Set it to `0`
to restore semantic-only retention. Both ranks enable cached-token reporting
(`usage.prompt_tokens_details.cached_tokens`), and Responses API `input_text`
and `output_text` blocks are supported. These are credited MiaAI/mrexodia
[PR12/PR21 adaptations](docs/upstream-api-cache.md), not a decode kernel change.

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

## How the display-reserved memory becomes KV

Normally this region is reserved for display buffers rather than ordinary
CUDA allocations. With DRM modesetting enabled and the framebuffer console
disabled, our allocator creates a display buffer, maps it into the process and
registers that mapping for CUDA access. The runtime wraps the resulting GPU
pointer as tensor storage and uses it for KV, keeping the allocation alive for
as long as the cache needs it. No image is rendered to a monitor.

The implementation is included in the clone:

- [Native allocator: `display_kv.c`](release/runtime/sources/display_kv.c)
  creates/maps the DRM buffer and registers it with CUDA.
- [Serving integration: `display_kv.py`](release/runtime/serving/ds41/display_kv.py)
  exposes the allocation to the tensor/KV path with the required ownership.

This profile uses **1792 MiB display-backed KV per GPU and zero ordinary KV
backing**. It does not add physical RAM, increase the ordinary `cudaMalloc`
budget, or make the reservation accessible to every application. Other recipes
need allocator integration, ownership/lifetime handling and memory accounting;
setting the module flags alone is not enough.

The nominal 256 MiB left within a 2 GiB reservation is separate from the host
memory watchdog's margin. This path had lower raw read bandwidth than ordinary
CUDA-backed memory in our probes, but enabled the larger usable KV allocation.
It is experimental: do not assume identical capacity or zero performance cost
on other drivers, firmware or workloads. The watchdog cannot guarantee recovery
from a hard unified-memory lockup.

## Measured results and limits

The forum post's rough **25–40 tok/s** single-stream and **over 50 tok/s aggregate**
observations are workload-dependent, not promises of 50 tok/s for one chat.
Comparisons there with GLM or MiaAI's published numbers are not matched
cross-recipe benchmarks. The more tightly scoped measurements follow.

Current **concurrent DCP overlay** measurements, September 18, 2026:

- **30.52 decode tok/s** pooled over 12 serial requests, and **1,026 input
  tok/s** on fully uncached 32K prefill. Easy-prose T=0 median was **30.10 tok/s**.
- About **3.4% decode / 4.8% prefill** above the saved v106 results below. This
  historical comparison is not a fresh controlled A/B; ten of twelve replies
  and acceptance fractions matched exactly, while two differed.
- Images, tools, multilingual generation and six simultaneous short requests
  passed. KV and memory limits are unchanged; no new million-token run was done.

See [overlap measurements and upgrade notes](docs/dcp-overlap-performance.md).

Preceding **GHCR rc2 / v106** measurements, September 17, 2026:

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
long-context test has **not** been repeated with rc2's attention/top-k kernels
or the new concurrent DCP overlay:

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

## Troubleshooting

Start with these read-only checks on the head:

```bash
./start-server.sh status
./start-server.sh logs --controller --no-follow
./start-server.sh logs --no-follow
./start-server.sh logs --node 1 --no-follow
```

Before a deployment/container exists, its corresponding log/status command may
not be available. Read the error printed by `prepare` or the controller log
first; startup does not always reach container creation.

The launcher sets each rank's `VLLM_HOST_IP` to its **own primary fabric IP**.
NCCL/Gloo interface selection alone does not select vLLM's ZeroMQ control
address. Both Sparks need bidirectional TCP reachability on that fabric,
including dynamically selected control ports—not just a working RoCE link or
the configured master port. No firewall or routing changes are made for you.
NCCL `INFO` logging is enabled for transport diagnosis. While waiting for
readiness, the controller adds bounded, fixed-text startup hints after three
minutes and then every five minutes; hints are not proof of the failure cause.
These do not shorten the existing startup deadline or change RAM safeguards.

| Symptom | What to check |
| --- | --- |
| SSH or Docker permission failure | Run the step 1 `BatchMode` SSH/Docker check using the exact `WORKER_HOST`. Reconnect after a Docker-group change. Do not use root SSH or `sudo start-server.sh`. |
| GPU busy / another server detected | Stop that workload using its own controls, then retry. This launcher will not adopt or terminate unrelated containers. |
| Display allocator unavailable | Check both hosts' `modeset=Y`, `fbdev=N`, actual NVIDIA DRM cards and 2 GB firmware reservation. Do not reload modules under a running server. |
| Driver mismatch / `register display IO: CUDA_ERROR_INVALID_VALUE` | Check that the **loaded** driver and `nvidia-smi` agree on each host. For registration failures, also check `modeset=1 fbdev=0`, the DRM card, kernel and firmware; see [driver compatibility](docs/display-memory.md#driver-compatibility). `595.84` is allowed; a version alone does not establish the cause. Do not increase utilization to address this error. |
| Both ranks stop after `Using ['PYNCCL'] all-reduce backends`, before loading weights | Inspect the controller's `startup_control_plane` / `startup_diagnostics` records and both rank logs. Verify the per-rank fabric IPs and TCP reachability, including dynamic ports; a fast RDMA benchmark does not test ZeroMQ's ready handshake. Wrong control addresses are one possible cause, not a confirmed diagnosis of every hang. |
| RoCE/GID preflight fails | Match every rail's IP, interface, case-sensitive HCA, port 1 and RoCE v2 IPv4 GID index on both hosts. Ordinary LAN connectivity is insufficient. Clear all six secondary fields for one rail. |
| Disk space failure | Check both hosts' asset and Docker filesystems; startup also requires 32 GiB free on the runs filesystem. Do not broadly prune Docker or delete unrelated checkpoints. |
| RAM preflight/watchdog failure | Stop background jobs/desktops and inspect available RAM on both hosts. Do not raise utilization, bypass checks or globally flush caches as the first remedy. |
| Interrupted preparation | Retry `prepare` for ordinary partial downloads. Existing mismatched files, `.copying` or `.extracting` staging directories may require inspection before retrying; do not wipe the whole cache. |
| HTTP 416 while resuming a download | A `.download-part` may already contain the complete file. Verify its size and SHA-256 against the pinned manifest before recovering it; do not blindly rename or discard it. This edge case is not automatically recovered yet. |
| SHA-256/manifest mismatch | Check the named file, available disk space and pinned revision. Do not edit the manifest to accept an unexplained mismatch. |
| Local API works, remote API does not | Use the head's LAN address, configured port and `/health` or `/v1/models`, not `/`. Check `API_HOST` is not loopback-only. The launcher creates no tunnel or firewall rules. |
| An `auto` tool-choice/parser error | Verify the client is hitting this deployment/model, not an older server on another port. This recipe already supplies the auto-tool-choice and DeepSeek V4.1 parser options. |

After fixing the cause, `./start-server.sh --restart` stops and relaunches only
this launcher's recorded pair, preserving weights, caches and logs. For a bug
report, include the Git revision, driver versions, sanitized configuration and
relevant error/log excerpt—not tokens, SSH keys or an unfiltered environment dump.

## License

Recipe integration and MiaAI-derived code: **AGPL-3.0-only**. Preserve attribution,
source availability and upstream notices when modifying or redistributing.
Model weights and separately licensed upstream components retain their own
terms. See [LICENSE](LICENSE), [CREDITS.md](CREDITS.md), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
