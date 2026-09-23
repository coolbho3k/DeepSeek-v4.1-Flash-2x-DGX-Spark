#!/usr/bin/env bash
# Run bench.py on both Sparks for one NCCL variant. Usage: run.sh LABEL [VAR=VALUE ...]
set -euo pipefail
label=$1; shift
BASE=(NCCL_MAX_CTAS=8 NCCL_CROSS_NIC=0 NCCL_IB_ADDR_FAMILY=AF_INET NCCL_LL128_BUFFSIZE=262144
  NCCL_NVLS_ENABLE=0 NCCL_DEBUG=WARN NCCL_IGNORE_CPU_AFFINITY=1 NCCL_BUFFSIZE=1048576
  NCCL_PROTO=^LL128 'NCCL_IB_HCA==rocep1s0f1:1,roceP2p1s0f1:1' NCCL_IB_ADDR_RANGE=10.100.32.0/23
  NCCL_IB_DISABLE=0 NCCL_NET=IB NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_GID_INDEX=3
  NCCL_IB_ROCE_VERSION_NUM=2 NCCL_IB_MERGE_NICS=1 NCCL_CUMEM_ENABLE=0 NCCL_MAX_NCHANNELS=8
  GLOO_SOCKET_IFNAME=enp1s0f1np1)
envs=()
for kv in "${BASE[@]}" "$@"; do envs+=(--env "$kv"); done
common=(--rm --runtime=runc --gpus=all --pull=never --network=host --ipc=host --device=/dev/infiniband
  --ulimit memlock=-1:-1 --user=1000:1000 --tmpfs /tmp:rw,exec,size=1g --env HOME=/tmp
  --entrypoint /opt/ds41-venv/bin/python)
ssh emi@10.100.32.2 docker run "${common[@]}" "${envs[@]}" -v /tmp/ds41-bench-fc.py:/b.py:ro -v /tmp/ds41-libfastcomm.so:/lib.so:ro \
  sha256:314893911de4009c4e989408d17d48ec0f050bcd972e9683adf02e6cc38a8e58 /b.py --rank 1 --master 10.100.32.1 --lib /lib.so >/dev/null 2>&1 &
docker run "${common[@]}" "${envs[@]}" -v /tmp/ds41-bench-fc.py:/b.py:ro -v /tmp/ds41-libfastcomm.so:/lib.so:ro \
  -v /home/emi/code/ds41/reports/nccl-small-v1:/out \
  sha256:a5ef1cecb16259d16e49578c334b05f60dc58873eeac94cc4a29c5c246d0bcbf /b.py --rank 0 --master 10.100.32.1 --lib /lib.so 2>&1 | grep '^{' || true
wait
