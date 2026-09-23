# Zero-copy RDMA through GPU-visible host memory (2026-09-23)

Both Sparks run driver 595.84 on kernel 6.17.0-1029-nvidia (spark-0: ASUS GX10,
head, 10.100.32.1; spark-1: NVIDIA DGX Spark, worker, 10.100.32.2).

GPUDirect RDMA is unavailable on GB10: CUDA reports GPU_DIRECT_RDMA_SUPPORTED=0
and DMA_BUF_SUPPORTED=0, `ibv_reg_mr` fails with EFAULT on cudaMalloc and
cudaMallocManaged memory, and NCCL 2.29.7 logs `GDR 0` even with
NCCL_NET_GDR_C2C=1, NCCL_NET_GDR_LEVEL=SYS and NCCL_DMABUF_ENABLE=1. The kernel
dma-buf code is gated by `nv->dma_buf_supported`, set in the resource manager.
Pinned (`cudaMallocHost`) and pageable (`malloc`) memory register fine, and GB10
reports native GPU access to pageable host memory. Each ConnectX-7 function is
behind PCIe Gen5 x4 (the ~20 GB/s dual-rail ceiling is hardware).

`pingpong.c`: RC RDMA WRITE ping-pong over RoCEv2 (GID 3, rocep1s0f1), one
registered buffer per side, receiver busy-polls the message's last 8 bytes.
No driver, kernel, package or serving change. 2,000 timed iterations each.

| Bytes | malloc one-way p50 | cudaMallocHost one-way p50 | p99 |
|---:|---:|---:|---:|
| 8 | 2.07 µs | 2.03 µs | ≤ 2.4 µs |
| 40,960 | 6.73 µs | 6.70 µs | ≤ 7.4 µs |
| 163,840 | 16.20 µs | 16.11 µs | ≤ 16.9 µs |
| 1,048,576 | 81.94 µs | 81.90 µs | ≤ 82.4 µs |

For comparison, NCCL at decode sizes: 40 KB all-reduce 38–46 µs, 8 KB
all-gather ~29 µs. A one-hop two-rank exchange on these buffers (GPU-flagged
CPU sender, GPU polling the coherent receive buffer) is projected at
~10–12 µs per small collective — not yet built or measured.
