// SPDX-License-Identifier: AGPL-3.0-only
// Two-rank collectives over RDMA into GPU-visible pinned host memory (GB10).
//
// GPUDirect RDMA is unavailable on GB10, but pinned host memory is registered
// by the NIC and read/written coherently by the integrated GPU. Each call:
//   stage kernel : copy the local input into this channel's send slot, then the
//                  last block publishes (seq, slot, bytes) in a host doorbell.
//   proxy thread : sees the doorbell, posts an RDMA WRITE of the slot into the
//                  peer's receive slot followed by an inline 8-byte flag WRITE
//                  on the same RC QP (memory registered without relaxed
//                  ordering, so the flag lands after the data).
//   finish kernel: waits for the peer's flag == seq, then adds (all-reduce) or
//                  places (all-gather) the peer's data.
// Kernels are CUDA-graph capturable: sequence numbers live in device memory.
// Two-rank BF16 sums are a single commutative add in FP32 rounded to BF16, so
// both ranks produce identical results (and match NCCL's two-rank sum).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>

namespace {
constexpr int kChannels = 2, kSlots = 4;
constexpr int kSignalEvery = 32;

struct alignas(64) Doorbell { volatile uint64_t seq; volatile uint32_t slot; volatile uint32_t bytes; };

struct DeviceView {
  uint8_t* send;             // host pinned [ch][slot][max_bytes]
  uint8_t* recv;             // host pinned [ch][slot][max_bytes]
  uint64_t* recv_flags;      // host pinned [ch][slot] (64 B apart)
  Doorbell* doorbells;       // host pinned [ch]
  uint64_t* seq;             // device [ch]
  unsigned int* arrivals;    // device [ch]
  uint64_t max_bytes;
};

struct Comm {
  ibv_context* ctx = nullptr; ibv_pd* pd = nullptr; ibv_cq* cq = nullptr; ibv_qp* qp = nullptr;
  ibv_mr* mr = nullptr;
  uint8_t* host = nullptr; size_t host_bytes = 0;
  DeviceView view{};
  uint64_t peer_recv = 0, peer_flags = 0; uint32_t peer_rkey = 0;
  int gid_index = 0;
  std::thread proxy; std::atomic<bool> stop{false}; std::atomic<int> error{0};
  uint64_t posted = 0;
};

struct Info { uint32_t qpn, psn, rkey, pad; uint64_t recv, flags; ibv_gid gid; };

__device__ __forceinline__ uint64_t load_acquire_sys(const uint64_t* p) {
  uint64_t v; asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void store_release_sys(uint64_t* p, uint64_t v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}

__global__ void stage_kernel(DeviceView v, const uint8_t* src, uint64_t bytes, int ch) {
  const uint64_t seq = v.seq[ch] + 1;
  const int slot = int(seq % kSlots);
  uint8_t* dst = v.send + (uint64_t(ch) * kSlots + slot) * v.max_bytes;
  const uint64_t words = bytes / 16;
  for (uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < words; i += uint64_t(gridDim.x) * blockDim.x)
    reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<const uint4*>(src)[i];
  for (uint64_t i = words * 16 + blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < bytes; i += uint64_t(gridDim.x) * blockDim.x)
    dst[i] = src[i];
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned done = atomicAdd(&v.arrivals[ch], 1u) + 1;
    if (done == gridDim.x) {                   // last block publishes the doorbell
      v.arrivals[ch] = 0;
      v.doorbells[ch].slot = slot;
      v.doorbells[ch].bytes = uint32_t(bytes);
      __threadfence_system();
      store_release_sys((uint64_t*)&v.doorbells[ch].seq, seq);
      v.seq[ch] = seq;
    }
  }
}

__device__ __forceinline__ const uint8_t* wait_peer(const DeviceView& v, int ch, uint64_t seq) {
  const int slot = int(seq % kSlots);
  if (threadIdx.x == 0) {
    const long long start = clock64();
    while (load_acquire_sys(v.recv_flags + (uint64_t(ch) * kSlots + slot) * 8) != seq)
      if (clock64() - start > (1LL << 36)) __trap();   // ~30 s: fail loudly, never hang silently
  }
  __syncthreads();
  return v.recv + (uint64_t(ch) * kSlots + slot) * v.max_bytes;
}

__global__ void finish_allreduce_bf16(DeviceView v, const __nv_bfloat16* x, __nv_bfloat16* out, uint64_t n, int ch) {
  const uint64_t seq = v.seq[ch];
  const __nv_bfloat16* peer = reinterpret_cast<const __nv_bfloat16*>(wait_peer(v, ch, seq));
  for (uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < n; i += uint64_t(gridDim.x) * blockDim.x)
    out[i] = __float2bfloat16(__bfloat162float(x[i]) + __bfloat162float(peer[i]));
}

__global__ void finish_allgather(DeviceView v, const uint8_t* x, uint8_t* out, uint64_t bytes, int ch, int rank) {
  const uint64_t seq = v.seq[ch];
  const uint8_t* peer = wait_peer(v, ch, seq);
  uint8_t* mine = out + uint64_t(rank) * bytes;
  uint8_t* theirs = out + uint64_t(1 - rank) * bytes;
  const uint64_t words = bytes / 16;
  for (uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < words; i += uint64_t(gridDim.x) * blockDim.x) {
    reinterpret_cast<uint4*>(mine)[i] = reinterpret_cast<const uint4*>(x)[i];
    reinterpret_cast<uint4*>(theirs)[i] = reinterpret_cast<const uint4*>(peer)[i];
  }
  for (uint64_t i = words * 16 + blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < bytes; i += uint64_t(gridDim.x) * blockDim.x) {
    mine[i] = x[i]; theirs[i] = peer[i];
  }
}

int blocks_for(uint64_t bytes) {
  const uint64_t b = (bytes + 16 * 256 - 1) / (16 * 256);
  return int(b < 1 ? 1 : b > 48 ? 48 : b);
}

void proxy_loop(Comm* c) {
  uint64_t last[kChannels] = {0, 0};
  ibv_wc wc[16];
  unsigned signaled_outstanding = 0;
  while (!c->stop.load(std::memory_order_relaxed)) {
    for (int ch = 0; ch < kChannels; ++ch) {
      Doorbell* d = &c->view.doorbells[ch];
      const uint64_t seq = __atomic_load_n(&d->seq, __ATOMIC_ACQUIRE);
      if (seq == last[ch]) continue;
      if (seq != last[ch] + 1) { c->error = 1; return; }
      const uint32_t slot = d->slot, bytes = d->bytes;
      const uint64_t off = (uint64_t(ch) * kSlots + slot) * c->view.max_bytes;
      ibv_sge sge{reinterpret_cast<uint64_t>(c->view.send + off), bytes, c->mr->lkey};
      ibv_send_wr data{}, flag{}, *bad = nullptr;
      data.sg_list = &sge; data.num_sge = 1; data.opcode = IBV_WR_RDMA_WRITE;
      data.wr.rdma.remote_addr = c->peer_recv + off; data.wr.rdma.rkey = c->peer_rkey;
      data.next = &flag;
      uint64_t value = seq;
      ibv_sge fsge{reinterpret_cast<uint64_t>(&value), 8, 0};
      flag.sg_list = &fsge; flag.num_sge = 1; flag.opcode = IBV_WR_RDMA_WRITE;
      flag.send_flags = IBV_SEND_INLINE;
      flag.wr.rdma.remote_addr = c->peer_flags + (uint64_t(ch) * kSlots + slot) * 64; flag.wr.rdma.rkey = c->peer_rkey;
      if (++c->posted % kSignalEvery == 0) { flag.send_flags |= IBV_SEND_SIGNALED; ++signaled_outstanding; }
      if (ibv_post_send(c->qp, &data, &bad)) { c->error = 2; return; }
      last[ch] = seq;
    }
    if (signaled_outstanding) {
      const int n = ibv_poll_cq(c->cq, 16, wc);
      for (int i = 0; i < n; ++i) if (wc[i].status != IBV_WC_SUCCESS) { c->error = 3; return; }
      if (n > 0) signaled_outstanding -= n;
    }
  }
}
}  // namespace

extern "C" {

// Returns a handle, or nullptr. `info` receives sizeof(Info) bytes to send to the peer.
void* fc_create(const char* device, int gid_index, uint64_t max_bytes, void* info, int* info_bytes) {
  auto* c = new Comm();
  int n = 0; ibv_device** list = ibv_get_device_list(&n);
  for (int i = 0; i < n; ++i) if (!strcmp(ibv_get_device_name(list[i]), device)) c->ctx = ibv_open_device(list[i]);
  ibv_free_device_list(list);
  if (!c->ctx) { delete c; return nullptr; }
  c->gid_index = gid_index;
  c->pd = ibv_alloc_pd(c->ctx);
  c->cq = ibv_create_cq(c->ctx, 1024, nullptr, nullptr, 0);
  ibv_qp_init_attr qia{};
  qia.send_cq = c->cq; qia.recv_cq = c->cq; qia.qp_type = IBV_QPT_RC;
  qia.cap.max_send_wr = 512; qia.cap.max_recv_wr = 1; qia.cap.max_send_sge = 1; qia.cap.max_recv_sge = 1;
  qia.cap.max_inline_data = 64;
  c->qp = ibv_create_qp(c->pd, &qia);
  if (!c->pd || !c->cq || !c->qp) return nullptr;
  const uint64_t ring = uint64_t(kChannels) * kSlots * max_bytes;
  const uint64_t flags = uint64_t(kChannels) * kSlots * 64;
  c->host_bytes = 2 * ring + flags + kChannels * sizeof(Doorbell);
  if (cudaHostAlloc(&c->host, c->host_bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) return nullptr;
  memset(c->host, 0, c->host_bytes);
  // No IBV_ACCESS_RELAXED_ORDERING: the flag WRITE must land after the data.
  c->mr = ibv_reg_mr(c->pd, c->host, c->host_bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  if (!c->mr) return nullptr;
  c->view.send = c->host; c->view.recv = c->host + ring;
  c->view.recv_flags = reinterpret_cast<uint64_t*>(c->host + 2 * ring);
  c->view.doorbells = reinterpret_cast<Doorbell*>(c->host + 2 * ring + flags);
  c->view.max_bytes = max_bytes;
  if (cudaMalloc(&c->view.seq, kChannels * sizeof(uint64_t)) != cudaSuccess) return nullptr;
  if (cudaMalloc(&c->view.arrivals, kChannels * sizeof(unsigned)) != cudaSuccess) return nullptr;
  cudaMemset(c->view.seq, 0, kChannels * sizeof(uint64_t));
  cudaMemset(c->view.arrivals, 0, kChannels * sizeof(unsigned));
  cudaDeviceSynchronize();
  ibv_qp_attr at{};
  at.qp_state = IBV_QPS_INIT; at.port_num = 1; at.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE;
  if (ibv_modify_qp(c->qp, &at, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) return nullptr;
  Info me{};
  me.qpn = c->qp->qp_num; me.psn = 4242; me.rkey = c->mr->rkey;
  me.recv = reinterpret_cast<uint64_t>(c->view.recv);
  me.flags = reinterpret_cast<uint64_t>(c->view.recv_flags);
  if (ibv_query_gid(c->ctx, 1, gid_index, &me.gid)) return nullptr;
  memcpy(info, &me, sizeof me); *info_bytes = int(sizeof me);
  return c;
}

int fc_connect(void* handle, const void* peer_info) {
  auto* c = static_cast<Comm*>(handle);
  Info peer; memcpy(&peer, peer_info, sizeof peer);
  ibv_qp_attr at{};
  at.qp_state = IBV_QPS_RTR; at.path_mtu = IBV_MTU_4096; at.dest_qp_num = peer.qpn; at.rq_psn = peer.psn;
  at.max_dest_rd_atomic = 1; at.min_rnr_timer = 12;
  at.ah_attr.is_global = 1; at.ah_attr.port_num = 1;
  at.ah_attr.grh.dgid = peer.gid; at.ah_attr.grh.sgid_index = c->gid_index; at.ah_attr.grh.hop_limit = 64;
  if (ibv_modify_qp(c->qp, &at, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                    IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) return errno ? errno : 1;
  at = ibv_qp_attr{};
  at.qp_state = IBV_QPS_RTS; at.timeout = 14; at.retry_cnt = 7; at.rnr_retry = 7; at.sq_psn = 4242; at.max_rd_atomic = 1;
  if (ibv_modify_qp(c->qp, &at, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                    IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) return errno ? errno : 1;
  c->peer_recv = peer.recv; c->peer_flags = peer.flags; c->peer_rkey = peer.rkey;
  c->proxy = std::thread(proxy_loop, c);
  return 0;
}

int fc_error(void* handle) { return static_cast<Comm*>(handle)->error.load(); }
uint64_t fc_max_bytes(void* handle) { return static_cast<Comm*>(handle)->view.max_bytes; }

int fc_allreduce_bf16(void* handle, const void* x, void* out, uint64_t n, int ch, void* stream) {
  auto* c = static_cast<Comm*>(handle);
  const uint64_t bytes = n * 2;
  if (ch < 0 || ch >= kChannels || bytes > c->view.max_bytes || bytes % 16) return 1;
  auto s = static_cast<cudaStream_t>(stream);
  stage_kernel<<<blocks_for(bytes), 256, 0, s>>>(c->view, static_cast<const uint8_t*>(x), bytes, ch);
  finish_allreduce_bf16<<<blocks_for(bytes), 256, 0, s>>>(c->view, static_cast<const __nv_bfloat16*>(x),
                                                            static_cast<__nv_bfloat16*>(out), n, ch);
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

// out holds 2*bytes: [rank 0 block][rank 1 block].
int fc_allgather(void* handle, const void* x, void* out, uint64_t bytes, int ch, int rank, void* stream) {
  auto* c = static_cast<Comm*>(handle);
  if (ch < 0 || ch >= kChannels || bytes > c->view.max_bytes || bytes % 16 || rank < 0 || rank > 1) return 1;
  auto s = static_cast<cudaStream_t>(stream);
  stage_kernel<<<blocks_for(bytes), 256, 0, s>>>(c->view, static_cast<const uint8_t*>(x), bytes, ch);
  finish_allgather<<<blocks_for(bytes), 256, 0, s>>>(c->view, static_cast<const uint8_t*>(x),
                                                       static_cast<uint8_t*>(out), bytes, ch, rank);
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

void fc_destroy(void* handle) {
  auto* c = static_cast<Comm*>(handle);
  c->stop = true;
  if (c->proxy.joinable()) c->proxy.join();
  // Buffers are intentionally leaked if the peer may still write into them.
}

}  // extern "C"
