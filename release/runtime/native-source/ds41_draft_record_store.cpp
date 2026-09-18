// SPDX-License-Identifier: AGPL-3.0-only
// DS41 lossless FP4 draft staging. Pool/lifetime architecture adapted from
// ds41_vocab_row_store.cpp and the attributed MiaAI native Engram integration.
// No CUDA calls, Python callbacks, mappings or callback-time allocations.
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
constexpr uint64_t record_bytes = 9400320, experts = 128, layers = 3, slots = 9;
constexpr uint64_t bank_bytes = layers * experts * record_bytes;
constexpr size_t alignment = 4096, max_threads = 4;
constexpr uint32_t invalid_work = 1, changed_file = 2, io_failure = 4, poisoned = 8;
struct Store;
}

struct DraftWork {
  Store* store;
  const int64_t* ids;
  uint8_t* records;
  int32_t* mapped_ids;
  uint32_t* status;
  uint64_t layer;
  uint64_t count;
};

namespace {
struct Worker { Store* store = nullptr; pthread_t thread{}; };
struct Store {
  int fd = -1;
  uint64_t device = 0, inode = 0, rank = 0;
  int64_t mtime_ns = 0, ctime_ns = 0;
  size_t thread_count = 0, started = 0, remaining = 0;
  std::mutex caller, job_mutex;
  std::condition_variable ready, done;
  bool stop = false;
  uint64_t generation = 0, unique_count = 0;
  DraftWork* job = nullptr;
  std::array<int64_t, slots> unique{};
  std::array<Worker, max_threads> workers;
  std::atomic<uint64_t> next{0}, calls{0}, reads{0}, bytes_read{0}, last_unique{0};
  std::atomic<uint32_t> failure{0};

  ~Store() {
    { std::lock_guard<std::mutex> lock(job_mutex); stop = true; }
    ready.notify_all();
    for (size_t i = 0; i < started; ++i) pthread_join(workers[i].thread, nullptr);
    if (fd >= 0) ::close(fd);
  }

  bool unchanged() const noexcept {
    struct stat info{};
    return fstat(fd, &info) == 0 && S_ISREG(info.st_mode)
        && uint64_t(info.st_size) == bank_bytes
        && uint64_t(info.st_dev) == device && uint64_t(info.st_ino) == inode
        && int64_t(info.st_mtim.tv_sec)*1000000000 + info.st_mtim.tv_nsec == mtime_ns
        && int64_t(info.st_ctim.tv_sec)*1000000000 + info.st_ctim.tv_nsec == ctime_ns;
  }

  void record(DraftWork* work, uint64_t slot) noexcept {
    if (failure.load(std::memory_order_relaxed)) return;
    const uint64_t offset = (work->layer * experts + uint64_t(unique[slot])) * record_bytes;
    ssize_t count;
    do { count = pread(fd, work->records + slot*record_bytes, record_bytes, offset); }
    while (count < 0 && errno == EINTR);
    if (count < 0 || uint64_t(count) != record_bytes) {
      failure.fetch_or(io_failure, std::memory_order_relaxed);
      return;
    }
    reads.fetch_add(1, std::memory_order_relaxed);
    bytes_read.fetch_add(record_bytes, std::memory_order_relaxed);
  }

  void drain(DraftWork* work) noexcept {
    for (;;) {
      const uint64_t slot = next.fetch_add(1, std::memory_order_relaxed);
      if (slot >= unique_count) return;
      record(work, slot);
    }
  }

  static void* loop(void* opaque) noexcept {
    auto& self = *static_cast<Worker*>(opaque)->store;
    uint64_t seen = 0;
    for (;;) {
      std::unique_lock<std::mutex> lock(self.job_mutex);
      self.ready.wait(lock, [&] { return self.stop || self.generation != seen; });
      if (self.stop) return nullptr;
      seen = self.generation;
      auto* work = self.job;
      lock.unlock();
      self.drain(work);
      lock.lock();
      if (--self.remaining == 0) self.done.notify_one();
    }
  }

  bool launch(size_t count) {
    thread_count = count;
    pthread_attr_t attr;
    if (pthread_attr_init(&attr)) return false;
    int error = pthread_attr_setstacksize(&attr, 256*1024);
    if (!error) {
      for (size_t i = 0; i < count; ++i) {
        workers[i].store = this;
        error = pthread_create(&workers[i].thread, &attr, loop, &workers[i]);
        if (error) break;
        ++started;
      }
    }
    pthread_attr_destroy(&attr);
    return !error;
  }

  void run(DraftWork* work) {
    std::lock_guard<std::mutex> serialize(caller);
    *work->status = 0;
    calls.fetch_add(1, std::memory_order_relaxed);
    std::fill_n(work->mapped_ids, work->count, int32_t(-1));
    if (failure.load(std::memory_order_relaxed)) {
      *work->status = poisoned;
      return;
    }
    if (!unchanged()) {
      failure.store(changed_file, std::memory_order_relaxed);
      *work->status = changed_file;
      return;
    }
    unique_count = 0;
    // Validate the complete route set before publishing any mapped ID or I/O.
    for (uint64_t i = 0; i < work->count; ++i) {
      if (work->ids[i] < -1 || work->ids[i] >= int64_t(experts)) {
        failure.store(invalid_work, std::memory_order_relaxed);
        *work->status = invalid_work;
        return;
      }
    }
    for (uint64_t i = 0; i < work->count; ++i) {
      const int64_t id = work->ids[i];
      if (id == -1) continue;
      uint64_t slot = 0;
      while (slot < unique_count && unique[slot] != id) ++slot;
      if (slot == unique_count) unique[unique_count++] = id;
      work->mapped_ids[i] = int32_t(slot);
    }
    last_unique.store(unique_count, std::memory_order_relaxed);
    // Deterministic inactive slots; active slots are completely overwritten.
    std::memset(work->records + unique_count*record_bytes, 0, (slots-unique_count)*record_bytes);
    if (thread_count == 0 || unique_count < 2) {
      for (uint64_t slot = 0; slot < unique_count; ++slot) record(work, slot);
    } else {
      {
        std::lock_guard<std::mutex> lock(job_mutex);
        job = work;
        next.store(0, std::memory_order_relaxed);
        remaining = thread_count;
        ++generation;
      }
      ready.notify_all();
      drain(work);
      std::unique_lock<std::mutex> lock(job_mutex);
      done.wait(lock, [&] { return remaining == 0; });
      job = nullptr;
    }
    if (!unchanged()) failure.fetch_or(changed_file, std::memory_order_relaxed);
    *work->status = failure.load(std::memory_order_relaxed);
  }
};
}

extern "C" uint64_t ds41_draft_record_abi() noexcept { return 1; }

extern "C" Store* ds41_draft_record_open(const char* path, uint64_t device,
    uint64_t inode, uint64_t file_size, int64_t mtime_ns, int64_t ctime_ns,
    uint64_t rank, uint64_t threads) noexcept {
  if (!path || file_size != bank_bytes || rank > 1 || threads > max_threads) return nullptr;
  try {
    auto result = std::make_unique<Store>();
    result->device = device; result->inode = inode; result->rank = rank;
    result->mtime_ns = mtime_ns; result->ctime_ns = ctime_ns;
    result->fd = open(path, O_RDONLY | O_DIRECT | O_CLOEXEC | O_NOFOLLOW);
    if (result->fd < 0 || !result->unchanged() || !result->launch(threads)) return nullptr;
    return result.release();
  } catch (...) { return nullptr; }
}

extern "C" void ds41_draft_record_lookup(void* opaque) noexcept {
  auto* work = static_cast<DraftWork*>(opaque);
  if (!work || !work->status) return;
  if (!work->store || !work->ids || !work->records || !work->mapped_ids
      || work->layer >= layers || work->count > slots
      || reinterpret_cast<uintptr_t>(work->records) % alignment) {
    *work->status = invalid_work;
    return;
  }
  try { work->store->run(work); }
  catch (...) {
    work->store->failure.fetch_or(poisoned, std::memory_order_relaxed);
    *work->status = poisoned;
  }
}

extern "C" void ds41_draft_record_stats(Store* store, uint64_t* output) noexcept {
  if (!store || !output) return;
  output[0] = store->calls.load(std::memory_order_relaxed);
  output[1] = store->reads.load(std::memory_order_relaxed);
  output[2] = store->bytes_read.load(std::memory_order_relaxed);
  output[3] = store->thread_count;
  output[4] = store->failure.load(std::memory_order_relaxed);
  output[5] = sizeof(Store);
  output[6] = store->last_unique.load(std::memory_order_relaxed);
  output[7] = store->rank;
}

// Caller must fence every outstanding callback and destroy all owning graphs.
extern "C" void ds41_draft_record_close(Store* store) noexcept { delete store; }
