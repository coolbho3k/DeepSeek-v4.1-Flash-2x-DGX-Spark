// SPDX-License-Identifier: AGPL-3.0-only
// DS41 lossless BF16 vocabulary-row staging. Architecture informed by the
// attributed MiaAI native Engram callback in miaai_row_store.cpp; no upstream
// source is copied here. Original upstream notices remain in vendor/.
// No CUDA calls, Python callbacks, table mappings, or callback-time allocation.
// Experimental: exact-byte direct-mapped row cache and parallel miss reads.
// Cached bytes come only from this store's own verified preads; the file
// identity check still runs before and after every lookup.
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
constexpr uint64_t rows = 129280, row_bytes = 10240, max_count = 256;
constexpr size_t alignment = 4096, buffer_bytes = 16384, max_threads = 16;
constexpr uint64_t cache_slots = 4096;  // 40 MiB per rank
constexpr uint32_t invalid_work = 1, changed_file = 2, io_failure = 4, poisoned = 8;
struct Store;
}

// Explicit stable ABI shared with the Python ctypes descriptor. The caller
// retains this descriptor, its pointers, and the Store until GPU work finishes.
struct VocabWork {
  Store* store;
  const int64_t* ids;
  uint8_t* output;
  uint32_t* status;
  uint64_t count;
};

namespace {
struct alignas(alignment) Buffer { uint8_t bytes[buffer_bytes]; };
struct Worker { Store* store = nullptr; size_t index = 0; pthread_t thread{}; };

struct Store {
  int fd = -1;
  uint64_t offset = 0, file_size = 0, lo = 0, hi = 0;
  int64_t mtime_ns = 0, ctime_ns = 0;
  size_t thread_count = 0, started = 0, remaining = 0;
  std::mutex caller, job_mutex;
  std::condition_variable ready, done;
  bool stop = false;
  uint64_t generation = 0;
  VocabWork* job = nullptr;
  // Rows the current job must read; owned by the caller under `caller`.
  std::array<uint32_t, max_count> order{};
  uint64_t order_count = 0;
  // Only the calling thread touches the cache, always under `caller`.
  std::unique_ptr<uint8_t[]> cache;
  std::array<int64_t, cache_slots> tags;
  uint64_t hits = 0, misses = 0;
  std::atomic<uint64_t> next{0}, reads{0}, bytes_read{0};
  std::atomic<uint32_t> failure{0};
  std::array<Buffer, max_threads + 1> buffers;
  std::array<Worker, max_threads> workers;

  ~Store() {
    {
      std::lock_guard<std::mutex> lock(job_mutex);
      stop = true;
    }
    ready.notify_all();
    for (size_t i = 0; i < started; ++i) pthread_join(workers[i].thread, nullptr);
    if (fd >= 0) ::close(fd);
  }

  bool unchanged() const noexcept {
    struct stat info{};
    return fstat(fd, &info) == 0 && S_ISREG(info.st_mode)
        && uint64_t(info.st_size) == file_size
        && int64_t(info.st_mtim.tv_sec) * 1000000000 + info.st_mtim.tv_nsec == mtime_ns
        && int64_t(info.st_ctim.tv_sec) * 1000000000 + info.st_ctim.tv_nsec == ctime_ns;
  }

  void row(VocabWork* work, uint64_t i, Buffer& scratch) noexcept {
    const int64_t id = work->ids[i];
    if (id < 0 || uint64_t(id) < lo || uint64_t(id) >= hi) return;
    if (failure.load(std::memory_order_relaxed)) return;
    const uint64_t position = offset + uint64_t(id) * row_bytes;
    const uint64_t aligned = position & ~(uint64_t(alignment) - 1);
    const size_t inside = position - aligned;
    const size_t wanted = ((inside + row_bytes + alignment - 1) / alignment) * alignment;
    ssize_t count;
    do { count = pread(fd, scratch.bytes, wanted, aligned); }
    while (count < 0 && errno == EINTR);
    if (count < 0 || size_t(count) < inside + row_bytes) {
      failure.fetch_or(io_failure, std::memory_order_relaxed);
      return;
    }
    std::memcpy(work->output + i*row_bytes, scratch.bytes + inside, row_bytes);
    reads.fetch_add(1, std::memory_order_relaxed);
    bytes_read.fetch_add(count, std::memory_order_relaxed);
  }

  void drain(VocabWork* work, Buffer& scratch) noexcept {
    for (;;) {
      const uint64_t i = next.fetch_add(1, std::memory_order_relaxed);
      if (i >= order_count) return;
      row(work, order[i], scratch);
    }
  }

  static void* loop(void* opaque) noexcept {
    auto& worker = *static_cast<Worker*>(opaque);
    auto& self = *worker.store;
    uint64_t seen = 0;
    for (;;) {
      std::unique_lock<std::mutex> lock(self.job_mutex);
      // Join only a job that is still open; the caller closes it (job=null)
      // under this mutex before waiting, so no worker joins a finished job.
      self.ready.wait(lock, [&] {
        return self.stop || (self.job && self.generation != seen); });
      if (self.stop) return nullptr;
      seen = self.generation;
      auto* work = self.job;
      ++self.remaining;
      lock.unlock();
      self.drain(work, self.buffers[worker.index]);
      lock.lock();
      if (--self.remaining == 0) self.done.notify_one();
    }
  }

  bool launch(size_t count) {
    thread_count = count;
    pthread_attr_t attr;
    if (pthread_attr_init(&attr)) return false;
    int error = pthread_attr_setstacksize(&attr, 256 * 1024);
    if (!error) {
      for (size_t i = 0; i < count; ++i) {
        workers[i].store = this;
        workers[i].index = i;
        error = pthread_create(&workers[i].thread, &attr, loop, &workers[i]);
        if (error) break;
        ++started;
      }
    }
    pthread_attr_destroy(&attr);
    return !error;
  }

  void run(VocabWork* work) {
    std::lock_guard<std::mutex> serialize(caller);
    *work->status = 0;
    std::memset(work->output, 0, work->count * row_bytes);
    if (failure.load(std::memory_order_relaxed)) {
      *work->status = poisoned;
      return;
    }
    if (!unchanged()) {
      failure.store(changed_file, std::memory_order_relaxed);
      *work->status = changed_file;
      return;
    }
    // Serve owned cached rows; queue every other owned row for a direct read.
    order_count = 0;
    for (uint64_t i = 0; i < work->count; ++i) {
      const int64_t id = work->ids[i];
      if (id < 0 || uint64_t(id) < lo || uint64_t(id) >= hi) continue;
      const uint64_t slot = uint64_t(id) % cache_slots;
      if (tags[slot] == id) {
        std::memcpy(work->output + i*row_bytes, cache.get() + slot*row_bytes, row_bytes);
        ++hits;
      } else {
        order[order_count++] = uint32_t(i);
        ++misses;
      }
    }
    if (order_count < 2 || thread_count == 0) {
      for (uint64_t i = 0; i < order_count; ++i) row(work, order[i], buffers[max_threads]);
    } else {
      {
        std::lock_guard<std::mutex> lock(job_mutex);
        job = work;
        next.store(0, std::memory_order_relaxed);
        ++generation;
      }
      // Wake only as many helpers as there are additional rows to read.
      const size_t helpers = std::min<size_t>(thread_count, order_count - 1);
      for (size_t i = 0; i < helpers; ++i) ready.notify_one();
      drain(work, buffers[max_threads]);
      std::unique_lock<std::mutex> lock(job_mutex);
      job = nullptr;
      done.wait(lock, [&] { return remaining == 0; });
    }
    if (!unchanged()) failure.fetch_or(changed_file, std::memory_order_relaxed);
    *work->status = failure.load(std::memory_order_relaxed);
    // Retain only rows from a fully successful, identity-checked lookup.
    if (*work->status) return;
    for (uint64_t i = 0; i < order_count; ++i) {
      const uint64_t index = order[i];
      const uint64_t slot = uint64_t(work->ids[index]) % cache_slots;
      std::memcpy(cache.get() + slot*row_bytes, work->output + index*row_bytes, row_bytes);
      tags[slot] = work->ids[index];
    }
  }
};
}

extern "C" uint64_t ds41_vocab_row_abi() noexcept { return 1; }

extern "C" Store* ds41_vocab_row_open(const char* path, uint64_t offset,
    uint64_t file_size, int64_t mtime_ns, int64_t ctime_ns,
    uint64_t rank, uint64_t threads) noexcept {
  if (!path || rank > 1 || threads > max_threads || offset > file_size
      || file_size - offset < rows * row_bytes) return nullptr;
  try {
    auto result = std::make_unique<Store>();
    result->offset = offset;
    result->file_size = file_size;
    result->mtime_ns = mtime_ns;
    result->ctime_ns = ctime_ns;
    result->lo = rank * (rows/2);
    result->hi = (rank + 1) * (rows/2);
    result->cache.reset(new uint8_t[cache_slots * row_bytes]);
    result->tags.fill(-1);
    result->fd = open(path, O_RDONLY | O_DIRECT | O_CLOEXEC | O_NOFOLLOW);
    if (result->fd < 0 || !result->unchanged() || !result->launch(threads)) return nullptr;
    return result.release();
  } catch (...) { return nullptr; }
}

extern "C" void ds41_vocab_row_lookup(void* opaque) noexcept {
  auto* work = static_cast<VocabWork*>(opaque);
  if (!work || !work->status) return;
  if (!work->store || work->count > max_count || !work->ids || !work->output) {
    *work->status = invalid_work;
    return;
  }
  try { work->store->run(work); }
  catch (...) {
    work->store->failure.fetch_or(poisoned, std::memory_order_relaxed);
    *work->status = poisoned;
  }
}

extern "C" void ds41_vocab_row_stats(Store* store, uint64_t* output) noexcept {
  if (!store || !output) return;
  output[0] = store->reads.load(std::memory_order_relaxed);
  output[1] = store->bytes_read.load(std::memory_order_relaxed);
  output[2] = sizeof(Store);
  output[3] = store->thread_count;
  output[4] = store->failure.load(std::memory_order_relaxed);
}

extern "C" void ds41_vocab_row_cache_stats(Store* store, uint64_t* output) noexcept {
  if (!store || !output) return;
  std::lock_guard<std::mutex> serialize(store->caller);
  output[0] = store->hits;
  output[1] = store->misses;
  output[2] = cache_slots * row_bytes;
}

// The caller must first fence every callback and destroy every owning CUDA
// graph. The Python stage refuses close while a graph still retains the store.
extern "C" void ds41_vocab_row_close(Store* store) noexcept { delete store; }
