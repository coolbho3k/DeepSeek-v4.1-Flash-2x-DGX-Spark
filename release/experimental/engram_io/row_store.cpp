// SPDX-License-Identifier: AGPL-3.0-only
// Uses MiaAI-Lab's row_store.cpp at 979e68a62c90b24d928f5638596e0ceed90e9f34.
// Copyright/attribution: Mia's AI Lab and upstream contributors.
// See ../vendor/miaai-dsv41-agpl/LICENSE and LICENSE.MIT.
// Local adaptation: bounded-memory entry points and serialized store lifetime.
// CUDA-graph host callback semantics and original row bytes are unchanged.

#include <chrono>
#include "row_store_core.cpp"

namespace {
std::mutex ds41_store_lifetime;

void ds41_validate_mode() {
  const char *mode = std::getenv("OFFLOAD_MODE");
  const char *scales = std::getenv("DSV41_RESIDENT_SCALES");
  const char *threads = std::getenv("DSV41_IO_THREADS");
  if ((mode && std::strcmp(mode, "ssd")) || !scales || std::strcmp(scales, "0"))
    fail("DS41 requires SSD mode with resident scales disabled");
  if (!threads || !*threads) fail("DS41 requires an explicit bounded IO thread count");
  for (const char *p = threads; *p; ++p)
    if (*p < '0' || *p > '9') fail("invalid DS41 IO thread count");
  const uint64_t count = env_u64("DSV41_IO_THREADS", 0);
  if (count < 1 || count > 96 || std::strlen(threads) > 2)
    fail("DS41 IO thread count must be in [1, 96]");
}
}

extern "C" Store *ds41_row_store_open(const char *path, uint64_t rows,
    uint64_t weight_offset, uint64_t scale_offset, uint64_t budget) {
  std::lock_guard<std::mutex> guard(ds41_store_lifetime);
  ds41_validate_mode();
  if (budget > (uint64_t(64) << 20)) fail("DS41 row cache exceeds 64 MiB per table");
  Store *s = row_store_open(path, rows, weight_offset, scale_offset, budget);
  if (!s) return nullptr;
  if (!(fcntl(s->fd, F_GETFL) & O_DIRECT) || s->resident || s->scales)
    fail("DS41 requires actual O_DIRECT without resident table/scale mappings");
  if (s->slots * (kRowBytes + sizeof(uint64_t)) + s->sets > budget)
    fail("DS41 row cache exceeded its budget");
  return s;
}

extern "C" void ds41_row_store_range(Store *s, uint64_t lo, uint64_t hi) {
  std::lock_guard<std::mutex> guard(ds41_store_lifetime);
  ds41_validate_mode();
  row_store_range(s, lo, hi);
  if (s->scales || s->resident) fail("DS41 unexpected resident allocation");
}

extern "C" void ds41_row_store_lookup(void *opaque) {
  std::lock_guard<std::mutex> guard(ds41_store_lifetime);
  ds41_validate_mode();
  auto *work = static_cast<Work *>(opaque);
  if (!work || !work->store || work->count > 1056 * 144)
    fail("DS41 row lookup exceeds bounded staging");
  const auto started = std::chrono::steady_clock::now();
  row_store_lookup(work);
  const auto elapsed = std::chrono::steady_clock::now() - started;
  work->store->lookup_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
  work->store->lookups.fetch_add(1);
}

extern "C" void ds41_row_store_close(Store *s) {
  std::lock_guard<std::mutex> guard(ds41_store_lifetime);
  if (s) row_store_close(s);
}

extern "C" uint64_t ds41_row_store_abi() { return 2; }

// This is a DS41 extension to MiaAI's packed reader, not an upstream claim.
extern "C" int ds41_row_store_attach_packed(Store *s, const char *path, uint64_t layer) {
  std::lock_guard<std::mutex> guard(ds41_store_lifetime);
  ds41_validate_mode();
  return row_store_attach_packed(s, path, layer);
}
extern "C" void ds41_row_store_profile(Store *s, uint64_t *out) {
  out[0] = s->io_bytes.load(); out[1] = s->lookup_ns.load();
  out[2] = s->lookups.load(); out[3] = s->packed_pages ? 2 : (s->packed_fd >= 0 ? 1 : 0);
}
// Component tests only; serialized against every lookup/store transition.
extern "C" void ds41_row_store_clear_cache(Store *s) {
  std::lock_guard<std::mutex> guard(ds41_store_lifetime);
  if (s->slots) std::memset(s->keys, 0, s->slots * sizeof(uint64_t));
  if (s->sets) std::memset(s->victim, 0, s->sets);
  s->hits.store(0); s->misses.store(0); s->reads.store(0);
  s->io_bytes.store(0); s->lookup_ns.store(0); s->lookups.store(0);
}
