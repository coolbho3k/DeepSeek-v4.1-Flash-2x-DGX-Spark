// SPDX-License-Identifier: AGPL-3.0-only
// Experimental derivatives of MiaAI Lab / Wesley Young's native Engram cache.
// Baseline: 979e68a62c90b24d928f5638596e0ceed90e9f34, with existing DS41 changes.
// Full attribution and original licenses: see README.md and ../../runtime/vendor/.
// CPU-only synthetic benchmark. Never link this file into the serving runtime.

#include "../../runtime/serving/row_store_core.cpp"
#include <arm_neon.h>
#include <arm_sve.h>
#include <asm/hwcap.h>
#include <array>
#include <chrono>
#include <future>
#include <numeric>
#include <random>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <sys/auxv.h>
#include <sys/resource.h>

using Clock = std::chrono::steady_clock;
using Kernel = void (*)(Work *);

// Compile the exact included baseline with generic AArch64 flags, not SVE2.
__attribute__((noinline)) static void baseline(Work *w) {
  serve(w, 0, w->count);
}

#pragma GCC push_options
#pragma GCC target("arch=armv8-a+sve2")

enum Search { Scalar, Neon, Sve };

template<Search search>
__attribute__((always_inline)) inline uint64_t find_key(
    const uint64_t *keys, uint64_t ways, uint64_t key) {
  if constexpr (search == Scalar) {
    for (uint64_t i = 0; i < ways; ++i) if (keys[i] == key) return i;
  } else if constexpr (search == Neon) {
    uint64_t i = 0;
    const auto wanted = vdupq_n_u64(key);
    for (; i + 2 <= ways; i += 2) {
      const auto eq = vceqq_u64(vld1q_u64(keys + i), wanted);
      if (vgetq_lane_u64(eq, 0)) return i;
      if (vgetq_lane_u64(eq, 1)) return i + 1;
    }
    if (i < ways && keys[i] == key) return i;
  } else {
    for (uint64_t i = 0; i < ways; i += svcntd()) {
      const auto pg = svwhilelt_b64(i, ways);
      const auto eq = svcmpeq_n_u64(pg, svld1_u64(pg, keys + i), key);
      if (svptest_any(pg, eq)) return i + svcntp_b64(pg, svbrkb_z(pg, eq));
    }
  }
  return ways;
}

inline bool owned(Work *w, uint64_t i) {
  const int64_t id = w->ids[i];
  if (id >= 0 && uint64_t(id) >= w->store->rows) fail("benchmark invalid row ID");
  if (id >= 0 && uint64_t(id) >= w->store->row_lo &&
      uint64_t(id) < w->store->row_hi) return true;
  std::memset(w->weights + i * kWeightBytes, 0, kWeightBytes);
  std::memset(w->scales + i * kScaleBytes, 0, kScaleBytes);
  return false;
}

inline void copy_hit(Work *w, uint64_t i, uint64_t slot) {
  const uint8_t *row = w->store->cache + slot * kRowBytes;
  std::memcpy(w->weights + i * kWeightBytes, row, kWeightBytes);
  std::memcpy(w->scales + i * kScaleBytes, row + kWeightBytes, kScaleBytes);
}

template<Search search, bool batch_stats = false>
__attribute__((noinline)) void direct(Work *w) {
  Store *s = w->store;
  uint64_t hits = 0;
  for (uint64_t i = 0; i < w->count; ++i) {
    if (!owned(w, i)) continue;
    bool found = false;
    if (s->sets) {
      const uint64_t set = uint64_t(w->ids[i]) % s->sets;
      std::lock_guard<std::mutex> guard(s->locks[set % 256]);
      const uint64_t base = set * s->ways;
      const uint64_t way = find_key<search>(s->keys + base, s->ways, w->ids[i] + 1);
      if (way < s->ways) {
        copy_hit(w, i, base + way);
        if constexpr (batch_stats) ++hits;
        else s->hits.fetch_add(1, std::memory_order_relaxed);
        found = true;
      }
    }
    // Unchanged fallback outside the cache lock: exact immutable row bytes.
    if (!found) serve(w, i, i + 1);
  }
  if constexpr (batch_stats) s->hits.fetch_add(hits, std::memory_order_relaxed);
}

struct Scratch {
  std::vector<uint32_t> indices, misses;
  std::vector<uint64_t> valid;
};
static thread_local Scratch scratch;

template<Search search>
__attribute__((noinline)) void grouped(Work *w) {
  Store *s = w->store;
  if (!s->sets) { serve(w, 0, w->count); return; }
  if (w->count > 32768) fail("benchmark batch cap");
  scratch.indices.resize(w->count);
  scratch.valid.resize(w->count);
  scratch.misses.clear();
  scratch.misses.reserve(w->count);
  std::array<uint32_t, 257> offsets{};
  // SVE variant also batches signed-ID/ownership predicates. Hash division and
  // bucket counts stay scalar; SVE has no integer-vector division instruction.
  if constexpr (search == Sve) {
    for (uint64_t i = 0; i < w->count; i += svcntd()) {
      const auto pg = svwhilelt_b64(i, w->count);
      const auto ids = svld1_s64(pg, w->ids + i);
      if (svptest_any(pg, svcmpge_n_s64(pg, ids, int64_t(s->rows))))
        fail("benchmark invalid row ID");
      auto valid = svcmpge_n_s64(pg, ids, int64_t(s->row_lo));
      valid = svand_b_z(pg, valid, svcmplt_n_s64(pg, ids, int64_t(s->row_hi)));
      svst1_u64(pg, scratch.valid.data() + i,
                svsel_u64(valid, svdup_n_u64(1), svdup_n_u64(0)));
    }
  }
  for (uint64_t i = 0; i < w->count; ++i) {
    bool valid;
    if constexpr (search == Sve) {
      valid = scratch.valid[i];
      if (!valid) {
        std::memset(w->weights + i * kWeightBytes, 0, kWeightBytes);
        std::memset(w->scales + i * kScaleBytes, 0, kScaleBytes);
      }
    } else {
      valid = owned(w, i);
      scratch.valid[i] = valid;
    }
    if (valid) ++offsets[(uint64_t(w->ids[i]) % s->sets) % 256 + 1];
  }
  for (size_t i = 1; i < offsets.size(); ++i) offsets[i] += offsets[i - 1];
  auto next = offsets;
  for (uint64_t i = 0; i < w->count; ++i)
    if (scratch.valid[i])
      scratch.indices[next[(uint64_t(w->ids[i]) % s->sets) % 256]++] = i;
  uint64_t hits = 0;
  for (uint64_t stripe = 0; stripe < 256; ++stripe) {
    if (offsets[stripe] == offsets[stripe + 1]) continue;
    std::lock_guard<std::mutex> guard(s->locks[stripe]);
    for (uint32_t j = offsets[stripe]; j < offsets[stripe + 1]; ++j) {
      const uint64_t i = scratch.indices[j];
      const uint64_t base = (uint64_t(w->ids[i]) % s->sets) * s->ways;
      const uint64_t way = find_key<search>(s->keys + base, s->ways, w->ids[i] + 1);
      if (way < s->ways) { copy_hit(w, i, base + way); ++hits; }
      else scratch.misses.push_back(i);
    }
  }
  s->hits.fetch_add(hits, std::memory_order_relaxed);
  // A miss may now hit after another fill: baseline rechecks under its lock.
  for (uint32_t i : scratch.misses) serve(w, i, i + 1);
}

static unsigned vector_bytes() { return svcntb(); }
#pragma GCC pop_options

struct Variant { const char *name; Kernel call; };
const std::array<Variant, 7> variants{{
  {"baseline", baseline},
  {"direct_scalar", direct<Scalar>},
  {"direct_neon", direct<Neon>},
  {"direct_sve", direct<Sve>},
  {"direct_scalar_stats", direct<Scalar, true>},
  {"grouped_scalar", grouped<Scalar>},
  {"grouped_sve", grouped<Sve>},
}};

static uint8_t value(uint64_t id, uint64_t byte) {
  return uint8_t((id * 1315423911ULL + byte * 2654435761ULL + (id >> 9)) >> ((byte % 7) * 8));
}

struct Fixture {
  Store s{};
  std::vector<uint8_t> bytes, victim, backing;
  std::vector<uint64_t> keys;
  Fixture(uint64_t sets, uint64_t ways, bool with_backing = false) {
    s.fd = -1; s.sets = sets; s.ways = ways; s.slots = sets * ways;
    s.row_lo = sets * 32; s.row_hi = sets * 64; s.rows = s.row_hi + sets;
    bytes.resize(s.slots * kRowBytes); keys.resize(s.slots); victim.resize(sets);
    s.cache = bytes.data(); s.keys = keys.data(); s.victim = victim.data();
    if (with_backing) {
      backing.resize(s.rows * kRowBytes);
      s.weight_offset = 0; s.scale_offset = s.rows * kWeightBytes;
      s.resident = backing.data();
      for (uint64_t id = 0; id < s.rows; ++id) {
        for (uint64_t j = 0; j < kWeightBytes; ++j)
          backing[id * kWeightBytes + j] = value(id, j);
        for (uint64_t j = 0; j < kScaleBytes; ++j)
          backing[s.scale_offset + id * kScaleBytes + j] = value(id, kWeightBytes + j);
      }
    }
    reset();
  }
  uint64_t id(uint64_t slot) const {
    return s.row_lo + slot / s.ways + (slot % s.ways) * s.sets;
  }
  void reset(bool cold = false) {
    for (uint64_t slot = 0; slot < s.slots; ++slot) {
      keys[slot] = cold ? 0 : id(slot) + 1;
      for (uint64_t j = 0; j < kRowBytes; ++j) bytes[slot * kRowBytes + j] = value(id(slot), j);
    }
    std::fill(victim.begin(), victim.end(), 0);
    s.hits = 0; s.misses = 0; s.reads = 0;
  }
};

struct Output {
  static constexpr size_t guard = 32;
  std::vector<uint8_t> w, s;
  Work work;
  Output(Store *store, const std::vector<int64_t> &ids)
      : w(ids.size() * kWeightBytes + guard * 2, 0xa5),
        s(ids.size() * kScaleBytes + guard * 2, 0xa5),
        work{store, ids.data(), w.data() + guard, s.data() + guard, ids.size()} {}
  void check() const {
    for (size_t j = 0; j < guard; ++j)
      if (w[j] != 0xa5 || w[w.size()-1-j] != 0xa5 || s[j] != 0xa5 || s[s.size()-1-j] != 0xa5)
        throw std::runtime_error("output guard overwritten");
    for (uint64_t i = 0; i < work.count; ++i) {
      const int64_t id = work.ids[i];
      const bool valid = id >= 0 && uint64_t(id) >= work.store->row_lo && uint64_t(id) < work.store->row_hi;
      for (uint64_t j = 0; j < kRowBytes; ++j) {
        const uint8_t actual = j < kWeightBytes ? work.weights[i*kWeightBytes+j] : work.scales[i*kScaleBytes+j-kWeightBytes];
        if (actual != (valid ? value(id, j) : 0))
          throw std::runtime_error("row byte mismatch at row " + std::to_string(i));
      }
    }
  }
};

static void correctness() {
  size_t cases = 0;
  std::mt19937_64 rng(193);
  for (uint64_t ways : {1, 2, 3, 4, 8, 16}) {
    Fixture f(17, ways, true);
    for (size_t n : {0, 1, 2, 3, 7, 8, 12, 31, 48, 144, 288, 577, 3072, 3456}) {
      std::vector<int64_t> ids(n);
      for (size_t i = 0; i < n; ++i) {
        // Mix resident hits, replacement collisions, both ownership boundaries,
        // negative image/dead markers and duplicates. More keys than slots.
        switch (i % 9) {
          case 0: ids[i] = -1; break;
          case 1: ids[i] = f.s.row_lo - 1; break;
          case 2: ids[i] = f.s.row_hi; break;
          case 3: ids[i] = f.s.row_lo; break;
          case 4: ids[i] = f.s.row_hi - 1; break;
          default: ids[i] = f.s.row_lo + rng() % (f.s.row_hi - f.s.row_lo);
        }
      }
      for (const auto &v : variants) for (bool cold : {false, true}) {
        f.reset(cold);
        Output out(&f.s, ids);
        for (int repeat = 0; repeat < 2; ++repeat) { v.call(&out.work); out.check(); }
        size_t expected = 0;
        for (auto id : ids) expected += id >= int64_t(f.s.row_lo) && id < int64_t(f.s.row_hi);
        if (f.s.hits + f.s.misses != expected * 2) throw std::runtime_error("statistics mismatch");
        ++cases;
      }
    }
    std::vector<int64_t> ids(577);
    for (auto &id : ids) id = f.s.row_lo + rng() % (f.s.row_hi - f.s.row_lo);
    for (const auto &v : variants) {
      f.reset(true);
      auto job = [&] {
        Output out(&f.s, ids);
        for (int j = 0; j < 4; ++j) { v.call(&out.work); out.check(); }
      };
      auto one = std::async(std::launch::async, job);
      auto two = std::async(std::launch::async, job);
      one.get(); two.get(); ++cases;
      if (f.s.hits + f.s.misses != ids.size() * 8)
        throw std::runtime_error("concurrent statistics mismatch");
    }
  }
  // Disabled cache must use the original fallback, not divide by zero.
  Fixture empty(17, 4, true);
  empty.s.sets = 0;
  std::vector<int64_t> ids{-1, int64_t(empty.s.row_lo), int64_t(empty.s.row_hi-1)};
  for (const auto &v : variants) { Output out(&empty.s, ids); v.call(&out.work); out.check(); ++cases; }
  std::printf("{\"kind\":\"correctness\",\"cases\":%zu,\"exact_bytes\":true}\n", cases);
  std::fflush(stdout);
}

static double quantile(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end()); return v[size_t(q * (v.size() - 1))];
}

static void benchmark(int repeats, int cache_mib) {
  const uint64_t sets = (uint64_t(cache_mib) << 20) / (4 * (kRowBytes + 8) + 1);
  Fixture f(sets, 4);
  std::mt19937_64 rng(41325);
  // Current TP2 stage has 12 local heads and chunks at most 256 tokens.
  // 12/48/288 represent 1/4/24 token rows, 3072 the maximum prefill chunk.
  // Larger 3456/24576 batches are intentionally labeled stress-only in docs.
  for (size_t count : {12, 48, 144, 288, 576, 3072, 3456, 24576}) {
    for (const std::string pattern : {"hot", "rotating", "masked"}) {
      constexpr size_t banks_count = 8;
      std::vector<std::vector<int64_t>> banks(banks_count, std::vector<int64_t>(count));
      for (size_t b = 0; b < banks_count; ++b) {
        for (size_t i = 0; i < count; ++i) {
          const uint64_t slot = pattern == "hot" ? rng() % std::min<uint64_t>(f.s.slots, 512) : rng() % f.s.slots;
          banks[b][i] = f.id(slot);
          if (pattern == "masked" && i % 4 == 0) banks[b][i] = -1;
          if (pattern == "masked" && i % 4 == 1) banks[b][i] = f.s.row_hi;
        }
      }
      std::vector<Output> outputs;
      outputs.reserve(banks_count);
      for (const auto &ids : banks) outputs.emplace_back(&f.s, ids);
      // One complete pass per variant/bank validates timed shapes too.
      for (const auto &v : variants) for (auto &out : outputs) { v.call(&out.work); out.check(); }
      const int iterations = std::max<int>(8, 131072 / count);
      std::array<std::vector<double>, variants.size()> times;
      for (int trial = -2; trial < repeats; ++trial) {
        std::array<size_t, variants.size()> order{};
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        for (size_t k : order) {
          const auto t0 = Clock::now();
          for (int j = 0; j < iterations; ++j) {
            auto &work = outputs[(j + trial + 2) % banks_count].work;
            variants[k].call(&work);
            asm volatile("" ::: "memory");
          }
          const double us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / iterations;
          if (trial >= 0) times[k].push_back(us);
        }
      }
      for (size_t k = 0; k < variants.size(); ++k) {
        std::vector<double> ratios;
        for (int trial = 0; trial < repeats; ++trial) ratios.push_back(times[0][trial] / times[k][trial]);
        std::printf("{\"kind\":\"timing\",\"rows\":%zu,\"pattern\":\"%s\",\"variant\":\"%s\","
                    "\"median_us\":%.6f,\"p10_us\":%.6f,\"p90_us\":%.6f,\"paired_speedup\":%.6f,"
                    "\"iterations\":%d,\"repeats\":%d,\"samples_us\":[",
                    count, pattern.c_str(), variants[k].name, quantile(times[k], .5),
                    quantile(times[k], .1), quantile(times[k], .9), quantile(ratios, .5), iterations, repeats);
        for (size_t j = 0; j < times[k].size(); ++j) std::printf("%s%.6f", j ? "," : "", times[k][j]);
        std::puts("]}");
      }
      if (f.s.misses != 0) throw std::runtime_error("unexpected miss in hit-only timing");
      std::fflush(stdout);
    }
  }
}

int main(int argc, char **argv) {
  try {
    int cpu = -1, repeats = 15, cache_mib = 64;
    bool check_only = false;
    for (int i = 1; i < argc; ++i) {
      const std::string arg(argv[i]);
      if (arg == "--check-only") check_only = true;
      else if (i + 1 < argc && arg == "--cpu") cpu = std::stoi(argv[++i]);
      else if (i + 1 < argc && arg == "--repeats") repeats = std::stoi(argv[++i]);
      else if (i + 1 < argc && arg == "--cache-mib") cache_mib = std::stoi(argv[++i]);
      else throw std::runtime_error("usage: --cpu N [--repeats 3..31] [--cache-mib 1..64] [--check-only]");
    }
    if (cpu < 0 || cpu >= CPU_SETSIZE || repeats < 3 || repeats > 31 || cache_mib < 1 || cache_mib > 64)
      throw std::runtime_error("invalid bounded benchmark settings");
    if (!(getauxval(AT_HWCAP) & HWCAP_SVE) || !(getauxval(AT_HWCAP2) & HWCAP2_SVE2))
      throw std::runtime_error("SVE2 required; refusing unsupported execution");
    cpu_set_t mask; CPU_ZERO(&mask); CPU_SET(cpu, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask)) throw std::runtime_error("CPU affinity failed");
    // Once a process has threads, glibc uses its actual threaded mutex path.
    std::promise<void> stop;
    auto done = stop.get_future();
    std::thread parked([&] { done.wait(); });
    struct Join { std::promise<void> &p; std::thread &t; ~Join() { p.set_value(); t.join(); } } join{stop, parked};
    std::printf("{\"kind\":\"metadata\",\"cpu\":%d,\"sve_bytes\":%u,\"cache_mib\":%d,"
                "\"compiler\":\"%s\",\"ssd_io\":false,\"cuda\":false,\"production_pool\":false}\n",
                cpu, vector_bytes(), cache_mib, __VERSION__);
    correctness();
    if (!check_only) benchmark(repeats, cache_mib);
    struct rusage ru{}; getrusage(RUSAGE_SELF, &ru);
    std::printf("{\"kind\":\"complete\",\"max_rss_kib\":%ld,\"cpu_seconds\":%.3f}\n",
                ru.ru_maxrss, ru.ru_utime.tv_sec + ru.ru_utime.tv_usec / 1e6 + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec / 1e6);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "Benchmark refused/failed: %s\n", e.what()); return 1;
  }
}
