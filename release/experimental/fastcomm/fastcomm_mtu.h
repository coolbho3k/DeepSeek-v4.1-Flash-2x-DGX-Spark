// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <cstdint>

namespace ds41_fastcomm {
// ibv_mtu encodes 256..4096 bytes as 1..5 in increasing order. Zero is also
// the old bootstrap padding value: reject it rather than mix old/new peers.
constexpr bool valid_mtu(uint32_t value) { return value >= 1 && value <= 5; }
constexpr uint32_t negotiate_mtu(uint32_t local, uint32_t peer) {
  return !valid_mtu(local) || !valid_mtu(peer) ? 0 : (local < peer ? local : peer);
}
}  // namespace ds41_fastcomm
