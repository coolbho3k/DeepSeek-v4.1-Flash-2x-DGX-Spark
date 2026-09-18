// SPDX-License-Identifier: AGPL-3.0-only
// Vendored from MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks
// Commit: 979e68a62c90b24d928f5638596e0ceed90e9f34
// See ../../LICENSE and ../../LICENSE.MIT for license and retained notices.
// DS41 adaptation: K3/MUL1; FP16 GEMM/output boundaries; FP32 pre-down routing.

#pragma once
#include <torch/extension.h>

// Grouped fat-expert MoE for EXL3 trellis experts (prefill), v2.
//
// v1 (GLM-5.3-Flash recipe) was compiled for K=4 / MCG tiles only and fed gate
// and up from ONE Hadamard-transformed input, which requires gate.suh == up.suh.
// v2 keeps the same three launches and the same E2 rounding boundaries but:
//   * templates the mainloop on (bits, codebook): instantiated for
//     (4, mcg), (4, mul1), (3, mul1), (2, mul1); the packed tile is bits*16
//     int16 words, decoded with exllamav3's dq_dispatch<bits, cb>;
//   * takes two gathered inputs (h13g = had128(x * gate.suh), h13u =
//     had128(x * up.suh)), so checkpoints whose gate/up sign vectors differ
//     (DeepSeek-V4.1-Flash EXL3 2.9bpw) run the grouped path too.
//
//   exl3_fat_moe_gather   : h13[row] = had128(x[token[row]] * suh[expert[row]])   (call once per suh table)
//   exl3_fat_moe_gateup   : h2 = had128(silu(clamp(had(g)*svh_g)) * clamp(had(u)*svh_u) * down_suh)
//   exl3_fat_moe_down     : out[token] += had128(h2 @ W_down) * svh_d * route_weight
//
// Segment tables (int32, device): seg_expert / seg_row0 / seg_rows describe
// row tiles of the fat-row buffer (rows are grouped by expert, expert order).
// num_segs / num_rows are 1-element int32 device tensors; kernels are launched
// with capacity-sized, grid-strided grids and read the live counts.
// cb: 1 = MCG, 2 = mul1 (exllamav3 numbering).

void exl3_fat_moe_gather(
    at::Tensor x,            // [tokens, K] half
    at::Tensor row_token,    // [rows_cap] int64
    at::Tensor row_expert,   // [rows_cap] int32
    at::Tensor suh_ptrs,     // [n_exp] int64 (device pointers, half[K])
    at::Tensor h13,          // [rows_cap, K] half (out)
    at::Tensor num_rows);    // [1] int32 device

void exl3_fat_moe_gateup(
    at::Tensor h13g,         // [rows_cap, K] half, gathered with gate.suh
    at::Tensor h13u,         // [rows_cap, K] half, gathered with up.suh (may alias h13g when shared)
    at::Tensor gate_ptrs,    // [n_exp] int64 trellis pointers (K/16, N/16, bits*16) int16
    at::Tensor up_ptrs,
    at::Tensor gate_svh_ptrs,
    at::Tensor up_svh_ptrs,
    at::Tensor down_suh_ptrs,
    at::Tensor h2,           // [rows_cap, N] half (out)
    at::Tensor row_weight,   // [rows_cap] float, applied before down-input FP16 rounding
    at::Tensor seg_expert,
    at::Tensor seg_row0,
    at::Tensor seg_rows,
    at::Tensor num_segs,
    double act_limit,
    int64_t bits,
    int64_t cb);

void exl3_fat_moe_down(
    at::Tensor h2,           // [rows_cap, K] half
    at::Tensor down_ptrs,    // trellis pointers (K/16, N/16, bits*16)
    at::Tensor down_svh_ptrs,
    at::Tensor out,          // [tokens, N] float (accumulated)
    at::Tensor row_token,    // [rows_cap] int64
    at::Tensor row_weight,   // [rows_cap] float; validated but NOT applied again
    at::Tensor seg_expert,
    at::Tensor seg_row0,
    at::Tensor seg_rows,
    at::Tensor num_segs,
    int64_t bits,
    int64_t cb);

// Row tile sizes the segment tables must be built with.
int64_t exl3_fat_moe_tile_rows_gateup();
int64_t exl3_fat_moe_tile_rows_down();
// Interface version: 2 = (bits, cb)-generic kernels with separate gate/up inputs.
int64_t exl3_fat_moe_abi();

// Private ABI 1003 supersedes upstream ABI 2: DS41 FP16 boundaries and pre-down routing.
