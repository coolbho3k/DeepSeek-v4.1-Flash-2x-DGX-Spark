# SPDX-License-Identifier: AGPL-3.0-only
# Vendored from MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks
# Commit: 979e68a62c90b24d928f5638596e0ceed90e9f34
# Copyright/attribution: Mia's AI Lab and upstream contributors.
# See ../LICENSE and ../LICENSE.MIT; original body below is unchanged.
# Local change: this provenance/license prefix only.

#!/usr/bin/env python3
"""The Engram row dequant kernel must decode fp8 e4m3 bytes (not integer byte values).
GPU test; prints a skip line without CUDA (image build)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for _d in (Path("/opt/dsv41"), ROOT / "overlay"):
    if (_d / "engram_file_backend.py").is_file():
        sys.path.insert(0, str(_d))
        break


def main() -> None:
    if not torch.cuda.is_available():
        print("test_engram_dequant: no CUDA here; skipped (runs in the GPU self-check)")
        return
    import engram_file_backend as fb

    kernel = fb._dequant_kernel()
    rows, dim, qb = 64, 256, 32
    torch.manual_seed(0)
    w = torch.randint(0, 256, (rows, dim), dtype=torch.uint8, device="cuda")
    w[w == 0x7F] = 0
    w[w == 0xFF] = 0  # NaN encodings
    s = torch.randint(120, 134, (rows, dim // qb), dtype=torch.uint8, device="cuda")
    ref = w.view(torch.float8_e4m3fn).float() * (2.0 ** (s.float() - 127)).repeat_interleave(qb, dim=1)
    out = torch.empty(rows, dim, dtype=torch.bfloat16, device="cuda")
    kernel[(4,)](w.view(torch.float8_e4m3fn), s, out, rows, DIM=dim, QUANT_BLOCK=qb, BLOCK_R=16, GRID=4)
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert rel < 1e-2, rel
    print(f"test_engram_dequant: ok (rel_err={rel:.2e})")


if __name__ == "__main__":
    main()
