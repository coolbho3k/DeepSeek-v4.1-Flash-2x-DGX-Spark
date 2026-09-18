#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Attribution: MiaAI Lab, Wesley Young and upstream contributors; derived ExLlamaV3 code by Turboderp.
# Upstream: MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks @ 8404ac7d389c418300d0bee960d52313247930e1
# Local change: provenance/license prefix only; original body follows unchanged.
# See repository LICENSE, LICENSE.MIT and native/LICENSE.exllamav3.

# Compile artifacts only; never select a profile or modify a running service.
# Requires an already-extracted pinned ExLlamaV3 tree (see archive_upstream.sh).
# Does not call git: the recipe image has no git executable.
set -euo pipefail

upstream_tree=${1:?Usage: build.sh EXTRACTED_EXLLAMAV3_TREE EMPTY_OUTPUT_DIRECTORY}
output_dir=${2:?Usage: build.sh EXTRACTED_EXLLAMAV3_TREE EMPTY_OUTPUT_DIRECTORY}
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

test -d "$upstream_tree/exllamav3/exllamav3_ext"
mkdir -p -- "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd)
test -z "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" || {
  echo 'Refusing a nonempty build output directory.' >&2
  exit 1
}

mkdir -- "$output_dir/upstream"
cp -a -- "$upstream_tree/exllamav3" "$output_dir/upstream/exllamav3"

# Preserve the validated compiler input names and ABI v1 symbols. Changing these
# can change the binary hash even when device arithmetic is identical.
cp -- "$source_dir/native/cooperative_moe.cu" "$output_dir/goal50_fixed_coop.cu"
cp -- "$source_dir/native/cooperative_moe_kernel.cuh" \
  "$output_dir/upstream/exllamav3/exllamav3_ext/quant/goal50_fixed_coop_kernel.cuh"
cp -- "$source_dir/native/exl3_moe_coop.cuh" \
  "$output_dir/upstream/exllamav3/exllamav3_ext/quant/exl3_moe_coop.cuh"
cp -- "$source_dir/runtime.py" "$output_dir/runtime.py"

# -lineinfo plus the GNU build-id make nvcc output non-reproducible across clean
# runs of the same command. Keep these flags for ABI compatibility with the
# GPU-validated artifact; do not treat a local rebuild as the release pin.
"${NVCC:-/usr/local/cuda/bin/nvcc}" -std=c++17 -O3 --use_fast_math -lineinfo --expt-relaxed-constexpr \
  -gencode arch=compute_121a,code=sm_121a -shared -Xcompiler -fPIC --ptxas-options=-v \
  -I "$output_dir/upstream/exllamav3/exllamav3_ext" \
  "$output_dir/goal50_fixed_coop.cu" -o "$output_dir/goal50-fixed-coop.so" \
  > "$output_dir/cooperative_moe-build.log" 2>&1
mv -- "$output_dir/goal50-fixed-coop.so" "$output_dir/cooperative_moe.so"
sha256sum "$output_dir/cooperative_moe.so" "$output_dir/runtime.py"
