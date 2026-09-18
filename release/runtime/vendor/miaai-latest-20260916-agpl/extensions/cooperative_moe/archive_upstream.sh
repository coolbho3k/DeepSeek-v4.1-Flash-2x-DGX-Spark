#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Attribution: MiaAI Lab, Wesley Young and upstream contributors; derived ExLlamaV3 code by Turboderp.
# Upstream: MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks @ 8404ac7d389c418300d0bee960d52313247930e1
# Local change: provenance/license prefix only; original body follows unchanged.
# See repository LICENSE, LICENSE.MIT and native/LICENSE.exllamav3.

# Host-side helper: archive the pinned ExLlamaV3 headers. The serving image has
# no git; do not run this inside the recipe container.
set -euo pipefail

pin=02aef45cd681b960a00afcd0749a4ab99e6c1bfe
checkout=${1:?Usage: archive_upstream.sh EXLLAMAV3_CHECKOUT EMPTY_OUTPUT_DIRECTORY}
output_dir=${2:?Usage: archive_upstream.sh EXLLAMAV3_CHECKOUT EMPTY_OUTPUT_DIRECTORY}

mkdir -p -- "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd)
test -z "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" || {
  echo 'Refusing a nonempty upstream output directory.' >&2
  exit 1
}

command -v git >/dev/null || {
  echo 'git is required on the host, not in the recipe image.' >&2
  exit 1
}
test "$(git -C "$checkout" rev-parse "$pin^{commit}")" = "$pin"
git -C "$checkout" archive "$pin" exllamav3/exllamav3_ext | tar -x -C "$output_dir"
test -d "$output_dir/exllamav3/exllamav3_ext"
