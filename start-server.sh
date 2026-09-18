#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Public launcher. No systemd units, tunnels, firewall or driver changes.
set -euo pipefail
RECIPE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${DS41_ENV_FILE:-${RECIPE_DIR}/.env.ds41}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi
exec python3 -B "$RECIPE_DIR/release/launch.py" "$@"
