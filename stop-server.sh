#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Stop only the pair recorded by this recipe's public launcher.
set -euo pipefail
RECIPE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$RECIPE_DIR/start-server.sh" stop "$@"
