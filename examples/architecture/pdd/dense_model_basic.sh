#!/bin/bash
# =============================================================================
# Compatibility entrypoint for the PDD dense offline example
# =============================================================================
# The canonical pre-release-v0.2 PDD example layout is:
#   examples/architecture/pdd/offline/dense_model_basic.sh
#   examples/architecture/pdd/online/dense_model_basic_online.sh
#
# Keep this top-level script as a backward-compatible alias for users who ran
# the early PDD dense smoke entrypoint before the offline/online split.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/offline/dense_model_basic.sh" "$@"
