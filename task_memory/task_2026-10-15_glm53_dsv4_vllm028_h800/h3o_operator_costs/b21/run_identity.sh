#!/usr/bin/env bash
# Run serially for wall-time comparison. Supply a new private scratch root.
set -euo pipefail
: "${PYTHON_BIN:?Set PYTHON_BIN to the Frontier CPU Python executable}"
: "${B21_RUN_ROOT:?Set B21_RUN_ROOT to a new private scratch directory}"
B21_TOOLS="$(cd "$(dirname "$0")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_DISABLED=true
export VIDUR_DISABLE_WANDB=1
ulimit -v 62914560
mkdir -p "$B21_RUN_ROOT"
timeout 7200 "$PYTHON_BIN" "$B21_TOOLS/run_small.py" "$B21_RUN_ROOT/plain" --without-observer > "$B21_RUN_ROOT/plain.log" 2>&1
timeout 7200 "$PYTHON_BIN" "$B21_TOOLS/run_small.py" "$B21_RUN_ROOT/observed" > "$B21_RUN_ROOT/observed.log" 2>&1
timeout 600 "$PYTHON_BIN" "$B21_TOOLS/check_identity.py" "$B21_RUN_ROOT/plain" "$B21_RUN_ROOT/observed" "$B21_RUN_ROOT/identity.json"
