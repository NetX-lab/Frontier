#!/usr/bin/env bash
# One JSON/text diagnostic per replay; native inputs remain read-only.
set -euo pipefail
: "${PYTHON_BIN:?Set PYTHON_BIN to the Frontier CPU Python executable}"
: "${A10_RUN_ROOT:?Set A10_RUN_ROOT to a completed observed run}"
: "${NATIVE_ARTIFACTS:?Set NATIVE_ARTIFACTS to the matching native artifacts directory}"
: "${B21_OUTPUT_ROOT:?Set B21_OUTPUT_ROOT to private scratch storage}"
B21_TOOLS="$(cd "$(dirname "$0")" && pwd)"
B21_REPO="$(cd "$B21_TOOLS/../../../.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
ulimit -v 62914560
mkdir -p "$B21_OUTPUT_ROOT"
timeout 600 "$PYTHON_BIN" "$B21_REPO/tests/analysis/v4_replay_census.py" \
    --run-root "$A10_RUN_ROOT" --native-artifacts "$NATIVE_ARTIFACTS" \
    --output "$B21_OUTPUT_ROOT/comparison.json" --summary "$B21_OUTPUT_ROOT/comparison.txt"
