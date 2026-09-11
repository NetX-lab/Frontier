#!/usr/bin/env bash
# Run the real-data MoE EP matrix.  The Python harness fails fast on missing
# profiles, illegal topology, or the first failed case unless the caller opts
# into --continue-on-failure for an explicit diagnostic sweep.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi

ARGS=(
  --repo-root "$REPO_ROOT"
  --task-dir "$REPO_ROOT/task_memory/task_2026-08-12_moe_ep_rank_stragger_analysis"
)
# Only forward --output-root when the caller set MATRIX_OUTPUT_ROOT explicitly.
# Otherwise the Python entry point derives its default from FRONTIER_TMP_ROOT.
if [ -n "${MATRIX_OUTPUT_ROOT:-}" ]; then
  ARGS+=(--output-root "$MATRIX_OUTPUT_ROOT")
fi
ARGS+=(--mode "${MATRIX_MODE:-run}")

exec "$PYTHON_BIN" "$REPO_ROOT/tests/e2e/moe_ep_non_dummy_matrix.py" "${ARGS[@]}" "$@"
