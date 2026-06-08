#!/bin/bash
# =============================================================================
# Co-location Offline/Online Example Suite
# =============================================================================
# Runs all release-supported co-location examples. Override CASE_FILTER with a
# substring to run a smaller slice, and pass additional Frontier CLI flags after
# "--"; those flags are forwarded to each case.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CASE_FILTER="${CASE_FILTER:-}"

CASES=(
  "offline/dense_model_basic.sh"
  "offline/moe_model_basic.sh"
  "offline/thinking_mode_basic.sh"
  "offline/moe_spec_dec.sh"
  "offline/moe_prefix_caching.sh"
  "online/dense_model_basic_online.sh"
  "online/moe_model_basic_online.sh"
  "online/thinking_mode_basic_online.sh"
  "online/moe_spec_dec_online.sh"
  "online/moe_prefix_caching_online.sh"
)

EXTRA_ARGS=()
if [ "$#" -gt 0 ]; then
  if [ "$1" = "--" ]; then
    shift
  fi
  EXTRA_ARGS=("$@")
fi

executed_cases=0

for case_path in "${CASES[@]}"; do
  if [ -n "$CASE_FILTER" ] && [[ "$case_path" != *"$CASE_FILTER"* ]]; then
    continue
  fi

  executed_cases=$((executed_cases + 1))
  echo "============================================"
  echo "Running co-location case: $case_path"
  echo "============================================"
  bash "$SCRIPT_DIR/$case_path" -- "${EXTRA_ARGS[@]}"
done

if [ "$executed_cases" -eq 0 ]; then
  echo "ERROR: CASE_FILTER matched no co-location cases: $CASE_FILTER" >&2
  exit 2
fi

echo "Completed $executed_cases co-location case(s)."
