#!/bin/bash
# =============================================================================
# Profiling-independent Architecture Dummy Smoke Matrix
# =============================================================================
# Runs the smallest dense/MoE smoke matrix across co-location/PDD and
# offline/online. The script forces dummy execution-time prediction and does not
# pass profiling CSV input flags to the child examples.
#
# Controls:
#   CASE_FILTER=co-location/online  bash examples/architecture/run_dummy_smoke_matrix.sh
#   NUM_REQUESTS=2 DECODE_TOKENS=4  bash examples/architecture/run_dummy_smoke_matrix.sh
#   bash examples/architecture/run_dummy_smoke_matrix.sh -- --metrics_config_enable_chrome_trace
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CASE_FILTER="${CASE_FILTER:-}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-dummy_smoke}"

export PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export VIDUR_DISABLE_WANDB="${VIDUR_DISABLE_WANDB:-1}"

export ENABLE_DUMMY_MODE=true
export DUMMY_EXEC_TIME_MS="${DUMMY_EXEC_TIME_MS:-1.0}"
export NUM_REQUESTS="${NUM_REQUESTS:-1}"
export PREFILL_TOKENS="${PREFILL_TOKENS:-8}"
export DECODE_TOKENS="${DECODE_TOKENS:-2}"
export QPS="${QPS:-1.0}"
export METRICS_OUTPUT_DIR="${METRICS_OUTPUT_DIR:-$REPO_ROOT/outputs/examples/dummy-smoke-matrix}"

CASES=(
  "co-location/offline/dense_model_basic.sh"
  "co-location/offline/moe_model_basic.sh"
  "co-location/online/dense_model_basic_online.sh"
  "co-location/online/moe_model_basic_online.sh"
  "pdd/offline/dense_model_basic.sh"
  "pdd/offline/moe_model_basic.sh"
  "pdd/online/dense_model_basic_online.sh"
  "pdd/online/moe_model_basic_online.sh"
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

  case_run_id="${RUN_ID_PREFIX}_${case_path//\//_}"
  case_run_id="${case_run_id//./_}"
  case_run_id="${case_run_id//-/_}"

  executed_cases=$((executed_cases + 1))
  echo "============================================"
  echo "Running dummy smoke case: $case_path"
  echo "RUN_ID=$case_run_id"
  echo "============================================"
  RUN_ID="$case_run_id" bash "$SCRIPT_DIR/$case_path" -- "${EXTRA_ARGS[@]}"
done

if [ "$executed_cases" -eq 0 ]; then
  echo "ERROR: CASE_FILTER matched no dummy smoke cases: $CASE_FILTER" >&2
  exit 2
fi

echo "Completed $executed_cases profiling-independent dummy smoke case(s)."
