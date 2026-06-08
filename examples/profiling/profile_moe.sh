#!/bin/bash
# =============================================================================
# Profiling Example - MoE Operators
# =============================================================================
# Profiles Mixture-of-Experts compute and writes the canonical CSV to:
#   data/profiling/compute/<device>/<model>/moe.csv
#
# Use --dry-run to validate the command without requiring a GPU profiling stack.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-Qwen3-30B-A3B-tiny}"
DEVICE="${DEVICE:-rtx_pro_6000}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_TOKENS="${MAX_TOKENS:-16}"
TP_SIZES="${TP_SIZES:-1}"
EP_SIZES="${EP_SIZES:-1}"
PROFILE_METHOD="${PROFILE_METHOD:-cuda_event}"
ROUTING_RUNTIME_PATH="${ROUTING_RUNTIME_PATH:-uniform_topk}"
GATING_RUNTIME_CONTEXT="${GATING_RUNTIME_CONTEXT:-prefill_hot}"
DATA_DIR_BASE="${DATA_DIR_BASE:-$REPO_ROOT/data/profiling}"
DRY_RUN="${DRY_RUN:-false}"

require_cli_value() {
  local option="$1"
  local value="${2-}"
  if [ -z "$value" ] || [[ "$value" == --* ]]; then
    echo "ERROR: $option requires a value" >&2
    exit 2
  fi
}

require_bool() {
  local name="$1"
  local value="$2"
  if [ "$value" != "true" ] && [ "$value" != "false" ]; then
    echo "ERROR: $name must be true or false; got $value" >&2
    exit 2
  fi
}

parse_positive_integer_list() {
  local name="$1"
  local value="$2"
  local -n output_ref="$3"

  read -r -a output_ref <<< "$value"
  if [ "${#output_ref[@]}" -eq 0 ]; then
    echo "ERROR: $name must contain positive integer values; got empty value" >&2
    exit 2
  fi

  local item
  for item in "${output_ref[@]}"; do
    if [[ ! "$item" =~ ^[1-9][0-9]*$ ]]; then
      echo "ERROR: $name must contain positive integer values; got $value" >&2
      exit 2
    fi
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) require_cli_value "$1" "${2-}"; MODEL="$2"; shift 2 ;;
    --device) require_cli_value "$1" "${2-}"; DEVICE="$2"; shift 2 ;;
    --num-gpus) require_cli_value "$1" "${2-}"; NUM_GPUS="$2"; shift 2 ;;
    --max-tokens) require_cli_value "$1" "${2-}"; MAX_TOKENS="$2"; shift 2 ;;
    --tp-sizes) require_cli_value "$1" "${2-}"; TP_SIZES="$2"; shift 2 ;;
    --ep-sizes) require_cli_value "$1" "${2-}"; EP_SIZES="$2"; shift 2 ;;
    --profile-method) require_cli_value "$1" "${2-}"; PROFILE_METHOD="$2"; shift 2 ;;
    --routing-runtime-path) require_cli_value "$1" "${2-}"; ROUTING_RUNTIME_PATH="$2"; shift 2 ;;
    --gating-runtime-context) require_cli_value "$1" "${2-}"; GATING_RUNTIME_CONTEXT="$2"; shift 2 ;;
    --output-root) require_cli_value "$1" "${2-}"; DATA_DIR_BASE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --) shift; break ;;
    *) echo "ERROR: unknown option: $1" >&2; exit 2 ;;
  esac
done

require_bool "DRY_RUN" "$DRY_RUN"
declare -a TP_SIZE_ARGS
declare -a EP_SIZE_ARGS
parse_positive_integer_list "TP_SIZES" "$TP_SIZES" TP_SIZE_ARGS
parse_positive_integer_list "EP_SIZES" "$EP_SIZES" EP_SIZE_ARGS

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: PYTHON_BIN is not executable or not on PATH: $PYTHON_BIN" >&2
  exit 2
fi

CMD=(
  "$PYTHON_BIN" -m frontier.profiling.moe.main
  --disable_ray
  --models "$MODEL"
  --device "$DEVICE"
  --num_gpus "$NUM_GPUS"
  --max_tokens "$MAX_TOKENS"
  --num_tensor_parallel_workers "${TP_SIZE_ARGS[@]}"
  --expert_parallel_sizes "${EP_SIZE_ARGS[@]}"
  --profile_method "$PROFILE_METHOD"
  --routing_runtime_path "$ROUTING_RUNTIME_PATH"
  --gating_runtime_context "$GATING_RUNTIME_CONTEXT"
  --disable_load_imbalance
  --output_dir "$DATA_DIR_BASE"
)

if [ "$#" -gt 0 ]; then
  CMD+=("$@")
fi

cat <<EOF
============================================
  Profiling Example - MoE Operators
============================================
Output taxonomy: data/profiling/compute/<device>/<model>/moe.csv
Resolved output: $DATA_DIR_BASE/compute/$DEVICE/$MODEL/moe.csv
Model: $MODEL
Device: $DEVICE
TP sizes: $TP_SIZES
EP sizes: $EP_SIZES
Profile method: $PROFILE_METHOD
Dry run: $DRY_RUN
============================================
EOF

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [ "$DRY_RUN" = "true" ]; then
  echo "Dry run completed; no profiling command was executed."
  exit 0
fi

cd "$REPO_ROOT"
"${CMD[@]}"

OUTPUT_CSV="$DATA_DIR_BASE/compute/$DEVICE/$MODEL/moe.csv"
if [ ! -f "$OUTPUT_CSV" ]; then
  echo "ERROR: expected profiling output was not generated: $OUTPUT_CSV" >&2
  exit 1
fi

echo "MoE profiling completed: $OUTPUT_CSV"
