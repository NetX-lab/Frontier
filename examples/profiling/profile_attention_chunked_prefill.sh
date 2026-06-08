#!/bin/bash
# =============================================================================
# Profiling Example - Attention Chunked Prefill
# =============================================================================
# Profiles attention prefill under a chunked prefill runtime state and writes:
#   data/profiling/compute/<device>/<model>/attention.csv
#
# Use --dry-run to validate the command without requiring a GPU profiling stack.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-qwen2_dense_test}"
DEVICE="${DEVICE:-rtx_pro_6000}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-128}"
TP_SIZES="${TP_SIZES:-1}"
PP_SIZES="${PP_SIZES:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-NO_OP}"
PROFILE_METHOD="${PROFILE_METHOD:-cuda_event}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MIN_BATCH_SIZE="${MIN_BATCH_SIZE:-1}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-2}"
FIXED_CHUNKED_PREFILL_SIZE="${FIXED_CHUNKED_PREFILL_SIZE:-64}"
ENABLE_CHUNKED_PREFILL_GRID_SEARCH="${ENABLE_CHUNKED_PREFILL_GRID_SEARCH:-true}"
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
    --max-seq-len) require_cli_value "$1" "${2-}"; MAX_SEQ_LEN="$2"; shift 2 ;;
    --tp-sizes) require_cli_value "$1" "${2-}"; TP_SIZES="$2"; shift 2 ;;
    --pp-sizes) require_cli_value "$1" "${2-}"; PP_SIZES="$2"; shift 2 ;;
    --attention-backend) require_cli_value "$1" "${2-}"; ATTENTION_BACKEND="$2"; shift 2 ;;
    --profile-method) require_cli_value "$1" "${2-}"; PROFILE_METHOD="$2"; shift 2 ;;
    --fixed-chunked-prefill-size) require_cli_value "$1" "${2-}"; FIXED_CHUNKED_PREFILL_SIZE="$2"; shift 2 ;;
    --disable-chunked-prefill-grid-search) ENABLE_CHUNKED_PREFILL_GRID_SEARCH=false; shift ;;
    --output-root) require_cli_value "$1" "${2-}"; DATA_DIR_BASE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --) shift; break ;;
    *) echo "ERROR: unknown option: $1" >&2; exit 2 ;;
  esac
done

require_bool "DRY_RUN" "$DRY_RUN"
require_bool "ENABLE_CHUNKED_PREFILL_GRID_SEARCH" "$ENABLE_CHUNKED_PREFILL_GRID_SEARCH"
declare -a TP_SIZE_ARGS
declare -a PP_SIZE_ARGS
parse_positive_integer_list "TP_SIZES" "$TP_SIZES" TP_SIZE_ARGS
parse_positive_integer_list "PP_SIZES" "$PP_SIZES" PP_SIZE_ARGS

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: PYTHON_BIN is not executable or not on PATH: $PYTHON_BIN" >&2
  exit 2
fi

if [ "$FIXED_CHUNKED_PREFILL_SIZE" -le 0 ]; then
  echo "ERROR: FIXED_CHUNKED_PREFILL_SIZE must be positive for the chunked prefill recipe; got $FIXED_CHUNKED_PREFILL_SIZE" >&2
  exit 2
fi

CMD=(
  "$PYTHON_BIN" -m frontier.profiling.attention.main
  --disable_ray
  --models "$MODEL"
  --num_gpus "$NUM_GPUS"
  --max_seq_len "$MAX_SEQ_LEN"
  --num_tensor_parallel_workers "${TP_SIZE_ARGS[@]}"
  --max_pipeline_parallel_size "${PP_SIZE_ARGS[@]}"
  --attention_backend "$ATTENTION_BACKEND"
  --block_size "$BLOCK_SIZE"
  --min_batch_size "$MIN_BATCH_SIZE"
  --max_batch_size "$MAX_BATCH_SIZE"
  --profile_only_prefill
  --fixed_chunked_prefill_size "$FIXED_CHUNKED_PREFILL_SIZE"
  --device "$DEVICE"
  --profile_method "$PROFILE_METHOD"
  --output_dir "$DATA_DIR_BASE"
)

if [ "$ENABLE_CHUNKED_PREFILL_GRID_SEARCH" = "true" ]; then
  CMD+=(--enable_chunked_prefill_grid_search)
fi

if [ "$#" -gt 0 ]; then
  CMD+=("$@")
fi

cat <<EOF
============================================
  Profiling Example - Attention Chunked Prefill
============================================
Output taxonomy: data/profiling/compute/<device>/<model>/attention.csv
Resolved output: $DATA_DIR_BASE/compute/$DEVICE/$MODEL/attention.csv
Model: $MODEL
Device: $DEVICE
TP sizes: $TP_SIZES
Chunked prefill fixed size: $FIXED_CHUNKED_PREFILL_SIZE
Chunked prefill grid search: $ENABLE_CHUNKED_PREFILL_GRID_SEARCH
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

OUTPUT_CSV="$DATA_DIR_BASE/compute/$DEVICE/$MODEL/attention.csv"
if [ ! -f "$OUTPUT_CSV" ]; then
  echo "ERROR: expected profiling output was not generated: $OUTPUT_CSV" >&2
  exit 1
fi

echo "Attention chunked prefill profiling completed: $OUTPUT_CSV"
