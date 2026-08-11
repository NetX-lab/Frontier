#!/bin/bash
# =============================================================================
# Profiling Recipe - DeepSeek MLA Attention (optionally via real AITER kernels)
# =============================================================================
# Companion to profile_true_mixed_batch.sh, for the *other* fidelity gap
# documented in ../AITER_KERNELS.md: real sglang serving on MI355X dispatches
# DeepSeek's MLA attention through AMD's actual production `aiter` kernels
# (AiterMlaAttentionWrapper); Frontier's profiling has mostly used the
# portable TORCH_SDPA_MLA reference backend instead (see
# ../DEEPSEEK_V3_MLA_MI355X_JOURNEY.md for how that backend was built).
#
# IMPORTANT - checkout-dependent, unlike the true-mixed-batch script:
#   The AITER backend (AttentionBackend.AITER -> AiterMlaAttentionWrapper)
#   only exists on server3 (amd-mi355x-3) and server8 (amd-mi355x-8), in
#   ~/frontier-work/Frontier. It is NOT present in this checkout, in
#   /home/dn/driventes-frontier, or on server1 -- see
#   ../INFRASTRUCTURE_MAP.md for the full checkout inventory. This script
#   preflight-checks for it and fails fast with a clear pointer instead of
#   letting a real run crash deep inside backend dispatch.
#
# Also NOTE (unlike profile_true_mixed_batch.sh's grid, which was run and
# sanity-checked end-to-end -- see GPTOSS_TRUE_MIXED_BATCH_PROFILING.md): the
# default batch/seq-len grid below is deliberately small (sanity-check scale,
# matching the tiny min=max=1 batch run DEEPSEEK_V3_MLA_MI355X_JOURNEY.md used
# to first validate the MLA wrapper), not a validated production sweep --
# there is no single "final validated AITER sweep command" recorded anywhere
# in this session's notes to reproduce faithfully. Widen
# --min-batch-size/--max-batch-size/--max-seq-len deliberately once you know
# what real workload shape you're targeting (see REAL_BENCHMARK_DATA_QUALITY.md
# and GPTOSS_TRUE_MIXED_BATCH_PROFILING.md's "narrow known / widen uncertain"
# design principle -- the same idea applies here).
#
# Use --dry-run to print the resolved command (and run the preflight check)
# without touching a GPU.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# --- Model ------------------------------------------------------------------
# deepseek-r1-0528 matches the real sglang captures under
# tools/inference_bench/deepseek/ that this data is meant to validate against.
# deepseek-v3 (DEEPSEEK_V3_MLA_MI355X_JOURNEY.md's model) is the other
# available MLA+MoE config in data/config/models/ if that's what you need.
MODEL="${MODEL:-deepseek-r1-0528}"

# --- Attention backend --------------------------------------------------
# AITER  = real AMD production MLA kernels (server3/server8 only today).
# TORCH_SDPA_MLA = portable reference backend, works on every checkout, but
#                  explicitly NOT peak-tuned (see AITER_KERNELS.md).
ATTENTION_BACKEND="${ATTENTION_BACKEND:-AITER}"
SKIP_AITER_PREFLIGHT="${SKIP_AITER_PREFLIGHT:-false}"

# --- Hardware / launch shape -------------------------------------------------
DEVICE="${DEVICE:-mi355x}"
NUM_GPUS="${NUM_GPUS:-8}"
TP="${TP:-1 2 4 8}"                 # DEEPSEEK_V3_MLA_MI355X_JOURNEY.md ran the full TP sweep; attn_tp=8 is what the final real simulation used
PP="${PP:-1}"
GPU_IDS="${GPU_IDS:-$(seq -s, 0 $((NUM_GPUS - 1)))}"

PROFILE_METHOD="${PROFILE_METHOD:-cuda_event}"
PRECISION="${PRECISION:-BF16}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"       # confirmed MLA convention -- see GPTOSS_TRUE_MIXED_BATCH_PROFILING.md's block_size table

# --- Grid (sanity-check scale by default -- see header note) ---------------
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
MIN_BATCH_SIZE="${MIN_BATCH_SIZE:-1}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-2}"
FIXED_CHUNKED_PREFILL_SIZE="${FIXED_CHUNKED_PREFILL_SIZE:-512}"
ENABLE_CHUNKED_PREFILL_GRID_SEARCH="${ENABLE_CHUNKED_PREFILL_GRID_SEARCH:-true}"

# --- Output / logging --------------------------------------------------------
DATA_DIR_BASE="${DATA_DIR_BASE:-$REPO_ROOT/data/profiling}"
LOG_DIR="${LOG_DIR:-$DATA_DIR_BASE/logs}"
ASSUME_YES="${ASSUME_YES:-true}"
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
    --attention-backend) require_cli_value "$1" "${2-}"; ATTENTION_BACKEND="$2"; shift 2 ;;
    --device) require_cli_value "$1" "${2-}"; DEVICE="$2"; shift 2 ;;
    --num-gpus) require_cli_value "$1" "${2-}"; NUM_GPUS="$2"; shift 2 ;;
    --tp) require_cli_value "$1" "${2-}"; TP="$2"; shift 2 ;;
    --pp) require_cli_value "$1" "${2-}"; PP="$2"; shift 2 ;;
    --gpu-ids) require_cli_value "$1" "${2-}"; GPU_IDS="$2"; shift 2 ;;
    --profile-method) require_cli_value "$1" "${2-}"; PROFILE_METHOD="$2"; shift 2 ;;
    --precision) require_cli_value "$1" "${2-}"; PRECISION="$2"; shift 2 ;;
    --block-size) require_cli_value "$1" "${2-}"; BLOCK_SIZE="$2"; shift 2 ;;
    --max-seq-len) require_cli_value "$1" "${2-}"; MAX_SEQ_LEN="$2"; shift 2 ;;
    --min-batch-size) require_cli_value "$1" "${2-}"; MIN_BATCH_SIZE="$2"; shift 2 ;;
    --max-batch-size) require_cli_value "$1" "${2-}"; MAX_BATCH_SIZE="$2"; shift 2 ;;
    --fixed-chunked-prefill-size) require_cli_value "$1" "${2-}"; FIXED_CHUNKED_PREFILL_SIZE="$2"; shift 2 ;;
    --disable-chunked-prefill-grid-search) ENABLE_CHUNKED_PREFILL_GRID_SEARCH=false; shift ;;
    --output-root) require_cli_value "$1" "${2-}"; DATA_DIR_BASE="$2"; shift 2 ;;
    --log-dir) require_cli_value "$1" "${2-}"; LOG_DIR="$2"; shift 2 ;;
    --no-yes) ASSUME_YES=false; shift ;;
    --skip-aiter-preflight) SKIP_AITER_PREFLIGHT=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --) shift; break ;;
    *) echo "ERROR: unknown option: $1" >&2; exit 2 ;;
  esac
done

require_bool "DRY_RUN" "$DRY_RUN"
require_bool "ASSUME_YES" "$ASSUME_YES"
require_bool "ENABLE_CHUNKED_PREFILL_GRID_SEARCH" "$ENABLE_CHUNKED_PREFILL_GRID_SEARCH"
require_bool "SKIP_AITER_PREFLIGHT" "$SKIP_AITER_PREFLIGHT"

declare -a TP_SIZE_ARGS
parse_positive_integer_list "TP" "$TP" TP_SIZE_ARGS

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: PYTHON_BIN is not executable or not on PATH: $PYTHON_BIN" >&2
  exit 2
fi

# --- AITER preflight check --------------------------------------------------
# Fails fast on the confirmed gap instead of letting a real profiling run
# crash deep inside get_attention_wrapper() dispatch. See AITER_KERNELS.md.
if [ "$ATTENTION_BACKEND" = "AITER" ] && [ "$SKIP_AITER_PREFLIGHT" != "true" ]; then
  WRAPPER_FILE="$REPO_ROOT/frontier/profiling/attention/backends/aiter_mla_attention_wrapper.py"
  BACKENDS_INIT="$REPO_ROOT/frontier/profiling/attention/backends/__init__.py"
  if [ ! -f "$WRAPPER_FILE" ] || ! grep -q '"AITER"' "$BACKENDS_INIT" 2>/dev/null; then
    cat >&2 <<EOF
ERROR: --attention-backend AITER requested, but this checkout ($REPO_ROOT) has
no AiterMlaAttentionWrapper / AttentionBackend.AITER value.

Confirmed present only on server3 (amd-mi355x-3) and server8 (amd-mi355x-8),
in ~/frontier-work/Frontier -- not here, not in /home/dn/driventes-frontier,
and not on server1. See profiling_knowledge/AITER_KERNELS.md and
profiling_knowledge/INFRASTRUCTURE_MAP.md.

Options:
  1. Run this script on server3 or server8 instead.
  2. Sync aiter_mla_attention_wrapper.py + backends/__init__.py from one of
     those hosts into this checkout first.
  3. Use the portable reference backend instead:
       --attention-backend TORCH_SDPA_MLA
  4. If you've verified AITER is actually available here and this check is
     wrong, bypass it with --skip-aiter-preflight.
EOF
    exit 2
  fi
fi

mkdir -p "$LOG_DIR"

CMD=(
  "$PYTHON_BIN" -m frontier.profiling.attention.main
  --disable_ray --num_gpus "$NUM_GPUS" --device "$DEVICE"
  --models "$MODEL"
  --num_tensor_parallel_workers "${TP_SIZE_ARGS[@]}" --max_pipeline_parallel_size "$PP"
  --attention_backend "$ATTENTION_BACKEND" --profile_method "$PROFILE_METHOD" --precision "$PRECISION"
  --block_size "$BLOCK_SIZE"
  --max_seq_len "$MAX_SEQ_LEN"
  --min_batch_size "$MIN_BATCH_SIZE" --max_batch_size "$MAX_BATCH_SIZE"
  --fixed_chunked_prefill_size "$FIXED_CHUNKED_PREFILL_SIZE"
  --output_dir "$DATA_DIR_BASE"
)
if [ "$ENABLE_CHUNKED_PREFILL_GRID_SEARCH" = "true" ]; then
  CMD+=(--enable_chunked_prefill_grid_search)
fi
if [ "$ASSUME_YES" = "true" ]; then
  CMD+=(--yes)
fi
if [ "$#" -gt 0 ]; then
  CMD+=("$@")
fi

MODEL_DIR="$DATA_DIR_BASE/compute/$DEVICE/$MODEL"
LOG_FILE="$LOG_DIR/profile_$(basename "$MODEL")_${ATTENTION_BACKEND,,}_block${BLOCK_SIZE}.log"

cat <<EOF
============================================================
  Profiling Recipe - DeepSeek MLA Attention
============================================================
Model:              $MODEL
Attention backend:  $ATTENTION_BACKEND ($PROFILE_METHOD, $PRECISION)
Device / TP / PP:   $DEVICE / TP=${TP_SIZE_ARGS[*]} / PP=$PP ($NUM_GPUS GPUs, ids: $GPU_IDS)
block_size:         $BLOCK_SIZE
max_seq_len:        $MAX_SEQ_LEN  (sanity-check-scale default -- see header note; widen deliberately)
batch_size range:   [$MIN_BATCH_SIZE, $MAX_BATCH_SIZE]
chunked_prefill:    fixed=$FIXED_CHUNKED_PREFILL_SIZE  grid_search=$ENABLE_CHUNKED_PREFILL_GRID_SEARCH
Output:             $MODEL_DIR/attention.csv
Log:                $LOG_FILE
Dry run:            $DRY_RUN
============================================================
EOF

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [ "$DRY_RUN" = "true" ]; then
  echo "Dry run completed; no profiling command was executed (AITER preflight check, if applicable, still ran above)."
  exit 0
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export HIP_VISIBLE_DEVICES="$GPU_IDS"   # see INFRASTRUCTURE_MAP.md's nvidia-smi GPU-discovery gotcha

cd "$REPO_ROOT"
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

if [ ! -f "$MODEL_DIR/attention.csv" ]; then
  echo "ERROR: expected profiling output was not generated: $MODEL_DIR/attention.csv" >&2
  exit 1
fi

echo "DeepSeek MLA attention profiling completed: $MODEL_DIR/attention.csv"
echo "NOTE: this run's attention.csv has NO true-mixed-batch rows (that's the"
echo "gpt-oss/qwen3-specific gap fixed by profile_true_mixed_batch.sh; MLA does"
echo "not need it -- see GPTOSS_TRUE_MIXED_BATCH_PROFILING.md and AITER_KERNELS.md"
echo "for why the two gaps are independent)."
