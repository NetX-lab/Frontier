#!/bin/bash
# =============================================================================
# Profiling Recipe - True-Mixed-Batch Attention (closes attn_decode_in_mixed)
# =============================================================================
# Reproduces, as a configurable script, the exact sweep documented in
# ../GPTOSS_TRUE_MIXED_BATCH_PROFILING.md that fixes:
#
#   ValueError: attn_decode_in_mixed prediction is required for true mixed
#   batches but not found for cluster monolithic. Please provide merged
#   attention profiling data via atten_input_file (Option A) and train
#   attn_decode_in_mixed.
#
# Follows the "narrow what's known, widen only what's genuinely uncertain"
# design from that doc:
#   - PINNED (known from the real workload): max_seq_len/max_model_len
#     (input_len + output_len), fixed_chunked_prefill_size (= input_len),
#     true-mixed prefill batch size (1) and chunk size (= input_len).
#   - WIDENED (scheduling-dependent, can't be guessed): decode batch size and
#     decode KV-cache size, for both the regular and true-mixed grids.
#
# Loops over MODELS x BLOCK_SIZES (real block_size was never confirmed for
# every backend -- see the doc's block_size table -- hence the sweep instead
# of a single guess), and renames each run's three output CSVs immediately
# (attention.csv / attention_true_mixed.csv / attention_combined.csv all
# share one overwrite-prone output path keyed only on (device, model,
# profile_method) -- NOT block_size or TP -- so skipping the rename step
# silently destroys the previous block_size's data).
#
# Validated end-to-end for openai/gpt-oss-20b and openai/gpt-oss-120b (~15s
# per model per block_size, 325 rows each -- see the doc's sanity-check
# section). qwen3-a3b-30b-moe hits the identical crash for the identical
# reason (same dense/GQA attention family) and is included below by the same
# recipe, but has NOT been independently run through this exact script --
# treat it as "should work," not "confirmed," until you've run it once and
# checked it against the doc's 4-point sanity check yourself.
#
# Use --dry-run to print every command (and the renames that would follow)
# without touching a GPU.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# --- Models to sweep -------------------------------------------------------
# Space-separated HF-style model ids, exactly as frontier.profiling.attention
# .main --models expects them (must resolve via data/config/models/*.json).
MODELS="${MODELS:-openai/gpt-oss-20b openai/gpt-oss-120b}"

# --- block_size sweep -------------------------------------------------------
# 1 = SGLang's real page_size (confirmed from real server logs).
# 16 = vLLM's assumed default (never independently confirmed -- see the doc's
#      "Confirmed real block_size values" table).
# 32 = kept for backward compatibility with the original DeepSeek profiling
#      data; not confirmed to match either real backend for these models.
BLOCK_SIZES="${BLOCK_SIZES:-1 16 32}"

# --- Hardware / launch shape -------------------------------------------------
DEVICE="${DEVICE:-mi355x}"
NUM_GPUS="${NUM_GPUS:-8}"
TP="${TP:-8}"                       # --num_tensor_parallel_workers; every real capture uses TP=8
PP="${PP:-1}"                       # --max_pipeline_parallel_size
# GPU_IDS backs CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES -- see
# INFRASTRUCTURE_MAP.md's "nvidia-smi GPU-discovery gotcha": on a fresh
# shell/tmux/nohup session frontier.profiling.attention.main's
# _get_available_gpus() has no ROCm fallback and fails outright without this.
GPU_IDS="${GPU_IDS:-$(seq -s, 0 $((NUM_GPUS - 1)))}"

ATTENTION_BACKEND="${ATTENTION_BACKEND:-TORCH_SDPA}"   # portable reference backend; see AITER_KERNELS.md for the (MLA-only, dense/GQA gap still open) alternative
PROFILE_METHOD="${PROFILE_METHOD:-cuda_event}"
PRECISION="${PRECISION:-BF16}"

# --- Real workload shape (drives the PINNED dimensions below) --------------
# Every real capture's config.txt uses input_len=8192, output_len=1024 across
# deepseek/qwen3/gpt-oss -- see REAL_BENCHMARK_DATA_QUALITY.md. Override these
# if you're targeting a different real workload shape.
INPUT_LEN="${INPUT_LEN:-8192}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"

# PINNED dimensions: default to exact derivations from INPUT_LEN/OUTPUT_LEN
# unless explicitly overridden. Left empty here so we can detect "not set by
# caller" below and compute the derived value instead of a hardcoded literal.
MAX_SEQ_LEN="${MAX_SEQ_LEN:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
FIXED_CHUNKED_PREFILL_SIZE="${FIXED_CHUNKED_PREFILL_SIZE:-}"
TRUE_MIXED_PREFILL_CHUNK_SIZES="${TRUE_MIXED_PREFILL_CHUNK_SIZES:-}"
TRUE_MIXED_PREFILL_BATCH_SIZES="${TRUE_MIXED_PREFILL_BATCH_SIZES:-1}"
TRUE_MIXED_PREFILL_KV_CACHE_SIZE="${TRUE_MIXED_PREFILL_KV_CACHE_SIZE:-0}"

# WIDENED dimensions: genuinely uncertain (depend on real scheduling
# dynamics we can't pin down in advance), so these are lists, not points.
# Defaults reproduce the doc's validated sweep exactly.
MIN_BATCH_SIZE="${MIN_BATCH_SIZE:-1}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-512}"
BATCH_SIZE_LIST="${BATCH_SIZE_LIST:-1 2 4 8 16 32 48 64 96 128 160 192 256 320 384 448 512}"
DECODE_KV_CACHE_SIZE_LIST="${DECODE_KV_CACHE_SIZE_LIST:-512 1024 2048 3072 4096 5120 6144 8192 9216}"
TRUE_MIXED_DECODE_BATCH_SIZES="${TRUE_MIXED_DECODE_BATCH_SIZES:-$BATCH_SIZE_LIST}"
TRUE_MIXED_DECODE_KV_CACHE_SIZES="${TRUE_MIXED_DECODE_KV_CACHE_SIZES:-$DECODE_KV_CACHE_SIZE_LIST}"

# --- Output / logging --------------------------------------------------------
DATA_DIR_BASE="${DATA_DIR_BASE:-$REPO_ROOT/data/profiling}"
LOG_DIR="${LOG_DIR:-$DATA_DIR_BASE/logs}"
ASSUME_YES="${ASSUME_YES:-true}"    # adds --yes (non-interactive confirmation)
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

parse_token_list() {
  local name="$1"
  local value="$2"
  local -n output_ref="$3"

  read -r -a output_ref <<< "$value"
  if [ "${#output_ref[@]}" -eq 0 ]; then
    echo "ERROR: $name must contain at least one value; got empty value" >&2
    exit 2
  fi
}

list_max() {
  local -n arr_ref="$1"
  local max="${arr_ref[0]}"
  local v
  for v in "${arr_ref[@]}"; do
    if [ "$v" -gt "$max" ]; then max="$v"; fi
  done
  echo "$max"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) require_cli_value "$1" "${2-}"; MODELS="$2"; shift 2 ;;
    --block-sizes) require_cli_value "$1" "${2-}"; BLOCK_SIZES="$2"; shift 2 ;;
    --device) require_cli_value "$1" "${2-}"; DEVICE="$2"; shift 2 ;;
    --num-gpus) require_cli_value "$1" "${2-}"; NUM_GPUS="$2"; shift 2 ;;
    --tp) require_cli_value "$1" "${2-}"; TP="$2"; shift 2 ;;
    --pp) require_cli_value "$1" "${2-}"; PP="$2"; shift 2 ;;
    --gpu-ids) require_cli_value "$1" "${2-}"; GPU_IDS="$2"; shift 2 ;;
    --attention-backend) require_cli_value "$1" "${2-}"; ATTENTION_BACKEND="$2"; shift 2 ;;
    --profile-method) require_cli_value "$1" "${2-}"; PROFILE_METHOD="$2"; shift 2 ;;
    --precision) require_cli_value "$1" "${2-}"; PRECISION="$2"; shift 2 ;;
    --input-len) require_cli_value "$1" "${2-}"; INPUT_LEN="$2"; shift 2 ;;
    --output-len) require_cli_value "$1" "${2-}"; OUTPUT_LEN="$2"; shift 2 ;;
    --max-seq-len) require_cli_value "$1" "${2-}"; MAX_SEQ_LEN="$2"; shift 2 ;;
    --max-model-len) require_cli_value "$1" "${2-}"; MAX_MODEL_LEN="$2"; shift 2 ;;
    --fixed-chunked-prefill-size) require_cli_value "$1" "${2-}"; FIXED_CHUNKED_PREFILL_SIZE="$2"; shift 2 ;;
    --min-batch-size) require_cli_value "$1" "${2-}"; MIN_BATCH_SIZE="$2"; shift 2 ;;
    --max-batch-size) require_cli_value "$1" "${2-}"; MAX_BATCH_SIZE="$2"; shift 2 ;;
    --batch-size-list) require_cli_value "$1" "${2-}"; BATCH_SIZE_LIST="$2"; shift 2 ;;
    --decode-kv-cache-size-list) require_cli_value "$1" "${2-}"; DECODE_KV_CACHE_SIZE_LIST="$2"; shift 2 ;;
    --true-mixed-prefill-batch-sizes) require_cli_value "$1" "${2-}"; TRUE_MIXED_PREFILL_BATCH_SIZES="$2"; shift 2 ;;
    --true-mixed-prefill-chunk-sizes) require_cli_value "$1" "${2-}"; TRUE_MIXED_PREFILL_CHUNK_SIZES="$2"; shift 2 ;;
    --true-mixed-prefill-kv-cache-size) require_cli_value "$1" "${2-}"; TRUE_MIXED_PREFILL_KV_CACHE_SIZE="$2"; shift 2 ;;
    --true-mixed-decode-batch-sizes) require_cli_value "$1" "${2-}"; TRUE_MIXED_DECODE_BATCH_SIZES="$2"; shift 2 ;;
    --true-mixed-decode-kv-cache-sizes) require_cli_value "$1" "${2-}"; TRUE_MIXED_DECODE_KV_CACHE_SIZES="$2"; shift 2 ;;
    --output-root) require_cli_value "$1" "${2-}"; DATA_DIR_BASE="$2"; shift 2 ;;
    --log-dir) require_cli_value "$1" "${2-}"; LOG_DIR="$2"; shift 2 ;;
    --no-yes) ASSUME_YES=false; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --) shift; break ;;
    *) echo "ERROR: unknown option: $1" >&2; exit 2 ;;
  esac
done

require_bool "DRY_RUN" "$DRY_RUN"
require_bool "ASSUME_YES" "$ASSUME_YES"

# Derive the PINNED dimensions now that CLI overrides (if any) have been applied.
if [ -z "$MAX_SEQ_LEN" ]; then MAX_SEQ_LEN=$((INPUT_LEN + OUTPUT_LEN)); fi
if [ -z "$MAX_MODEL_LEN" ]; then MAX_MODEL_LEN="$MAX_SEQ_LEN"; fi
if [ -z "$FIXED_CHUNKED_PREFILL_SIZE" ]; then FIXED_CHUNKED_PREFILL_SIZE="$INPUT_LEN"; fi
if [ -z "$TRUE_MIXED_PREFILL_CHUNK_SIZES" ]; then TRUE_MIXED_PREFILL_CHUNK_SIZES="$INPUT_LEN"; fi

declare -a MODEL_ARGS
declare -a BLOCK_SIZE_ARGS
declare -a BATCH_SIZE_LIST_ARGS
declare -a DECODE_KV_CACHE_SIZE_LIST_ARGS
declare -a TRUE_MIXED_PREFILL_BATCH_SIZES_ARGS
declare -a TRUE_MIXED_PREFILL_CHUNK_SIZES_ARGS
declare -a TRUE_MIXED_DECODE_BATCH_SIZES_ARGS
declare -a TRUE_MIXED_DECODE_KV_CACHE_SIZES_ARGS

parse_token_list "MODELS" "$MODELS" MODEL_ARGS
parse_positive_integer_list "BLOCK_SIZES" "$BLOCK_SIZES" BLOCK_SIZE_ARGS
parse_positive_integer_list "BATCH_SIZE_LIST" "$BATCH_SIZE_LIST" BATCH_SIZE_LIST_ARGS
parse_positive_integer_list "DECODE_KV_CACHE_SIZE_LIST" "$DECODE_KV_CACHE_SIZE_LIST" DECODE_KV_CACHE_SIZE_LIST_ARGS
parse_positive_integer_list "TRUE_MIXED_PREFILL_BATCH_SIZES" "$TRUE_MIXED_PREFILL_BATCH_SIZES" TRUE_MIXED_PREFILL_BATCH_SIZES_ARGS
parse_positive_integer_list "TRUE_MIXED_PREFILL_CHUNK_SIZES" "$TRUE_MIXED_PREFILL_CHUNK_SIZES" TRUE_MIXED_PREFILL_CHUNK_SIZES_ARGS
parse_positive_integer_list "TRUE_MIXED_DECODE_BATCH_SIZES" "$TRUE_MIXED_DECODE_BATCH_SIZES" TRUE_MIXED_DECODE_BATCH_SIZES_ARGS
parse_positive_integer_list "TRUE_MIXED_DECODE_KV_CACHE_SIZES" "$TRUE_MIXED_DECODE_KV_CACHE_SIZES" TRUE_MIXED_DECODE_KV_CACHE_SIZES_ARGS

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: PYTHON_BIN is not executable or not on PATH: $PYTHON_BIN" >&2
  exit 2
fi

# Same validation frontier.profiling.attention.main itself does -- caught here
# first with a clearer message, since hitting it mid-sweep after several
# minutes of profiling on earlier (model, block_size) pairs is a worse way to
# find out. (This bit us for real during this work: --max_batch_size defaulted
# to 128 while a wider --batch_size_list was supplied.)
bs_list_max=$(list_max BATCH_SIZE_LIST_ARGS)
if [ "$bs_list_max" -gt "$MAX_BATCH_SIZE" ]; then
  echo "ERROR: BATCH_SIZE_LIST contains $bs_list_max, which is > MAX_BATCH_SIZE=$MAX_BATCH_SIZE. Raise --max-batch-size or narrow --batch-size-list." >&2
  exit 2
fi
tm_decode_bs_max=$(list_max TRUE_MIXED_DECODE_BATCH_SIZES_ARGS)
if [ "$tm_decode_bs_max" -gt "$MAX_BATCH_SIZE" ]; then
  echo "ERROR: TRUE_MIXED_DECODE_BATCH_SIZES contains $tm_decode_bs_max, which is > MAX_BATCH_SIZE=$MAX_BATCH_SIZE. Raise --max-batch-size or narrow --true-mixed-decode-batch-sizes." >&2
  exit 2
fi
kv_list_max=$(list_max DECODE_KV_CACHE_SIZE_LIST_ARGS)
if [ "$kv_list_max" -gt "$MAX_SEQ_LEN" ]; then
  echo "ERROR: DECODE_KV_CACHE_SIZE_LIST contains $kv_list_max, which is > MAX_SEQ_LEN=$MAX_SEQ_LEN. Raise --max-seq-len (and --input-len/--output-len if it should be derived) or narrow --decode-kv-cache-size-list." >&2
  exit 2
fi

mkdir -p "$LOG_DIR"

cat <<EOF
============================================================
  Profiling Recipe - True-Mixed-Batch Attention
============================================================
Models:                 ${MODEL_ARGS[*]}
Block sizes:             ${BLOCK_SIZE_ARGS[*]}
Device / TP / PP / GPUs: $DEVICE / TP=$TP / PP=$PP / $NUM_GPUS (ids: $GPU_IDS)
Attention backend:       $ATTENTION_BACKEND ($PROFILE_METHOD, $PRECISION)

PINNED (known from real workload, input_len=$INPUT_LEN output_len=$OUTPUT_LEN):
  max_seq_len=max_model_len=$MAX_SEQ_LEN  fixed_chunked_prefill_size=$FIXED_CHUNKED_PREFILL_SIZE
  true_mixed_prefill_batch_sizes=${TRUE_MIXED_PREFILL_BATCH_SIZES_ARGS[*]}  true_mixed_prefill_chunk_sizes=${TRUE_MIXED_PREFILL_CHUNK_SIZES_ARGS[*]}
  true_mixed_prefill_kv_cache_size=$TRUE_MIXED_PREFILL_KV_CACHE_SIZE

WIDENED (scheduling-dependent, swept not guessed):
  batch_size_list=${BATCH_SIZE_LIST_ARGS[*]}
  decode_kv_cache_size_list=${DECODE_KV_CACHE_SIZE_LIST_ARGS[*]}
  true_mixed_decode_batch_sizes=${TRUE_MIXED_DECODE_BATCH_SIZES_ARGS[*]}
  true_mixed_decode_kv_cache_sizes=${TRUE_MIXED_DECODE_KV_CACHE_SIZES_ARGS[*]}

Output taxonomy (per model, per block_size -- renamed immediately after each
run to avoid the confirmed output-overwrite bug):
  \$DATA_DIR_BASE/compute/\$DEVICE/<model>/attention_block<N>.csv
  \$DATA_DIR_BASE/compute/\$DEVICE/<model>/attention_true_mixed_block<N>.csv
  \$DATA_DIR_BASE/compute/\$DEVICE/<model>/attention_combined_block<N>.csv   <- what --atten-input-file should point at
Resolved output root: $DATA_DIR_BASE
Logs:                 $LOG_DIR
Dry run:              $DRY_RUN
============================================================
EOF

if [ "$DRY_RUN" != "true" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
  export HIP_VISIBLE_DEVICES="$GPU_IDS"   # ROCm's native equivalent, set for safety -- see INFRASTRUCTURE_MAP.md
fi

cd "$REPO_ROOT"

for MODEL in "${MODEL_ARGS[@]}"; do
  for BS in "${BLOCK_SIZE_ARGS[@]}"; do
    MODEL_DIR="$DATA_DIR_BASE/compute/$DEVICE/$MODEL"
    LOG_FILE="$LOG_DIR/profile_$(basename "$MODEL")_block${BS}.log"

    CMD=(
      "$PYTHON_BIN" -m frontier.profiling.attention.main
      --disable_ray --num_gpus "$NUM_GPUS" --device "$DEVICE"
      --models "$MODEL"
      --num_tensor_parallel_workers "$TP" --max_pipeline_parallel_size "$PP"
      --attention_backend "$ATTENTION_BACKEND" --profile_method "$PROFILE_METHOD" --precision "$PRECISION"
      --block_size "$BS"
      --max_seq_len "$MAX_SEQ_LEN" --max_model_len "$MAX_MODEL_LEN"
      --min_batch_size "$MIN_BATCH_SIZE" --max_batch_size "$MAX_BATCH_SIZE"
      --batch_size_list "${BATCH_SIZE_LIST_ARGS[@]}"
      --decode_kv_cache_size_list "${DECODE_KV_CACHE_SIZE_LIST_ARGS[@]}"
      --fixed_chunked_prefill_size "$FIXED_CHUNKED_PREFILL_SIZE"
      --enable_true_mixed
      --true_mixed_prefill_batch_sizes "${TRUE_MIXED_PREFILL_BATCH_SIZES_ARGS[@]}"
      --true_mixed_prefill_chunk_sizes "${TRUE_MIXED_PREFILL_CHUNK_SIZES_ARGS[@]}"
      --true_mixed_decode_batch_sizes "${TRUE_MIXED_DECODE_BATCH_SIZES_ARGS[@]}"
      --true_mixed_decode_kv_cache_sizes "${TRUE_MIXED_DECODE_KV_CACHE_SIZES_ARGS[@]}"
      --true_mixed_prefill_kv_cache_size "$TRUE_MIXED_PREFILL_KV_CACHE_SIZE"
      --output_dir "$DATA_DIR_BASE"
    )
    if [ "$ASSUME_YES" = "true" ]; then
      CMD+=(--yes)
    fi
    if [ "$#" -gt 0 ]; then
      CMD+=("$@")
    fi

    echo "------------------------------------------------------------"
    echo "Model=$MODEL block_size=$BS"
    printf 'Command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    echo "Log: $LOG_FILE"
    echo "Renames after run:"
    echo "  $MODEL_DIR/attention.csv            -> $MODEL_DIR/attention_block${BS}.csv"
    echo "  $MODEL_DIR/attention_true_mixed.csv -> $MODEL_DIR/attention_true_mixed_block${BS}.csv"
    echo "  $MODEL_DIR/attention_combined.csv   -> $MODEL_DIR/attention_combined_block${BS}.csv"

    if [ "$DRY_RUN" = "true" ]; then
      continue
    fi

    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"

    for suffix in "" "_true_mixed" "_combined"; do
      src="$MODEL_DIR/attention${suffix}.csv"
      dst="$MODEL_DIR/attention${suffix}_block${BS}.csv"
      if [ -f "$src" ]; then
        mv "$src" "$dst"
      else
        echo "WARNING: expected output not found, skipping rename: $src" >&2
      fi
    done
  done
done

if [ "$DRY_RUN" = "true" ]; then
  echo "Dry run completed; no profiling command was executed."
  exit 0
fi

echo "True-mixed-batch profiling sweep completed for: ${MODEL_ARGS[*]} (block sizes: ${BLOCK_SIZE_ARGS[*]})"
echo "Point run_validation.py / frontier_cli_translator.py's --atten-input-file at the"
echo "'_combined_block<N>.csv' file matching the block_size you're validating against."
