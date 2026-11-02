#!/bin/bash

# ============================================================
# Attention Architecture Component Profiling Script
# ============================================================
# This script profiles attention operations for LLM models.
# It is independent of cluster deployment topology and focuses
# solely on the attention architecture component.
#
# Environment: active profiling Python
#
# Usage:
#   bash frontier/profiling/example/test_profiling_attn.sh \
#     --model <model_name> \
#     --device <device_type> \
#     [--max-seq-len <length>] \
#     [--tp-sizes "1 2 4"] \
#     [--num-gpus <num>]
#     [--profile-method <method>] (default: cuda_event; record_function -> KERNEL_ONLY)
#     [--enable-true-mixed [true|false]]
#     [--dry-run]
#
# Note: --device accepts only a SINGLE device type (e.g., a100 OR h100).
#       To profile multiple devices, run the script multiple times.

set -e

echo "============================================================"
echo "Attention Architecture Component Profiling"
echo "============================================================"
echo ""

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Tee console output to a log file for later inspection
LOG_FILE="${FRONTIER_PROFILE_LOG_FILE:-$PROJECT_ROOT/frontier/profiling/attention/output.log}"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

# Source device validation utility
source "$PROJECT_ROOT/tests/common/device_validation.sh"

# Set default visible GPUs without overriding a user-provided selection.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"

# ============================================================
# CUDA Architecture Configuration
# ============================================================
# Respect a user-provided CUDA architecture list. Leave it unset by default so
# PyTorch/CUDA can target the active GPU architecture.
if [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    export TORCH_CUDA_ARCH_LIST
fi

# ============================================================
# Default Configuration
# ============================================================
# Llama-3.2-1B-Instruct
# Phi-tiny-MoE-instruct
MODEL="Phi-tiny-MoE-instruct"
MAX_SEQ_LEN=4096 # min_len=64
NUM_GPUS=3

# Device configuration - single device only
DEVICE="a800"

# Parallelism configuration
TP_SIZES="1"
PP_SIZES="1"

# Environment
# Use Python from the active profiling environment. Override with PYTHON_BIN=/path/to/python.
PYTHON_BIN="${PYTHON_BIN:-python}"

# Profiling method: cuda_event -> CUDA_EVENT, record_function -> KERNEL_ONLY
PROFILE_METHOD="cuda_event"

profiled_csv_name() {
    local op_name="$1"
    case "$PROFILE_METHOD" in
        cuda|cuda_event)
            printf "%s.csv\n" "$op_name"
            ;;
        kernel_only|record_function)
            printf "%s_kernel_only.csv\n" "$op_name"
            ;;
        *)
            echo "Unsupported PROFILE_METHOD=$PROFILE_METHOD. Expected cuda_event or record_function." >&2
            return 1
            ;;
    esac
}

# Output directory root passed to profiling CLIs.
DATA_DIR_BASE="${DATA_DIR_BASE:-$PROJECT_ROOT/data/profiling}"

# Attention-specific parameters
ATTENTION_BACKEND="FLASHINFER"
BLOCK_SIZE=16
MIN_BATCH_SIZE=1
MAX_BATCH_SIZE=16
PROFILE_ONLY_DECODE=false
PROFILE_ONLY_PREFILL=false

ENABLE_MIXED_PREFILL=false
MIXED_MODE="even"
ENABLE_CHUNKED_PREFILL_GRID_SEARCH=false
FIXED_CHUNKED_PREFILL_SIZE=-1

ENABLE_TRUE_MIXED=false
TRUE_MIXED_PREFILL_BATCH_SIZES="1 2 4"
TRUE_MIXED_PREFILL_CHUNK_SIZES="64 128 256 512 1024"
TRUE_MIXED_DECODE_BATCH_SIZES="1 2 4 8"
TRUE_MIXED_DECODE_KV_CACHE_SIZES="128 256 512 1024 2048"
TRUE_MIXED_PREFILL_KV_CACHE_SIZE=0
DRY_RUN=false

parse_optional_bool() {
    local value="${1:-}"
    case "$value" in
        true|True|TRUE)
            echo "true"
            return 0
            ;;
        false|False|FALSE)
            echo "false"
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# ============================================================
# Parse Command Line Arguments
# ============================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --max-seq-len)
            MAX_SEQ_LEN="$2"
            shift 2
            ;;
        --tp-sizes)
            TP_SIZES="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --attention-backend)
            ATTENTION_BACKEND="$2"
            shift 2
            ;;
        --block-size)
            BLOCK_SIZE="$2"
            shift 2
            ;;
        --min-batch-size)
            MIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --max-batch-size)
            MAX_BATCH_SIZE="$2"
            shift 2
            ;;
        --profile-only-decode)
            PROFILE_ONLY_DECODE=true
            shift
            ;;
        --profile-only-prefill)
            PROFILE_ONLY_PREFILL=true
            shift
            ;;
        --enable-mixed-prefill)
            if parsed_bool=$(parse_optional_bool "${2:-}"); then
                ENABLE_MIXED_PREFILL="$parsed_bool"
                shift 2
            else
                ENABLE_MIXED_PREFILL=true
                shift
            fi
            ;;
        --enable-chunked-prefill-grid-search)
            if parsed_bool=$(parse_optional_bool "${2:-}"); then
                ENABLE_CHUNKED_PREFILL_GRID_SEARCH="$parsed_bool"
                shift 2
            else
                ENABLE_CHUNKED_PREFILL_GRID_SEARCH=true
                shift
            fi
            ;;
        --fixed-chunked-prefill-size)
            FIXED_CHUNKED_PREFILL_SIZE="$2"
            shift 2
            ;;
        --mixed-mode|--mixed_mode)
            MIXED_MODE="$2"
            shift 2
            ;;
        --enable-true-mixed)
            if parsed_bool=$(parse_optional_bool "${2:-}"); then
                ENABLE_TRUE_MIXED="$parsed_bool"
                shift 2
            else
                ENABLE_TRUE_MIXED=true
                shift
            fi
            ;;
        --profile-method|--profile_method)
            PROFILE_METHOD="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --true-mixed-prefill-batch-sizes)
            TRUE_MIXED_PREFILL_BATCH_SIZES="$2"
            shift 2
            ;;
        --true-mixed-prefill-chunk-sizes)
            TRUE_MIXED_PREFILL_CHUNK_SIZES="$2"
            shift 2
            ;;
        --true-mixed-decode-batch-sizes)
            TRUE_MIXED_DECODE_BATCH_SIZES="$2"
            shift 2
            ;;
        --true-mixed-decode-kv-cache-sizes)
            TRUE_MIXED_DECODE_KV_CACHE_SIZES="$2"
            shift 2
            ;;
        --true-mixed-prefill-kv-cache-size)
            TRUE_MIXED_PREFILL_KV_CACHE_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --model <model> --device <device_type> [options]"
            echo ""
            echo "Options:"
            echo "  --profile-method <method> Profiling method (default: cuda_event; cuda_event -> CUDA_EVENT, record_function -> KERNEL_ONLY)"
            echo "  --enable-mixed-prefill [bool] Enable mixed prefill (default: false)"
            echo "  --enable-true-mixed [bool] Enable true mixed prefill/decode (default: false)"
            echo "  --enable-chunked-prefill-grid-search [bool] Enable chunked-prefill grid search (default: false)"
            echo ""
            echo "Note: --device accepts only a SINGLE device type (e.g., a100 OR h100)."
            echo "      To profile multiple devices, run the script multiple times."
            exit 1
            ;;
    esac
done

# ============================================================
# Display Configuration
# ============================================================

echo "Configuration:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Architecture Component: Attention"
echo "  Model: $MODEL"
echo "  Max Sequence Length: $MAX_SEQ_LEN"
echo "  Number of GPUs: $NUM_GPUS"
echo ""
echo "  Target Device: $DEVICE"
echo ""
echo "  Parallelism Configurations:"
echo "    - Tensor Parallel Sizes: $TP_SIZES"
echo ""
echo "  Attention-Specific Parameters:"
echo "    - Backend: $ATTENTION_BACKEND"
echo "    - Profile Method: $PROFILE_METHOD (record_function -> KERNEL_ONLY, cuda_event -> CUDA_EVENT)"
echo "    - Block Size: $BLOCK_SIZE"
echo "    - Batch Size Range: [$MIN_BATCH_SIZE, $MAX_BATCH_SIZE]"
echo "    - Chunked Prefill Grid Search: $ENABLE_CHUNKED_PREFILL_GRID_SEARCH"
echo "    - Fixed Chunked Prefill Size: $FIXED_CHUNKED_PREFILL_SIZE"
if [ "$PROFILE_ONLY_DECODE" = true ]; then
    echo "    - Mode: Decode Only"
elif [ "$PROFILE_ONLY_PREFILL" = true ]; then
    echo "    - Mode: Prefill Only"
else
    echo "    - Mode: Both Prefill and Decode"
fi
if [ "$ENABLE_MIXED_PREFILL" = true ]; then
    echo "    - Mixed-Length Batch Prefill: Enabled"
    echo "    - Mixed-Length Batch Mode: $MIXED_MODE"
else
    echo "    - Mixed-Length Batch Prefill: Disabled"
fi
if [ "$ENABLE_TRUE_MIXED" = true ]; then
    echo "    - True Mixed Batch (Prefill+Decode): Enabled"
    echo "    - True Mixed Prefill Batch Sizes: $TRUE_MIXED_PREFILL_BATCH_SIZES"
    echo "    - True Mixed Prefill Chunk Sizes: $TRUE_MIXED_PREFILL_CHUNK_SIZES"
    echo "    - True Mixed Decode Batch Sizes: $TRUE_MIXED_DECODE_BATCH_SIZES"
    echo "    - True Mixed Decode KV Cache Sizes: $TRUE_MIXED_DECODE_KV_CACHE_SIZES"
    echo "    - True Mixed Prefill KV Cache Size: $TRUE_MIXED_PREFILL_KV_CACHE_SIZE"
else
    echo "    - True Mixed Batch (Prefill+Decode): Disabled"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "Dry run completed; no profiling command was executed."
    exit 0
fi


# Print GPU information
print_gpu_info

# ============================================================
# Device Validation
# ============================================================

echo "============================================================"
echo "Validating Device Parameter"
echo "============================================================"
echo ""

# Validate single device
echo "Validating device: $DEVICE"
validate_device "$DEVICE"
echo ""

# Check for space in DEVICE value (indicates user tried to pass multiple devices)
if [[ "$DEVICE" == *" "* ]]; then
    echo "❌ ERROR: --device parameter accepts only a SINGLE device type."
    echo ""
    echo "You provided: $DEVICE"
    echo ""
    echo "Correct usage:"
    echo "  --device a100    (correct)"
    echo "  --device h100    (correct)"
    echo ""
    echo "Incorrect usage:"
    echo "  --device \"a100 h100\"    (WRONG - multiple devices not allowed)"
    echo ""
    echo "To profile multiple devices, run the script multiple times:"
    echo "  bash $0 --device a100 --model $MODEL"
    echo "  bash $0 --device h100 --model $MODEL"
    echo ""
    exit 1
fi

# ============================================================
# Attention Profiling
# ============================================================

echo "============================================================"
echo "Attention Architecture Component Profiling"
echo "============================================================"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Profiling Attention on Device: $DEVICE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$DATA_DIR_BASE"

echo "Running Attention profiling..."
echo "  Device: $DEVICE"
echo "  Model: $MODEL"
echo "  TP Sizes: $TP_SIZES"
echo "  PP Sizes: $PP_SIZES"
echo "  Max Sequence Length: $MAX_SEQ_LEN"
echo "  Attention Backend: $ATTENTION_BACKEND"
echo "  Output: $OUTPUT_DIR"
echo ""

# Set FlashInfer cache directory to avoid disk quota issues
# FlashInfer uses FLASHINFER_WORKSPACE_BASE to determine cache location
# Cache will be stored at: $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/
FLASHINFER_CACHE_DIR="$PROJECT_ROOT/profiling"
mkdir -p "$FLASHINFER_CACHE_DIR/.cache/flashinfer"

# Build command with optional flags
# Note: Using --disable_ray for multiprocessing mode (more stable than Ray)
# Set FLASHINFER_WORKSPACE_BASE to redirect FlashInfer JIT cache
CMD="cd \"$PROJECT_ROOT\" && FLASHINFER_WORKSPACE_BASE=\"$FLASHINFER_CACHE_DIR\" PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\" "$PYTHON_BIN" -m frontier.profiling.attention.main \
    --disable_ray \
    --models $MODEL \
    --num_gpus $NUM_GPUS \
    --max_seq_len $MAX_SEQ_LEN \
    --num_tensor_parallel_workers $TP_SIZES \
    --max_pipeline_parallel_size $PP_SIZES \
    --attention_backend $ATTENTION_BACKEND \
    --block_size $BLOCK_SIZE \
    --min_batch_size $MIN_BATCH_SIZE \
    --max_batch_size $MAX_BATCH_SIZE \
    --fixed_chunked_prefill_size $FIXED_CHUNKED_PREFILL_SIZE \
    --device $DEVICE \
    --profile_method $PROFILE_METHOD \
    --output_dir $OUTPUT_DIR"

if [ "$PROFILE_ONLY_DECODE" = true ]; then
    CMD="$CMD --profile_only_decode"
fi

if [ "$PROFILE_ONLY_PREFILL" = true ]; then
    CMD="$CMD --profile_only_prefill"
fi

if [ "$ENABLE_MIXED_PREFILL" = true ]; then
    CMD="$CMD --enable_mixed_prefill"
fi

if [ "$ENABLE_CHUNKED_PREFILL_GRID_SEARCH" = true ]; then
    CMD="$CMD --enable_chunked_prefill_grid_search"
fi

if [ "$MIXED_MODE" != "both" ]; then
    CMD="$CMD --mixed_mode $MIXED_MODE"
fi

if [ "$ENABLE_TRUE_MIXED" = true ]; then
    CMD="$CMD --enable_true_mixed"
    CMD="$CMD --true_mixed_prefill_batch_sizes $TRUE_MIXED_PREFILL_BATCH_SIZES"
    CMD="$CMD --true_mixed_prefill_chunk_sizes $TRUE_MIXED_PREFILL_CHUNK_SIZES"
    CMD="$CMD --true_mixed_decode_batch_sizes $TRUE_MIXED_DECODE_BATCH_SIZES"
    CMD="$CMD --true_mixed_decode_kv_cache_sizes $TRUE_MIXED_DECODE_KV_CACHE_SIZES"
    CMD="$CMD --true_mixed_prefill_kv_cache_size $TRUE_MIXED_PREFILL_KV_CACHE_SIZE"
fi

eval $CMD

# Copy to data directory
DATA_DIR="$DATA_DIR_BASE/compute/$DEVICE/$MODEL"
mkdir -p "$DATA_DIR"

# Find the most recent Attention profiling output
ATTN_COMBINED_CSV_NAME="$(profiled_csv_name attention_combined)"
ATTN_CSV_NAME="$(profiled_csv_name attention)"
ATTN_MIXED_CSV_NAME="$(profiled_csv_name attention_mixed)"
ATTN_TRUE_MIXED_CSV_NAME="$(profiled_csv_name attention_true_mixed)"
ATTN_COMBINED_CSV="$OUTPUT_DIR/compute/$DEVICE/$MODEL/$(profiled_csv_name attention_combined)"
ATTN_CSV="$OUTPUT_DIR/compute/$DEVICE/$MODEL/$(profiled_csv_name attention)"
ATTN_MIXED_CSV="$OUTPUT_DIR/compute/$DEVICE/$MODEL/$(profiled_csv_name attention_mixed)"
ATTN_TRUE_MIXED_CSV="$OUTPUT_DIR/compute/$DEVICE/$MODEL/$(profiled_csv_name attention_true_mixed)"

FOUND_ATTENTION_OUTPUT=false

if [ -f "$ATTN_CSV" ]; then
    echo "Found standard attention CSV: $ATTN_CSV"
    if [ "$ATTN_CSV" != "$DATA_DIR/$ATTN_CSV_NAME" ]; then
        cp "$ATTN_CSV" "$DATA_DIR/$ATTN_CSV_NAME"
    fi
    FOUND_ATTENTION_OUTPUT=true
fi

if [ -f "$ATTN_COMBINED_CSV" ]; then
    echo "Found combined attention CSV: $ATTN_COMBINED_CSV"
    if [ "$ATTN_COMBINED_CSV" != "$DATA_DIR/$ATTN_COMBINED_CSV_NAME" ]; then
        cp "$ATTN_COMBINED_CSV" "$DATA_DIR/$ATTN_COMBINED_CSV_NAME"
    fi
    FOUND_ATTENTION_OUTPUT=true
fi

if [ "$FOUND_ATTENTION_OUTPUT" = false ]; then
    echo "❌ Error: Attention profiling output not found in $OUTPUT_DIR"
    exit 1
fi

if [ ! -f "$ATTN_CSV" ]; then
    echo "❌ Error: Standard attention profiling output not found: $ATTN_CSV"
    echo "The simulator default consumes $DATA_DIR/$ATTN_CSV_NAME, so release-facing profiling must produce it."
    exit 1
fi

if [ "$ENABLE_MIXED_PREFILL" = true ] && [ ! -f "$ATTN_MIXED_CSV" ]; then
    echo "❌ Error: Mixed-prefill attention profiling was requested, but output was not found: $ATTN_MIXED_CSV"
    exit 1
fi

if [ -f "$ATTN_MIXED_CSV" ]; then
    echo "Found mixed prefill attention CSV: $ATTN_MIXED_CSV"
    if [ "$ATTN_MIXED_CSV" != "$DATA_DIR/$ATTN_MIXED_CSV_NAME" ]; then
        cp "$ATTN_MIXED_CSV" "$DATA_DIR/$ATTN_MIXED_CSV_NAME"
    fi
fi

if [ "$ENABLE_TRUE_MIXED" = true ] && [ ! -f "$ATTN_TRUE_MIXED_CSV" ]; then
    echo "❌ Error: True-mixed attention profiling was requested, but output was not found: $ATTN_TRUE_MIXED_CSV"
    exit 1
fi

if [ -f "$ATTN_TRUE_MIXED_CSV" ]; then
    echo "Found true mixed attention CSV: $ATTN_TRUE_MIXED_CSV"
    if [ "$ATTN_TRUE_MIXED_CSV" != "$DATA_DIR/$ATTN_TRUE_MIXED_CSV_NAME" ]; then
        cp "$ATTN_TRUE_MIXED_CSV" "$DATA_DIR/$ATTN_TRUE_MIXED_CSV_NAME"
    fi
fi

echo ""
echo "✓ Attention profiling completed for $DEVICE"
if [ -f "$ATTN_CSV" ]; then
    echo "  Standard data saved to: $DATA_DIR/$ATTN_CSV_NAME"
fi
if [ -f "$ATTN_COMBINED_CSV" ]; then
    echo "  Combined data saved to: $DATA_DIR/$ATTN_COMBINED_CSV_NAME"
fi
if [ -f "$ATTN_MIXED_CSV" ]; then
    echo "  Mixed prefill data saved to: $DATA_DIR/$ATTN_MIXED_CSV_NAME"
fi
if [ -f "$ATTN_TRUE_MIXED_CSV" ]; then
    echo "  True mixed data saved to: $DATA_DIR/$ATTN_TRUE_MIXED_CSV_NAME"
fi
echo ""

# ============================================================
# Summary
# ============================================================

echo "============================================================"
echo "Attention Profiling Summary"
echo "============================================================"
echo ""
echo "✓ Attention profiling completed successfully"
echo ""
echo "Device: $DEVICE"
echo "  ✓ Attention: $DATA_DIR_BASE/compute/$DEVICE/$MODEL/attention.csv"
if [ -f "$ATTN_MIXED_CSV" ]; then
    echo "  ✓ Mixed Prefill Attention: $DATA_DIR_BASE/compute/$DEVICE/$MODEL/attention_mixed.csv"
fi
if [ -f "$ATTN_TRUE_MIXED_CSV" ]; then
    echo "  ✓ True Mixed Attention: $DATA_DIR_BASE/compute/$DEVICE/$MODEL/attention_true_mixed.csv"
fi
echo ""
echo "Next steps:"
echo "  - Run linear_op profiling: bash frontier/profiling/example/test_profiling_linear_op.sh --device $DEVICE"
echo "  - Run MoE profiling: bash frontier/profiling/example/test_profiling_moe.sh --device $DEVICE"
echo "  - To profile another device, run this script again with --device <other_device>"
echo ""
