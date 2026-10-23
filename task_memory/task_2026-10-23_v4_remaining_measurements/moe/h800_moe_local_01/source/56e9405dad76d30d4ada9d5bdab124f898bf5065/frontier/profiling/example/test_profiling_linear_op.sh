#!/bin/bash

# ============================================================
# Linear Operations Architecture Component Profiling Script
# ============================================================
# This script profiles linear operations (MLP, LayerNorm, etc.) for LLM models.
# It is independent of cluster deployment topology and focuses
# solely on linear-complexity architecture components.
#
# Environment: active profiling Python
#
# Usage:
#   bash frontier/profiling/example/test_profiling_linear_op.sh \
#     --model <model_name> \
#     --device <device_type> \
#     [--max-tokens <num>] \
#     [--tp-sizes "1 2 4"] \
#     [--num-gpus <num>] \
#     [--is-moe]
#     [--skip-dense-mlp]
#
# Note: --device accepts only a SINGLE device type (e.g., a100 OR h100).
#       To profile multiple devices, run the script multiple times.
#
# Note: --is-moe / --skip-dense-mlp skips dense MLP timing columns.
#       The default is dense linear collection so release CSVs include MLP timings.
# Measurement contract: record_function -> KERNEL_ONLY, cuda_event -> CUDA_EVENT
# ============================================================

set -e

echo "============================================================"
echo "Linear Operations Architecture Component Profiling"
echo "============================================================"
echo ""

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Source device validation utility
source "$PROJECT_ROOT/tests/common/device_validation.sh"

# Set default visible GPUs without overriding a user-provided selection.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

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
# mixtral_8x7b_moe
# Phi-tiny-MoE-instruct
MODEL="Phi-tiny-MoE-instruct"
MAX_TOKENS=4096
NUM_GPUS=2

# Device configuration - single device only
DEVICE="a800"

# Parallelism configuration
TP_SIZES="1"

# Environment
# Use Python from the active profiling environment. Override with PYTHON_BIN=/path/to/python.
PYTHON_BIN="${PYTHON_BIN:-python}"

# Output directory root passed to profiling CLIs.
DATA_DIR_BASE="${DATA_DIR_BASE:-$PROJECT_ROOT/data/profiling}"

# Linear operation-specific parameters
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
IS_MOE=""

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
        --max-tokens)
            MAX_TOKENS="$2"
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
        --profile-method)
            PROFILE_METHOD="$2"
            shift 2
            ;;
        --is-moe|--skip-dense-mlp)
            IS_MOE="--is_moe"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --model <model> --device <device_type> [options]"
            echo ""
            echo "Options:"
            echo "  --model <name>        Model name to profile"
            echo "  --device <type>       Device type (e.g., a100, h100) - SINGLE device only"
            echo "  --max-tokens <num>    Maximum tokens to profile (default: 1024)"
            echo "  --tp-sizes <sizes>    Tensor parallel sizes (default: \"1 2\")"
            echo "  --num-gpus <num>      Number of GPUs (default: 2)"
            echo "  --profile-method      Profiling method (default: cuda_event; cuda_event -> CUDA_EVENT, record_function -> KERNEL_ONLY)"
            echo "  --is-moe              Skip dense MLP timing columns"
            echo "  --skip-dense-mlp      Alias for --is-moe"
            echo ""
            echo "Note: --device accepts only a SINGLE device type."
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
echo "  Architecture Component: Linear Operations (MLP, LayerNorm, etc.)"
echo "  Model: $MODEL"
echo "  Max Tokens: $MAX_TOKENS"
echo "  Number of GPUs: $NUM_GPUS"
echo ""
echo "  Target Device: $DEVICE"
echo ""
echo "  Parallelism Configurations:"
echo "    - Tensor Parallel Sizes: $TP_SIZES"
echo ""
echo "  Linear Op-Specific Parameters:"
echo "    - Profile Method: $PROFILE_METHOD (record_function -> KERNEL_ONLY, cuda_event -> CUDA_EVENT)"
echo "    - Is MoE: ${IS_MOE:-false}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ============================================================
# Environment Check
# ============================================================

echo "============================================================"
echo "Checking Profiling Environment"
echo "============================================================"
echo ""

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "❌ ERROR: Profiling Python not found: $PYTHON_BIN"
    echo "Set PYTHON_BIN to the Python executable in environment_profiling.yml or an existing vLLM/FlashInfer environment."
    exit 1
fi

echo "✓ Profiling Python: $PYTHON_BIN"
echo ""

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
# Linear Operations Profiling
# ============================================================

echo "============================================================"
echo "Linear Operations Architecture Component Profiling"
echo "============================================================"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Profiling Linear Operations on Device: $DEVICE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$DATA_DIR_BASE"

echo "Running Linear Operations profiling..."
echo "  Device: $DEVICE"
echo "  Model: $MODEL"
echo "  TP Sizes: $TP_SIZES"
echo "  Max Tokens: $MAX_TOKENS"
    echo "  Profile Method: $PROFILE_METHOD (cuda_event -> CUDA_EVENT, record_function -> KERNEL_ONLY)"
echo "  Output: $OUTPUT_DIR"
echo ""

cd "$PROJECT_ROOT" && PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" "$PYTHON_BIN" -m frontier.profiling.linear_op.main \
    --disable_ray \
    --models $MODEL \
    --num_gpus $NUM_GPUS \
    --max_tokens $MAX_TOKENS \
    --num_tensor_parallel_workers $TP_SIZES \
    --profile_method $PROFILE_METHOD \
    --device $DEVICE \
    --output_dir $OUTPUT_DIR \
    $IS_MOE

# Copy to data directory
DATA_DIR="$DATA_DIR_BASE/compute/$DEVICE/$MODEL"
mkdir -p "$DATA_DIR"

# Find the most recent linear_op profiling output
LINEAR_OP_CSV_NAME="$(profiled_csv_name linear_op)"
LINEAR_OP_CSV="$OUTPUT_DIR/compute/$DEVICE/$MODEL/$(profiled_csv_name linear_op)"
if [ ! -f "$LINEAR_OP_CSV" ]; then
    if [ -n "$IS_MOE" ]; then
        echo ""
        echo "ℹ️  No linear_op.csv generated (expected with --is-moe flag)"
    else
        echo "❌ Error: Linear operation profiling output not found in $OUTPUT_DIR"
        exit 1
    fi
else
    if [ "$LINEAR_OP_CSV" != "$DATA_DIR/$LINEAR_OP_CSV_NAME" ]; then
        cp "$LINEAR_OP_CSV" "$DATA_DIR/$LINEAR_OP_CSV_NAME"
    fi
    echo ""
    echo "✓ Linear Operations profiling completed for $DEVICE"
    echo "  Data saved to: $DATA_DIR/$LINEAR_OP_CSV_NAME"
fi
echo ""

# ============================================================
# Summary
# ============================================================

echo "============================================================"
echo "Linear Operations Profiling Summary"
echo "============================================================"
echo ""
echo "✓ Linear Operations profiling completed successfully"
echo ""
echo "Device: $DEVICE"
echo "  ✓ Linear Ops: $DATA_DIR_BASE/compute/$DEVICE/$MODEL/$LINEAR_OP_CSV_NAME"
echo ""
echo "Next steps:"
echo "  - Run Attention profiling: bash frontier/profiling/example/test_profiling_attn.sh --device $DEVICE"
echo "  - Run MoE profiling: bash frontier/profiling/example/test_profiling_moe.sh --device $DEVICE"
echo "  - To profile another device, run this script again with --device <other_device>"
echo ""
