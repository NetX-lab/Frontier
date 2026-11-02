#!/bin/bash

# ============================================================
# MoE Architecture Component Profiling Script
# ============================================================
# This script profiles Mixture-of-Experts (MoE) operations for LLM models.
# It is independent of cluster deployment topology and focuses
# solely on the MoE architecture component.
#
# Environment: active profiling Python
#
# Usage:
#   bash frontier/profiling/example/test_profiling_moe.sh \
#     --model <model_name> \
#     --device <device_type> \
#     [--max-tokens <num>] \
#     [--tp-sizes "1 2 4"] \
#     [--ep-sizes "1 2 4"] \
#     [--num-gpus <num>]
#     [--routing-runtime-path standard_fused_topk|uniform_topk]
#     [--gating-runtime-context standalone_legacy|prefill_hot]
#     [--dry-run]
#
# Note: --device accepts only a SINGLE device type (e.g., a100 OR h100).
#       To profile multiple devices, run the script multiple times.
# Measurement contract: record_function -> KERNEL_ONLY, cuda_event -> CUDA_EVENT
# ============================================================

set -e

echo "============================================================"
echo "MoE Architecture Component Profiling"
echo "============================================================"
echo ""

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Source device validation utility
source "$PROJECT_ROOT/tests/common/device_validation.sh"

# Set default visible GPUs without overriding a user-provided selection.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

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

MODEL="Phi-tiny-MoE-instruct"
MAX_TOKENS=4096
NUM_GPUS=4

# Device configuration - single device only
DEVICE="a800"

# Parallelism configuration
TP_SIZES="1"
EP_SIZES="1 2"

# Environment
# Use Python from the active profiling environment. Override with PYTHON_BIN=/path/to/python.
PYTHON_BIN="${PYTHON_BIN:-python}"

# Output directory root passed to profiling CLIs.
DATA_DIR_BASE="${DATA_DIR_BASE:-$PROJECT_ROOT/data/profiling}"

# MoE-specific parameters
PROFILE_METHOD="cuda_event"  # default: cuda_event
ROUTING_RUNTIME_PATH="standard_fused_topk"
GATING_RUNTIME_CONTEXT="standalone_legacy"

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

# Load imbalance configuration
# Set to "true" to enable load imbalance profiling, "false" to disable
LOAD_IMBALANCE="false"
# LOAD_DISTRIBUTIONS="uniform skewed extremely_skewed"
LOAD_DISTRIBUTIONS="uniform" 
NUM_SAMPLES_PER_DISTRIBUTION=3
DRY_RUN=false

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
        --ep-sizes)
            EP_SIZES="$2"
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
        --routing_runtime_path|--routing-runtime-path)
            ROUTING_RUNTIME_PATH="$2"
            shift 2
            ;;
        --gating_runtime_context|--gating-runtime-context)
            GATING_RUNTIME_CONTEXT="$2"
            shift 2
            ;;
        --load-imbalance)
            LOAD_IMBALANCE="$2"
            shift 2
            ;;
        --load-distributions)
            LOAD_DISTRIBUTIONS="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES_PER_DISTRIBUTION="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --model <model> --device <device_type> [options]"
            echo ""
            echo "Options:"
            echo "  --model <model>           Model name (default: mixtral_8x7b_moe)"
            echo "  --device <device_type>    Device type (e.g., a100, h100)"
            echo "  --max-tokens <num>        Max tokens (default: 1024)"
            echo "  --tp-sizes \"1 2 4\"        Tensor parallel sizes"
            echo "  --ep-sizes \"1 2 4\"        Expert parallel sizes"
            echo "  --num-gpus <num>          Number of GPUs (default: 8)"
            echo "  --profile-method <method> Profile method (default: cuda_event; cuda_event -> CUDA_EVENT, record_function -> KERNEL_ONLY)"
            echo "  --routing-runtime-path <path> Routing runtime path: standard_fused_topk or uniform_topk"
            echo "  --gating-runtime-context <ctx> Gating runtime context: standalone_legacy or prefill_hot"
            echo "  --load-imbalance <bool>   Enable load imbalance (true/false, default: false)"
            echo '  --load-distributions <d>  Load distributions (default: "uniform")'
            echo "  --num-samples <num>       Samples per distribution (default: 3)"
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
echo "  Architecture Component: Mixture-of-Experts (MoE)"
echo "  Model: $MODEL"
echo "  Max Tokens: $MAX_TOKENS"
echo "  Number of GPUs: $NUM_GPUS"
echo ""
echo "  Target Device: $DEVICE"
echo ""
echo "  Parallelism Configurations:"
echo "    - Tensor Parallel Sizes: $TP_SIZES"
echo "    - Expert Parallel Sizes: $EP_SIZES"
echo ""
echo "  MoE-Specific Parameters:"
echo "    - Profile Method: $PROFILE_METHOD (record_function -> KERNEL_ONLY, cuda_event -> CUDA_EVENT)"
echo "    - Routing Runtime Path: $ROUTING_RUNTIME_PATH"
echo "    - Gating Runtime Context: $GATING_RUNTIME_CONTEXT"
echo "    - Enable Load Imbalance: $LOAD_IMBALANCE"
if [[ "$LOAD_IMBALANCE" == "true" || "$LOAD_IMBALANCE" == "True" || "$LOAD_IMBALANCE" == "TRUE" ]]; then
    echo "    - Load Distributions: $LOAD_DISTRIBUTIONS"
    echo "    - Samples per Distribution: $NUM_SAMPLES_PER_DISTRIBUTION"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "Dry run completed; no profiling command was executed."
    exit 0
fi

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
# MoE Profiling
# ============================================================

echo "============================================================"
echo "MoE Architecture Component Profiling"
echo "============================================================"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Profiling MoE on Device: $DEVICE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$DATA_DIR_BASE"

echo "Running MoE profiling..."
echo "  Device: $DEVICE"
echo "  Model: $MODEL"
echo "  TP Sizes: $TP_SIZES"
echo "  EP Sizes: $EP_SIZES"
echo "  Max Tokens: $MAX_TOKENS"
echo "  Profile Method: $PROFILE_METHOD (record_function -> KERNEL_ONLY, cuda_event -> CUDA_EVENT)"
echo "  Execution Mode: Non-Ray (multiprocessing)"
echo "  Enable Load Imbalance: $LOAD_IMBALANCE"
if [[ "$LOAD_IMBALANCE" == "true" || "$LOAD_IMBALANCE" == "True" || "$LOAD_IMBALANCE" == "TRUE" ]]; then
    echo "  Load Distributions: $LOAD_DISTRIBUTIONS"
    echo "  Samples per Distribution: $NUM_SAMPLES_PER_DISTRIBUTION"
fi
echo "  Output: $OUTPUT_DIR"
echo ""

# NOTE: --disable_ray is REQUIRED due to Ray 2.52.1 + grpcio 1.67.1 incompatibility.
# Ray mode causes raylet crashes with "Trying to connect an http1.x server" error.
# The non-Ray mode uses torch.multiprocessing for multi-GPU support instead.
# See CHANGELOG_MULTI_GPU.md for details.

# Build command based on LOAD_IMBALANCE setting
if [[ "$LOAD_IMBALANCE" == "true" || "$LOAD_IMBALANCE" == "True" || "$LOAD_IMBALANCE" == "TRUE" ]]; then
    # With load imbalance profiling
    cd "$PROJECT_ROOT" && PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" "$PYTHON_BIN" -m frontier.profiling.moe.main \
        --models $MODEL \
        --device $DEVICE \
        --num_gpus $NUM_GPUS \
        --max_tokens $MAX_TOKENS \
        --num_tensor_parallel_workers $TP_SIZES \
        --expert_parallel_sizes $EP_SIZES \
        --profile_method $PROFILE_METHOD \
        --routing_runtime_path $ROUTING_RUNTIME_PATH \
        --gating_runtime_context $GATING_RUNTIME_CONTEXT \
        --enable_load_imbalance \
        --load_distributions $LOAD_DISTRIBUTIONS \
        --num_samples_per_distribution $NUM_SAMPLES_PER_DISTRIBUTION \
        --disable_ray \
        --output_dir $OUTPUT_DIR
else
    # Without load imbalance profiling (standard mode)
    cd "$PROJECT_ROOT" && PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" "$PYTHON_BIN" -m frontier.profiling.moe.main \
        --models $MODEL \
        --device $DEVICE \
        --num_gpus $NUM_GPUS \
        --max_tokens $MAX_TOKENS \
        --num_tensor_parallel_workers $TP_SIZES \
        --expert_parallel_sizes $EP_SIZES \
        --profile_method $PROFILE_METHOD \
        --routing_runtime_path $ROUTING_RUNTIME_PATH \
        --gating_runtime_context $GATING_RUNTIME_CONTEXT \
        --disable_load_imbalance \
        --disable_ray \
        --output_dir $OUTPUT_DIR
fi


# Copy to data directory
DATA_DIR="$DATA_DIR_BASE/compute/$DEVICE/$MODEL"
mkdir -p "$DATA_DIR"

# Find the most recent MoE profiling output
MOE_CSV_NAME="$(profiled_csv_name moe)"
MOE_CSV="$OUTPUT_DIR/compute/$DEVICE/$MODEL/$(profiled_csv_name moe)"
if [ ! -f "$MOE_CSV" ]; then
    echo "❌ Error: MoE profiling output not found in $OUTPUT_DIR"
    exit 1
fi
if [ "$MOE_CSV" != "$DATA_DIR/$MOE_CSV_NAME" ]; then
    cp "$MOE_CSV" "$DATA_DIR/$MOE_CSV_NAME"
fi

echo ""
echo "✓ MoE profiling completed for $DEVICE"
echo "  Data saved to: $DATA_DIR/$MOE_CSV_NAME"
echo ""

# ============================================================
# Summary
# ============================================================

echo "============================================================"
echo "MoE Profiling Summary"
echo "============================================================"
echo ""
echo "✓ MoE profiling completed successfully"
echo ""
echo "Device: $DEVICE"
echo "  ✓ MoE: $DATA_DIR_BASE/compute/$DEVICE/$MODEL/$MOE_CSV_NAME"
echo ""
echo "Next steps:"
echo "  - Run Attention profiling: bash frontier/profiling/example/test_profiling_attn.sh --device $DEVICE"
echo "  - Run linear_op profiling: bash frontier/profiling/example/test_profiling_linear_op.sh --device $DEVICE"
echo "  - To profile another device, run this script again with --device <other_device>"
echo ""
