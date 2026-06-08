#!/bin/bash

# ============================================================
# Legacy internal script: PD+AF Disaggregation Profiling Script
# ============================================================
# This script collects profiling data for PD+AF disaggregation architecture.
# It is not part of the pre-release-v0.1 co-location release.
#
# Environment: active profiling Python
#
# Profiling modules by cluster type:
#   - Prefill Cluster: MoE + MLP + Attention
#   - Decode-Attn Cluster: Attention only
#   - Decode-FFN Cluster: MoE only
#
# Usage:
#   bash tests/test_pd_af_profiling.sh \
#     --model <model_name> \
#     --prefill-device <device> \
#     --decode-attn-device <device> \
#     --decode-ffn-device <device> \
#     [--max-tokens <num>] \
#     [--attn-tp-sizes "1 2"] \
#     [--moe-tp-sizes "1 2"] \
#     [--moe-ep-sizes "1 2"]
# ============================================================

set -e

# ============================================================
# CUDA Device Configuration
# ============================================================
# Set default visible GPU without overriding a user-provided selection.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "============================================================"
echo "PD+AF Disaggregation Profiling"
echo "============================================================"
echo ""

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Source device validation utility
source "$PROJECT_ROOT/tests/common/device_validation.sh"

# ============================================================
# Default Configuration
# ============================================================

MODEL="qwen2_moe_example"
MAX_TOKENS=1024

# Device configuration
PREFILL_DEVICE="a100"
DECODE_ATTN_DEVICE="h100"
DECODE_FFN_DEVICE="a100"

# Parallelism configuration
ATTN_TP_SIZES="1 2"
MOE_TP_SIZES="1 2"
MOE_EP_SIZES="1 2"

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

# ============================================================
# Parse Command Line Arguments
# ============================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --prefill-device)
            PREFILL_DEVICE="$2"
            shift 2
            ;;
        --decode-attn-device)
            DECODE_ATTN_DEVICE="$2"
            shift 2
            ;;
        --decode-ffn-device)
            DECODE_FFN_DEVICE="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --attn-tp-sizes)
            ATTN_TP_SIZES="$2"
            shift 2
            ;;
        --moe-tp-sizes)
            MOE_TP_SIZES="$2"
            shift 2
            ;;
        --moe-ep-sizes)
            MOE_EP_SIZES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --model <model> --prefill-device <device> --decode-attn-device <device> --decode-ffn-device <device> [options]"
            exit 1
            ;;
    esac
done

# ============================================================
# Display Configuration
# ============================================================

echo "Configuration:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model: $MODEL"
echo "  Max Tokens: $MAX_TOKENS"
echo ""
echo "  Cluster Devices:"
echo "    - Prefill: $PREFILL_DEVICE"
echo "    - Decode-Attn: $DECODE_ATTN_DEVICE"
echo "    - Decode-FFN: $DECODE_FFN_DEVICE"
echo ""
echo "  Parallelism Configurations:"
echo "    - Attention TP Sizes: $ATTN_TP_SIZES"
echo "    - MoE TP Sizes: $MOE_TP_SIZES"
echo "    - MoE EP Sizes: $MOE_EP_SIZES"
echo ""
echo "  Profiling Modules by Cluster:"
echo "    - Prefill: MoE + MLP + Attention"
echo "    - Decode-Attn: Attention only"
echo "    - Decode-FFN: MoE only"
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
echo "Validating Device Parameters"
echo "============================================================"
echo ""

echo "Validating Prefill device: $PREFILL_DEVICE"
validate_device "$PREFILL_DEVICE"
echo ""

echo "Validating Decode-Attn device: $DECODE_ATTN_DEVICE"
validate_device "$DECODE_ATTN_DEVICE"
echo ""

echo "Validating Decode-FFN device: $DECODE_FFN_DEVICE"
validate_device "$DECODE_FFN_DEVICE"
echo ""

# ============================================================
# Determine Unique Devices and Required Profiling Modules
# ============================================================

# Collect unique devices
UNIQUE_DEVICES=()
for device in "$PREFILL_DEVICE" "$DECODE_ATTN_DEVICE" "$DECODE_FFN_DEVICE"; do
    if [[ ! " ${UNIQUE_DEVICES[@]} " =~ " ${device} " ]]; then
        UNIQUE_DEVICES+=("$device")
    fi
done

echo "============================================================"
echo "Profiling Plan"
echo "============================================================"
echo ""
echo "Unique devices to profile: ${UNIQUE_DEVICES[@]}"
echo ""
echo "Profiling modules required:"
echo "  - MoE: Prefill ($PREFILL_DEVICE), Decode-FFN ($DECODE_FFN_DEVICE)"
echo "  - MLP: Prefill ($PREFILL_DEVICE)"
echo "  - Attention: Prefill ($PREFILL_DEVICE), Decode-Attn ($DECODE_ATTN_DEVICE)"
echo ""

# ============================================================
# Step 1: MoE Profiling
# ============================================================

echo "============================================================"
echo "Step 1: MoE Profiling"
echo "============================================================"
echo ""
echo "Profiling MoE operations for Prefill and Decode-FFN clusters"
echo ""

# Determine which devices need MoE profiling
MOE_DEVICES=()
for device in "$PREFILL_DEVICE" "$DECODE_FFN_DEVICE"; do
    if [[ ! " ${MOE_DEVICES[@]} " =~ " ${device} " ]]; then
        MOE_DEVICES+=("$device")
    fi
done

echo "Devices requiring MoE profiling: ${MOE_DEVICES[@]}"
echo ""

for DEVICE in "${MOE_DEVICES[@]}"; do
    echo "----------------------------------------"
    echo "MoE Profiling on $DEVICE"
    echo "----------------------------------------"
    echo ""
    
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    OUTPUT_DIR="$DATA_DIR_BASE"
    
    echo "Running MoE profiling..."
    echo "  Device: $DEVICE"
    echo "  Model: $MODEL"
    echo "  TP Sizes: $MOE_TP_SIZES"
    echo "  EP Sizes: $MOE_EP_SIZES"
    echo "  Output: $OUTPUT_DIR"
    echo ""
    
    cd "$PROJECT_ROOT" && PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" "$PYTHON_BIN" -m frontier.profiling.moe.main \
        --models $MODEL \
        --device $DEVICE \
        --num_gpus 2 \
        --max_tokens $MAX_TOKENS \
        --num_tensor_parallel_workers $MOE_TP_SIZES \
        --expert_parallel_sizes $MOE_EP_SIZES \
        --profile_method $PROFILE_METHOD \
        --output_dir $OUTPUT_DIR
    
    # Copy to data directory
    DATA_DIR="$DATA_DIR_BASE/compute/$DEVICE/$MODEL"
    mkdir -p "$DATA_DIR"
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
done

echo "✓ Step 1 completed: MoE profiling"
echo ""

# ============================================================
# Step 2: Linear Op Profiling (Prefill only)
# ============================================================

echo "============================================================"
echo "Step 2: Linear Op Profiling"
echo "============================================================"
echo ""
echo "Profiling linear operations for Prefill cluster only"
echo ""

MLP_DEVICE="$PREFILL_DEVICE"

echo "----------------------------------------"
echo "Linear op profiling on $MLP_DEVICE"
echo "----------------------------------------"
echo ""

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$DATA_DIR_BASE"

echo "Running linear_op profiling..."
echo "  Device: $MLP_DEVICE"
echo "  Model: $MODEL"
echo "  TP Sizes: $ATTN_TP_SIZES"
echo "  Output: $OUTPUT_DIR"
echo ""

cd "$PROJECT_ROOT" && PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" "$PYTHON_BIN" -m frontier.profiling.linear_op.main \
    --disable_ray \
    --models $MODEL \
    --num_gpus 2 \
    --max_tokens $MAX_TOKENS \
    --num_tensor_parallel_workers $ATTN_TP_SIZES \
    --profile_method $PROFILE_METHOD \
    --device $MLP_DEVICE \
    --output_dir $OUTPUT_DIR

# Copy to data directory
# Linear op profiling uses canonical compute output schema under the selected output root.
DATA_DIR="$DATA_DIR_BASE/compute/$MLP_DEVICE/$MODEL"
mkdir -p "$DATA_DIR"
# Find the most recent linear op profiling output
LINEAR_OP_CSV_NAME="$(profiled_csv_name linear_op)"
LINEAR_OP_CSV="$OUTPUT_DIR/compute/$MLP_DEVICE/$MODEL/$(profiled_csv_name linear_op)"
if [ ! -f "$LINEAR_OP_CSV" ]; then
    echo "❌ Error: Linear op profiling output not found in $OUTPUT_DIR"
    exit 1
fi
if [ "$LINEAR_OP_CSV" != "$DATA_DIR/$LINEAR_OP_CSV_NAME" ]; then
    cp "$LINEAR_OP_CSV" "$DATA_DIR/$LINEAR_OP_CSV_NAME"
fi

echo ""
echo "✓ Linear op profiling completed for $MLP_DEVICE"
echo "  Data saved to: $DATA_DIR/$LINEAR_OP_CSV_NAME"
echo ""

echo "✓ Step 2 completed: Linear op profiling"
echo ""

# ============================================================
# Step 3: Attention Profiling
# ============================================================

echo "============================================================"
echo "Step 3: Attention Profiling"
echo "============================================================"
echo ""
echo "Profiling Attention operations for Prefill and Decode-Attn clusters"
echo ""

# Determine which devices need Attention profiling
ATTN_DEVICES=()
for device in "$PREFILL_DEVICE" "$DECODE_ATTN_DEVICE"; do
    if [[ ! " ${ATTN_DEVICES[@]} " =~ " ${device} " ]]; then
        ATTN_DEVICES+=("$device")
    fi
done

echo "Devices requiring Attention profiling: ${ATTN_DEVICES[@]}"
echo ""

for DEVICE in "${ATTN_DEVICES[@]}"; do
    echo "----------------------------------------"
    echo "Attention Profiling on $DEVICE"
    echo "----------------------------------------"
    echo ""
    
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    OUTPUT_DIR="$DATA_DIR_BASE"
    
    echo "Running Attention profiling..."
    echo "  Device: $DEVICE"
    echo "  Model: $MODEL"
    echo "  TP Sizes: $ATTN_TP_SIZES"
    echo "  Output: $OUTPUT_DIR"
    echo ""

    cd "$PROJECT_ROOT" && PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" "$PYTHON_BIN" -m frontier.profiling.attention.main \
        --models $MODEL \
        --num_gpus 1 \
        --max_seq_len $MAX_TOKENS \
        --num_tensor_parallel_workers $ATTN_TP_SIZES \
        --device $DEVICE \
        --profile_method $PROFILE_METHOD \
        --output_dir $OUTPUT_DIR
    
    # Copy to data directory
    # Attention profiling uses canonical compute output schema under the selected output root.
    DATA_DIR="$DATA_DIR_BASE/compute/$DEVICE/$MODEL"
    mkdir -p "$DATA_DIR"
    # Find the most recent Attention profiling output
    ATTN_CSV_NAME="$(profiled_csv_name attention)"
    ATTN_CSV="$OUTPUT_DIR/compute/$DEVICE/$MODEL/$(profiled_csv_name attention)"
    if [ ! -f "$ATTN_CSV" ]; then
        echo "❌ Error: Attention profiling output not found in $OUTPUT_DIR"
        exit 1
    fi
    if [ "$ATTN_CSV" != "$DATA_DIR/$ATTN_CSV_NAME" ]; then
        cp "$ATTN_CSV" "$DATA_DIR/$ATTN_CSV_NAME"
    fi
    
    echo ""
    echo "✓ Attention profiling completed for $DEVICE"
    echo "  Data saved to: $DATA_DIR/$ATTN_CSV_NAME"
    echo ""
done

echo "✓ Step 3 completed: Attention profiling"
echo ""

# ============================================================
# Summary
# ============================================================

echo "============================================================"
echo "Profiling Summary"
echo "============================================================"
echo ""
echo "✓ All profiling completed successfully"
echo ""
echo "Profiling data organized in:"
echo "  $DATA_DIR_BASE/"
echo ""
echo "Data files created:"
echo ""

# List all created data files
for device in "${UNIQUE_DEVICES[@]}"; do
    echo "Device: $device"
    
    # Check MoE data
    if [ -f "$DATA_DIR_BASE/compute/$device/$MODEL/moe.csv" ]; then
        echo "  ✓ MoE: $DATA_DIR_BASE/compute/$device/$MODEL/moe.csv"
    fi
    
    # Check Linear op data
    if [ -f "$DATA_DIR_BASE/compute/$device/$MODEL/linear_op.csv" ]; then
        echo "  ✓ Linear op: $DATA_DIR_BASE/compute/$device/$MODEL/linear_op.csv"
    fi
    
    # Check Attention data
    if [ -f "$DATA_DIR_BASE/compute/$device/$MODEL/attention.csv" ]; then
        echo "  ✓ Attention: $DATA_DIR_BASE/compute/$device/$MODEL/attention.csv"
    fi
    
    echo ""
done

echo "Next step: Run training script"
echo "  This legacy PD-AF workflow is not part of the pre-release-v0.1 co-location release."
echo "  For the release workflow, use the co-location profiling and training examples in README.md."
echo ""
