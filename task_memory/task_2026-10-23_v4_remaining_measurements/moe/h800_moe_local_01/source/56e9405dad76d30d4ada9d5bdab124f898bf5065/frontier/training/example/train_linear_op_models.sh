#!/bin/bash

# ============================================================
# Linear Operations Model Training Script
# ============================================================
# This script trains execution time prediction models for linear operations
# (MLP, LayerNorm, etc.) using profiling data.
#
# Usage:
#   bash frontier/training/example/train_linear_op_models.sh \
#     --model <model_name> \
#     --device <device_type> \
#     [--tp-size <num>] \
#     [--predictor-type <type>] \
#     [--is-moe]
#
# Note: --is-moe flag skips training (for MoE models that use expert layers)
# ============================================================

set -e

# Use Python from the active simulator/training environment. Override with PYTHON_BIN=/path/to/python.
PYTHON_BIN="${PYTHON_BIN:-python}"


echo "============================================================"
echo "Linear Operations Model Training"
echo "============================================================"
echo ""

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ============================================================
# Default Configuration
# ============================================================

MODEL="mixtral_8x7b_moe"
DEVICE="a800"
TP_SIZE=2
PREDICTOR_TYPE="random_forest"
MEASUREMENT_TYPE="CUDA_EVENT"
IS_MOE=""

# Directories
DATA_DIR_BASE="$PROJECT_ROOT/data/profiling"
OUTPUT_DIR="cache"

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
        --tp-size)
            TP_SIZE="$2"
            shift 2
            ;;
        --predictor-type)
            PREDICTOR_TYPE="$2"
            shift 2
            ;;
        --measurement-type|--measurement_type)
            MEASUREMENT_TYPE="$2"
            shift 2
            ;;
        --is-moe)
            IS_MOE="--is_moe"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --model <model> --device <device> [options]"
            echo ""
            echo "Options:"
            echo "  --model <name>          Model name (default: mixtral_8x7b_moe)"
            echo "  --device <type>         Device type (default: a800)"
            echo "  --tp-size <num>         Tensor parallel size (default: 2)"
            echo "  --predictor-type <type> Predictor type (default: random_forest)"
            echo "  --measurement-type <t>  Measurement type: CUDA_EVENT or KERNEL_ONLY (default: CUDA_EVENT)"
            echo "  --is-moe                Skip training (for MoE models)"
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
echo "  Device: $DEVICE"
echo "  Tensor Parallel Size: $TP_SIZE"
echo "  Predictor Type: $PREDICTOR_TYPE"
echo "  Measurement Type: $MEASUREMENT_TYPE"
echo "  Is MoE: ${IS_MOE:-false}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ============================================================
# Validate Input Data
# ============================================================

case "$MEASUREMENT_TYPE" in
    CUDA_EVENT)
        DATASET_FILENAME="linear_op.csv"
        ;;
    KERNEL_ONLY)
        DATASET_FILENAME="linear_op_kernel_only.csv"
        ;;
    *)
        echo "❌ ERROR: Unsupported measurement type: $MEASUREMENT_TYPE"
        echo "  Expected: CUDA_EVENT or KERNEL_ONLY"
        exit 1
        ;;
esac

DATASET_PATH="$DATA_DIR_BASE/compute/$DEVICE/$MODEL/$DATASET_FILENAME"

if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ ERROR: Linear operation profiling data not found"
    echo "  Expected: $DATASET_PATH"
    echo ""
    echo "Please run profiling first:"
    echo "  bash frontier/profiling/example/test_profiling_linear_op.sh --model $MODEL --device $DEVICE --profile-method ${MEASUREMENT_TYPE,,}"
    echo ""
    exit 1
fi

echo "✓ Found profiling data: $DATASET_PATH"
echo ""

# ============================================================
# Train Linear Operation Models
# ============================================================

echo "============================================================"
echo "Training Linear Operation Models"
echo "============================================================"
echo ""

cd "$PROJECT_ROOT" && PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" "$PYTHON_BIN" -m frontier.training.cli linear_op \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL" \
    --device "$DEVICE" \
    --tensor_parallel_size $TP_SIZE \
    --predictor_type "$PREDICTOR_TYPE" \
    --measurement_type "$MEASUREMENT_TYPE" \
    $IS_MOE

echo ""
echo "✓ Linear operation model training completed"
echo "  Models saved to: $OUTPUT_DIR"
echo ""
