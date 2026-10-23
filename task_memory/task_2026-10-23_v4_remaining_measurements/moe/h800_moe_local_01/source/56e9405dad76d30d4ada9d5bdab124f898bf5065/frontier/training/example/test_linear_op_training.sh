#!/bin/bash

# Test script for Linear Operation trainer
# This script validates the linear operation training functionality using test data
# NOTE: This file validates the canonical linear_op training flow.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHON_BIN

echo "Using Python executable: $PYTHON_BIN"


echo "=========================================="
echo "Linear Operation Trainer Test Suite"
echo "=========================================="
echo ""

# Configuration
TEST_DATA_PATH="data/profiling/compute/a100/qwen2_moe_example/linear_op.csv"
TEST_OUTPUT_DIR="cache"
TEST_MODEL_NAME="qwen2_moe_example"
TEST_DEVICE="a100"
TEST_TP_SIZE=1

# Parse command line arguments for is_moe
IS_MOE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --is-moe|--is_moe)
            IS_MOE=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Clean up previous test output
NORMALIZED_OUTPUT_DIR=$(cd "$(dirname "$TEST_OUTPUT_DIR")" 2>/dev/null && pwd)/$(basename "$TEST_OUTPUT_DIR") || echo "$TEST_OUTPUT_DIR"
CACHE_DIR_PATTERN="(^|/)cache$|(^|/)cache/$"

if [[ "$TEST_OUTPUT_DIR" =~ $CACHE_DIR_PATTERN ]] || [[ "$NORMALIZED_OUTPUT_DIR" =~ $CACHE_DIR_PATTERN ]]; then
    echo "⚠️  Skipping cleanup: TEST_OUTPUT_DIR points to cache directory"
elif [ -d "$TEST_OUTPUT_DIR" ]; then
    echo "Cleaning up previous test output..."
    rm -rf "$TEST_OUTPUT_DIR"
fi

mkdir -p "$TEST_OUTPUT_DIR"

echo "Test Configuration:"
echo "  Dataset:     $TEST_DATA_PATH"
echo "  Output Dir:  $TEST_OUTPUT_DIR"
echo "  Model:       $TEST_MODEL_NAME"
echo "  Device:      $TEST_DEVICE"
echo "  TP Size:     $TEST_TP_SIZE"
echo "  Is MoE:      $IS_MOE"
echo ""

# Check if test data exists
if [ ! -f "$TEST_DATA_PATH" ]; then
    echo "❌ Error: Test data not found"
    echo "  Expected: $TEST_DATA_PATH"
    echo "Please ensure you have linear operation profiling data available."
    echo "You can generate it by running:"
    echo "  python -m frontier.profiling.linear_op.main --device a100 --models meta-llama/Llama-2-7b-hf"
    exit 1
fi

echo "✓ Test data found"
echo ""

# Test 1: Train with model configuration
echo "=========================================="
echo "Test 1: Train Linear Operation models with model config"
echo "=========================================="
echo ""

BASE_PATH="$SCRIPT_DIR"

TRAIN_CMD="bash \"$BASE_PATH/train_linear_op_models.sh\" \
    --model $TEST_MODEL_NAME \
    --device $TEST_DEVICE \
    --tp-size $TEST_TP_SIZE"

if [ "$IS_MOE" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --is-moe"
fi

eval $TRAIN_CMD

echo ""
echo "✓ Test 1 passed: Training completed successfully"
echo ""

# Test 2: Verify output files
echo "=========================================="
echo "Test 2: Verify trained model files"
echo "=========================================="
echo ""

# Define expected models based on is_moe flag
if [ "$IS_MOE" = true ]; then
    EXPECTED_MODELS=(
        "input_layernorm"
        "post_attention_layernorm"
        "add"
        "attn_pre_proj"
        "attn_post_proj"
        "attn_rope"
    )
    SKIP_MODELS=(
        "mlp_up_proj"
        "mlp_down_proj"
        "mlp_act"
    )
else
    EXPECTED_MODELS=(
        "mlp_up_proj"
        "mlp_down_proj"
        "mlp_act"
        "input_layernorm"
        "post_attention_layernorm"
        "add"
        "attn_pre_proj"
        "attn_post_proj"
        "attn_rope"
    )
    SKIP_MODELS=()
fi

MISSING_MODELS=()
UNEXPECTED_MODELS=()

for model in "${EXPECTED_MODELS[@]}"; do
    if ls "$TEST_OUTPUT_DIR"/${model}_*.pkl 1> /dev/null 2>&1; then
        echo "✓ Found model: $model"
    else
        echo "❌ Missing model: $model"
        MISSING_MODELS+=("$model")
    fi
done

for model in "${SKIP_MODELS[@]}"; do
    if ls "$TEST_OUTPUT_DIR"/${model}_*.pkl 1> /dev/null 2>&1; then
        echo "❌ Unexpected model (should be skipped): $model"
        UNEXPECTED_MODELS+=("$model")
    else
        echo "✓ Correctly skipped model: $model"
    fi
done

echo ""

if [ ${#MISSING_MODELS[@]} -eq 0 ] && [ ${#UNEXPECTED_MODELS[@]} -eq 0 ]; then
    echo "✓ Test 2 passed: All expected models found"
else
    echo "❌ Test 2 failed"
    exit 1
fi

echo ""

# Test 3: Python API test
echo "=========================================="
echo "Test 3: Test Python API"
echo "=========================================="
echo ""

IS_MOE_PYTHON=$([ "$IS_MOE" = true ] && echo "True" || echo "False")

"$PYTHON_BIN" << EOF
import sys
from frontier.training.linear_op_trainer import create_linear_op_trainer_from_model_config

print("Creating Linear Operation trainer via Python API...")

trainer = create_linear_op_trainer_from_model_config(
    dataset_path="$TEST_DATA_PATH",
    output_dir="$TEST_OUTPUT_DIR",
    model_name="$TEST_MODEL_NAME",
    device="$TEST_DEVICE",
    tensor_parallel_size=$TEST_TP_SIZE,
    is_moe=$IS_MOE_PYTHON,
)

print("✓ Trainer created successfully")
print(f"  Model name: {trainer.model_name}")
print(f"  Device: {trainer.device}")
print(f"  Tensor parallel size: {trainer.tensor_parallel_size}")
print(f"  Is MoE: {trainer.is_moe}")

# Verify model names based on is_moe
is_moe = $IS_MOE_PYTHON
if is_moe:
    expected_models = [
        "input_layernorm",
        "post_attention_layernorm",
        "add",
        "attn_pre_proj",
        "attn_post_proj",
        "attn_rope",
    ]
else:
    expected_models = [
        "mlp_up_proj",
        "mlp_down_proj",
        "mlp_act",
        "input_layernorm",
        "post_attention_layernorm",
        "add",
        "attn_pre_proj",
        "attn_post_proj",
        "attn_rope",
    ]

actual_models = trainer._get_model_names()

if set(expected_models) == set(actual_models):
    print("✓ Model names match expected")
else:
    print(f"❌ Model names mismatch!")
    print(f"  Expected: {expected_models}")
    print(f"  Actual: {actual_models}")
    sys.exit(1)

print("\n✓ Python API test passed")
EOF

if [ $? -ne 0 ]; then
    echo "❌ Test 3 failed: Python API test failed"
    exit 1
fi

echo ""

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""
echo "✓ All tests passed!"
echo ""
echo "Tests completed:"
echo "  ✓ Test 1: Training with model configuration"
echo "  ✓ Test 2: Verify trained model files"
echo "  ✓ Test 3: Python API test"
echo ""
echo "Test configuration:"
echo "  Is MoE: $IS_MOE"
echo "  Output directory: $TEST_OUTPUT_DIR"
echo "=========================================="
