#!/bin/bash

# Test script for MoE trainer
# This script validates the MoE training functionality using test data

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHON_BIN

echo "Using Python executable: $PYTHON_BIN"

echo "=========================================="
echo "MoE Trainer Test Suite"
echo "=========================================="
echo ""

# Configuration
TEST_DATA_PATH="data/profiling/compute/a800/mixtral_8x7b_moe/moe.csv"
TEST_OUTPUT_DIR="cache/trained_models"
TEST_MODEL_NAME="mixtral_8x7b_moe"
TEST_DEVICE="a800"
TEST_MOE_TP_SIZE=2
TEST_EP_SIZE=2

# Clean up previous test output
# IMPORTANT: Do NOT clean up if TEST_OUTPUT_DIR is the cache directory
# to preserve models trained by other scripts (e.g., Attention, MLP models)
NORMALIZED_OUTPUT_DIR=$(cd "$(dirname "$TEST_OUTPUT_DIR")" 2>/dev/null && pwd)/$(basename "$TEST_OUTPUT_DIR") || echo "$TEST_OUTPUT_DIR"
CACHE_DIR_PATTERN="(^|/)cache$|(^|/)cache/$"

if [[ "$TEST_OUTPUT_DIR" =~ $CACHE_DIR_PATTERN ]] || [[ "$NORMALIZED_OUTPUT_DIR" =~ $CACHE_DIR_PATTERN ]]; then
    echo "⚠️  Skipping cleanup: TEST_OUTPUT_DIR points to cache directory"
    echo "   Preserving existing models to avoid deleting models trained by other scripts"
    echo "   Output directory: $TEST_OUTPUT_DIR"
elif [ -d "$TEST_OUTPUT_DIR" ]; then
    echo "Cleaning up previous test output..."
    rm -rf "$TEST_OUTPUT_DIR"
fi

mkdir -p "$TEST_OUTPUT_DIR"

echo "Test Configuration:"
echo "  Dataset:                 $TEST_DATA_PATH"
echo "  Output Dir:              $TEST_OUTPUT_DIR"
echo "  Model:                   $TEST_MODEL_NAME"
echo "  Device:                  $TEST_DEVICE"
echo "  MoE TP Size:             $TEST_MOE_TP_SIZE"
echo "  Expert Parallel Size:    $TEST_EP_SIZE"
echo ""

# Check if test data exists
if [ ! -f "$TEST_DATA_PATH" ]; then
    echo "❌ Error: Test data not found at $TEST_DATA_PATH"
    echo ""
    echo "Please ensure you have MoE profiling data available."
    echo "You can generate it by running:"
    echo "  python -m frontier.profiling.moe.main --device a100 --models qwen2_moe_example"
    exit 1
fi

echo "✓ Test data found"
echo ""

# Test 1: Train with model configuration
echo "=========================================="
echo "Test 1: Train MoE models with model config"
echo "=========================================="
echo ""

BASE_PATH="$SCRIPT_DIR"

bash "$BASE_PATH/train_moe_models.sh" \
    --dataset_path "$TEST_DATA_PATH" \
    --output_dir "$TEST_OUTPUT_DIR" \
    --model_name "$TEST_MODEL_NAME" \
    --device "$TEST_DEVICE" \
    --moe_tensor_parallel_size "$TEST_MOE_TP_SIZE" \
    --expert_parallel_size "$TEST_EP_SIZE"

echo ""
echo "✓ Test 1 passed: Training completed successfully"
echo ""

# Test 2: Verify output files
echo "=========================================="
echo "Test 2: Verify trained model files"
echo "=========================================="
echo ""

EXPECTED_MODELS=(
    "moe_gating_linear"
    "moe_gating_routing_topk"
    "moe_shuffling"
    "moe_grouped_gemm"
)

MISSING_MODELS=()

for model in "${EXPECTED_MODELS[@]}"; do
    # Look for any file matching the pattern: {model_name}_*.pkl
    if ls "$TEST_OUTPUT_DIR"/${model}_*.pkl 1> /dev/null 2>&1; then
        echo "✓ Found model: $model"
    else
        echo "❌ Missing model: $model"
        MISSING_MODELS+=("$model")
    fi
done

echo ""

if [ ${#MISSING_MODELS[@]} -eq 0 ]; then
    echo "✓ Test 2 passed: All expected models found"
else
    echo "❌ Test 2 failed: Missing models: ${MISSING_MODELS[*]}"
    exit 1
fi

echo ""

# Test 3: Verify model loading
echo "=========================================="
echo "Test 3: Verify model loading from cache"
echo "=========================================="
echo ""

# Run training again - should load from cache
echo "Running training again (should use cache)..."
echo ""

bash "$BASE_PATH/train_moe_models.sh" \
    --dataset_path "$TEST_DATA_PATH" \
    --output_dir "$TEST_OUTPUT_DIR" \
    --model_name "$TEST_MODEL_NAME" \
    --device "$TEST_DEVICE" \
    --moe_tensor_parallel_size "$TEST_MOE_TP_SIZE" \
    --expert_parallel_size "$TEST_EP_SIZE" \
    2>&1 | grep -q "Using cached model" || {
        echo "❌ Test 3 failed: Models were not loaded from cache"
        exit 1
    }

echo "✓ Test 3 passed: Models loaded from cache successfully"
echo ""
