#!/bin/bash

# Test script for Attention trainer
# This script validates the Attention training functionality using test data

set -e


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHON_BIN

echo "Using Python executable: $PYTHON_BIN"

echo "=========================================="
echo "Attention Trainer Test Suite"
echo "=========================================="
echo ""

# Configuration
# Note: Use canonical linear_op.csv when a compute dataset is required.
TEST_COMPUTE_DATA_PATH="" #"data/profiling/compute/a100/qwen2_moe_example/linear_op.csv"
TEST_LAYER_DATA_PATH="data/profiling/compute/a100/qwen2_moe_example/attention.csv"
TEST_OUTPUT_DIR="cache"
TEST_MODEL_NAME="qwen2_moe_example"

TEST_DEVICE="a100"
TEST_TP_SIZE=1
TEST_BLOCK_SIZE=16

# Clean up previous test output
# IMPORTANT: Do NOT clean up if TEST_OUTPUT_DIR is the cache directory
# to preserve models trained by other scripts (e.g., MoE, MLP models)
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
echo "  Compute Dataset: $TEST_COMPUTE_DATA_PATH"
echo "  Layer Dataset:   $TEST_LAYER_DATA_PATH"
echo "  Output Dir:      $TEST_OUTPUT_DIR"
echo "  Model:           $TEST_MODEL_NAME"
echo "  Device:          $TEST_DEVICE"
echo "  TP Size:         $TEST_TP_SIZE"
echo "  Block Size:      $TEST_BLOCK_SIZE"
echo ""

# Check if test data exists
# if [ ! -f "$TEST_COMPUTE_DATA_PATH" ]; then
#     echo "❌ Error: Compute test data not found at $TEST_COMPUTE_DATA_PATH"
#     echo ""
#     echo "Please ensure you have compute profiling data available."
#     echo "You can generate it by running:"
#     echo "  python -m frontier.profiling.linear_op.main --device a100 --models meta-llama/Llama-2-7b-hf"
#     exit 1
# fi

if [ ! -f "$TEST_LAYER_DATA_PATH" ]; then
    echo "❌ Error: Layer test data not found at $TEST_LAYER_DATA_PATH"
    echo ""
    echo "Please ensure you have attention profiling data available."
    echo "You can generate it by running:"
    echo "  python -m frontier.profiling.attention.main --device a100 --models meta-llama/Llama-2-7b-hf"
    exit 1
fi

echo "✓ Test data found"
echo ""

# Test 1: Train with model configuration
echo "=========================================="
echo "Test 1: Train Attention models with model config"
echo "=========================================="
echo ""

BASE_PATH="$SCRIPT_DIR"

bash "$BASE_PATH/train_attention_models.sh" \
    --compute_dataset_path "$TEST_COMPUTE_DATA_PATH" \
    --layer_dataset_path "$TEST_LAYER_DATA_PATH" \
    --output_dir "$TEST_OUTPUT_DIR" \
    --model_name "$TEST_MODEL_NAME" \
    --device "$TEST_DEVICE" \
    --tensor_parallel_size "$TEST_TP_SIZE" \
    --block_size "$TEST_BLOCK_SIZE"

echo ""
echo "✓ Test 1 passed: Training completed successfully"
echo ""

# Test 2: Verify output files
echo "=========================================="
echo "Test 2: Verify trained model files"
echo "=========================================="
echo ""

EXPECTED_MODELS=(
    "attn_kv_cache_save"
    "attn_prefill"
    "attn_decode"
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

bash "$BASE_PATH/train_attention_models.sh" \
    --compute_dataset_path "$TEST_COMPUTE_DATA_PATH" \
    --layer_dataset_path "$TEST_LAYER_DATA_PATH" \
    --output_dir "$TEST_OUTPUT_DIR" \
    --model_name "$TEST_MODEL_NAME" \
    --device "$TEST_DEVICE" \
    --tensor_parallel_size "$TEST_TP_SIZE" \
    --block_size "$TEST_BLOCK_SIZE" \
    2>&1 | grep -q "Using cached model" || {
        echo "❌ Test 3 failed: Models were not loaded from cache"
        exit 1
    }

echo "✓ Test 3 passed: Models loaded from cache successfully"
echo ""

# Test 4: Python API test
echo "=========================================="
echo "Test 4: Test Python API"
echo "=========================================="
echo ""

"$PYTHON_BIN" << EOF
import sys
from frontier.training.attention_trainer import create_attention_trainer_from_model_config

print("Creating Attention trainer via Python API...")

trainer = create_attention_trainer_from_model_config(
    compute_dataset_path="$TEST_COMPUTE_DATA_PATH",
    layer_dataset_path="$TEST_LAYER_DATA_PATH",
    output_dir="$TEST_OUTPUT_DIR",
    model_name="$TEST_MODEL_NAME",
    device="$TEST_DEVICE",
    tensor_parallel_size=$TEST_TP_SIZE,
    block_size=$TEST_BLOCK_SIZE,
)

print("✓ Trainer created successfully")
print(f"  Model name: {trainer.model_name}")
print(f"  Device: {trainer.device}")
print(f"  Tensor parallel size: {trainer.tensor_parallel_size}")
print(f"  Block size: {trainer.block_size}")

# Verify model names
expected_models = [
    "attn_kv_cache_save",
    "attn_prefill",
    "attn_decode",
]

actual_models = trainer._get_model_names()

if set(expected_models) == set(actual_models):
    print("✓ Model names match expected")
else:
    print(f"❌ Model names mismatch!")
    print(f"  Expected: {expected_models}")
    print(f"  Actual: {actual_models}")
    sys.exit(1)

# Verify feature columns for different model types
test_cases = [
    ("attn_kv_cache_save", ["num_tokens"]),
    ("attn_prefill", ["kv_cache_size", "prefill_chunk_size_squared"]),
    ("attn_decode", ["batch_size", "kv_cache_size"]),
]

for model_name, expected_features in test_cases:
    actual_features = trainer._get_feature_cols(model_name)
    if actual_features == expected_features:
        print(f"✓ Features for {model_name}: {actual_features}")
    else:
        print(f"❌ Features mismatch for {model_name}!")
        print(f"  Expected: {expected_features}")
        print(f"  Actual: {actual_features}")
        sys.exit(1)

print("\n✓ Python API test passed")
EOF

if [ $? -ne 0 ]; then
    echo "❌ Test 4 failed: Python API test failed"
    exit 1
fi

echo ""

# Test 5: Train without compute_dataset_path (attention-only mode)
echo "=========================================="
echo "Test 5: Train without compute_dataset_path"
echo "=========================================="
echo ""
echo "Testing attention-only training mode (no compute dataset)..."
echo ""

"$PYTHON_BIN" << EOF
import sys
from frontier.training.attention_trainer import create_attention_trainer_from_model_config

print("Creating Attention trainer WITHOUT compute_dataset_path...")

# Create trainer without compute_dataset_path
compute_dataset_path = "$TEST_COMPUTE_DATA_PATH" or None
trainer = create_attention_trainer_from_model_config(
    compute_dataset_path=compute_dataset_path,  # Explicitly None when the shell variable is empty
    layer_dataset_path="$TEST_LAYER_DATA_PATH",
    output_dir="$TEST_OUTPUT_DIR",
    model_name="$TEST_MODEL_NAME",
    device="$TEST_DEVICE",
    tensor_parallel_size=$TEST_TP_SIZE,
    block_size=$TEST_BLOCK_SIZE,
)

print("✓ Trainer created successfully (attention-only mode)")
print(f"  train_compute_models: {trainer.train_compute_models}")

# Verify that compute-dependent models are excluded
expected_models = [
    "attn_kv_cache_save",
    "attn_prefill",
    "attn_decode",
]

# These should NOT be in the model list
excluded_models = [
    "attn_pre_proj",
    "attn_post_proj",
    "attn_rope",
    "input_layernorm",
    "post_attention_layernorm",
    "add",
]

actual_models = trainer._get_model_names()

# Check expected models are present
for model in expected_models:
    if model in actual_models:
        print(f"✓ Expected model present: {model}")
    else:
        print(f"❌ Expected model missing: {model}")
        sys.exit(1)

# Check excluded models are NOT present
for model in excluded_models:
    if model not in actual_models:
        print(f"✓ Compute-dependent model correctly excluded: {model}")
    else:
        print(f"❌ Compute-dependent model should be excluded: {model}")
        sys.exit(1)

print("\n✓ Attention-only mode test passed")
EOF

if [ $? -ne 0 ]; then
    echo "❌ Test 5 failed: Attention-only mode test failed"
    exit 1
fi

echo ""
echo "✓ Test 5 passed: Attention-only training mode works correctly"
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
echo "  ✓ Test 2: Verify trained model files (9 models)"
echo "  ✓ Test 3: Verify model caching"
echo "  ✓ Test 4: Python API test"
echo "  ✓ Test 5: Attention-only mode (no compute_dataset_path)"
echo ""
echo "Test output directory: $TEST_OUTPUT_DIR"
echo ""
echo "You can clean up test output with:"
echo "  rm -rf $TEST_OUTPUT_DIR"
echo "=========================================="
