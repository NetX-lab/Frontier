#!/bin/bash

# ============================================================
# Legacy internal script: Automated Training Script for PD-AF Disaggregation Simulation
# ============================================================
# This script trains all required models with the CORRECT configuration
# to match legacy internal PD-AF simulation requirements.
# It is not part of the pre-release-v0.1 co-location release.
#
# Usage:
#   bash tests/training/train_models_for_simulation.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHON_BIN

echo "Using Python executable: $PYTHON_BIN"


echo "=========================================="
echo "Automated Model Training for Simulation"
echo "=========================================="
echo ""
echo "This script will train models with the following configurations:"
echo ""
echo "PREFILL Cluster:"
echo "  - Device: a100"
echo "  - Attention TP Size: 2"
echo "  - MoE TP Size: 2"
echo "  - MoE EP Size: 2"
echo ""
echo "DECODE_ATTN Cluster:"
echo "  - Device: h100"
echo "  - Attention TP Size: 2"
echo ""
echo "DECODE_FFN Cluster:"
echo "  - Device: a100"
echo "  - MoE TP Size: 2"
echo "  - MoE EP Size: 2"
echo ""
echo "=========================================="
echo ""

# Configuration
MODEL_NAME="qwen2_moe_example"
CACHE_DIR="cache"
BLOCK_SIZE=16

# ============================================================
# Step 1: Verify profiling data exists
# ============================================================
echo "Step 1: Verify profiling data availability"
echo "=========================================="
echo ""

REQUIRED_DATA=(
    "data/profiling/compute/a100/qwen2_moe_example/linear_op.csv"
    "data/profiling/compute/a100/qwen2_moe_example/attention.csv"
    "data/profiling/compute/h100/qwen2_moe_example/linear_op.csv"
    "data/profiling/compute/h100/qwen2_moe_example/attention.csv"
    "data/profiling/compute/a100/qwen2_moe_example/moe.csv"
)

MISSING_DATA=()

for data_file in "${REQUIRED_DATA[@]}"; do
    if [ -f "$data_file" ]; then
        echo "✓ Found: $data_file"
    else
        echo "❌ Missing: $data_file"
        MISSING_DATA+=("$data_file")
    fi
done

echo ""

if [ ${#MISSING_DATA[@]} -ne 0 ]; then
    echo "❌ Error: Missing required profiling data files:"
    for missing in "${MISSING_DATA[@]}"; do
        echo "   - $missing"
    done
    echo ""
    echo "Please generate profiling data before training models."
    echo "See frontier/profiling/ for profiling scripts."
    exit 1
fi

echo "✓ All required profiling data files found"
echo ""

# ============================================================
# Step 2: Train models for PREFILL cluster (a100, TP=2)
# ============================================================
echo "Step 2: Train models for PREFILL cluster"
echo "=========================================="
echo "Configuration: device=a100, attn_tp=2, moe_tp=2, moe_ep=2"
echo ""

echo "Training Attention models..."
bash "$SCRIPT_DIR/train_attention_models.sh" \
    --compute_dataset_path "data/profiling/compute/a100/$MODEL_NAME/linear_op.csv" \
    --layer_dataset_path "data/profiling/compute/a100/$MODEL_NAME/attention.csv" \
    --output_dir "$CACHE_DIR" \
    --model_name "$MODEL_NAME" \
    --device "a100" \
    --tensor_parallel_size 2 \
    --block_size "$BLOCK_SIZE"

echo ""
echo "Training Linear Operation models..."
bash "$SCRIPT_DIR/train_linear_op_models.sh" \
    --model "$MODEL_NAME" \
    --device "a100" \
    --tp-size 2 \
    --measurement-type CUDA_EVENT

echo ""
echo "Training MoE models..."
bash "$SCRIPT_DIR/train_moe_models.sh" \
    --dataset_path "data/profiling/compute/a100/$MODEL_NAME/moe.csv" \
    --output_dir "$CACHE_DIR" \
    --model_name "$MODEL_NAME" \
    --device "a100" \
    --moe_tensor_parallel_size 2 \
    --expert_parallel_size 2

echo ""
echo "✓ PREFILL cluster models trained successfully"
echo ""

# ============================================================
# Step 3: Train models for DECODE_ATTN cluster (h100, TP=2)
# ============================================================
echo "Step 3: Train models for DECODE_ATTN cluster"
echo "=========================================="
echo "Configuration: device=h100, attn_tp=2"
echo ""

echo "Training Attention models..."
bash "$SCRIPT_DIR/train_attention_models.sh" \
    --compute_dataset_path "data/profiling/compute/h100/$MODEL_NAME/linear_op.csv" \
    --layer_dataset_path "data/profiling/compute/h100/$MODEL_NAME/attention.csv" \
    --output_dir "$CACHE_DIR" \
    --model_name "$MODEL_NAME" \
    --device "h100" \
    --tensor_parallel_size 2 \
    --block_size "$BLOCK_SIZE"

echo ""
echo "✓ DECODE_ATTN cluster models trained successfully"
echo ""

# ============================================================
# Step 4: Verify cache completeness
# ============================================================
echo "Step 4: Verify cache completeness"
echo "=========================================="
echo ""

echo "Checking cache directory..."
CACHE_FILE_COUNT=$(find "$CACHE_DIR" -name "*.pkl" -type f | wc -l)
echo "Total cached models: $CACHE_FILE_COUNT"
echo ""

echo "Expected models for PREFILL cluster (a100, TP=2):"
echo "  - Attention models (9): attn_pre_proj, attn_post_proj, attn_rope, etc."
echo "  - MLP models (6): mlp_up_proj, mlp_down_proj, mlp_act, etc."
echo "  - MoE models (4): moe_gating_linear, moe_gating_routing_topk, moe_shuffling, moe_grouped_gemm"
echo ""

echo "Expected models for DECODE_ATTN cluster (h100, TP=2):"
echo "  - Attention models (9): attn_pre_proj, attn_post_proj, attn_rope, etc."
echo ""

echo "Listing all cached models:"
ls -lh "$CACHE_DIR"/*.pkl | awk '{print "  " $9 " (" $5 ")"}'
echo ""

# ============================================================
# Step 5: Verify hash calculation (optional)
# ============================================================
echo "Step 5: Verify hash calculation (optional)"
echo "=========================================="
echo ""

echo "You can verify hash calculation for specific models using:"
echo ""
echo "  python tests/training/verify_hash_calculation.py \\"
echo "      --model_name $MODEL_NAME \\"
echo "      --device a100 \\"
echo "      --tensor_parallel_size 2 \\"
echo "      --operation attn_pre_proj"
echo ""
echo "  python tests/training/verify_hash_calculation.py \\"
echo "      --model_name $MODEL_NAME \\"
echo "      --device h100 \\"
echo "      --tensor_parallel_size 2 \\"
echo "      --operation attn_pre_proj"
echo ""

# ============================================================
# Summary
# ============================================================
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "✅ All models have been trained with the correct configuration"
echo ""
echo "Next steps:"
echo "  1. Verify cache completeness:"
echo "     bash tests/training/verify_cache_completeness.sh"
echo ""
echo "  2. Run simulation:"
echo "     This legacy PD-AF simulation helper is not part of the pre-release-v0.1 co-location release."
echo "     Use frontier.main with --sys_arch co-location for release validation."
echo ""
echo "  3. If you encounter cache miss errors, use the hash verification script:"
echo "     python tests/training/verify_hash_calculation.py --help"
echo ""
