#!/bin/bash

# Shell script for training Attention execution time prediction models
# This script provides a convenient way to train Attention models with common configurations

set -e

# Use Python from the active simulator/training environment. Override with PYTHON_BIN=/path/to/python.
PYTHON_BIN="${PYTHON_BIN:-python}"


# Default values
COMPUTE_DATASET_PATH=""
LAYER_DATASET_PATH="data/profiling/compute/a800/mixtral_8x7b_moe/attention.csv"
OUTPUT_DIR="cache"
MODEL_NAME="mixtral_8x7b_moe"
DEVICE="a800"
TENSOR_PARALLEL_SIZE=2
BLOCK_SIZE=16
PREDICTOR_TYPE="random_forest"
MEASUREMENT_TYPE="CUDA_EVENT"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --compute_dataset_path)
            COMPUTE_DATASET_PATH="$2"
            shift 2
            ;;
        --layer_dataset_path)
            LAYER_DATASET_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --tensor_parallel_size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --block_size)
            BLOCK_SIZE="$2"
            shift 2
            ;;
        --predictor_type|--predictor-type)
            PREDICTOR_TYPE="$2"
            shift 2
            ;;
        --measurement_type|--measurement-type)
            MEASUREMENT_TYPE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --compute_dataset_path PATH  Path to compute profiling dataset CSV (linear_op.csv) (OPTIONAL)"
            echo "                               When not provided, compute-dependent models are skipped."
            echo "  --layer_dataset_path PATH    Path to layer profiling dataset CSV (attention.csv) (required)"
            echo "  --output_dir DIR             Output directory for trained models (default: cache)"
            echo "  --model_name NAME            Model name (e.g., meta-llama/Llama-2-7b-hf)"
            echo "  --device DEVICE              Device SKU (default: a100)"
            echo "  --tensor_parallel_size N     Tensor parallel size (default: 1)"
            echo "  --block_size N               Block size for KV cache (default: 16)"
            echo "  --predictor_type TYPE        Predictor type: random_forest or linear_regression (default: random_forest)"
            echo "  --measurement_type TYPE      Measurement type: CUDA_EVENT or KERNEL_ONLY (default: CUDA_EVENT)"
            echo "  -h, --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Train all models (with compute dataset)"
            echo "  $0 --compute_dataset_path data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/linear_op.csv \\"
            echo "     --layer_dataset_path data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/attention.csv \\"
            echo "     --model_name meta-llama/Llama-2-7b-hf --device a100"
            echo ""
            echo "  # Train attention-only models (without compute dataset)"
            echo "  $0 --layer_dataset_path data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/attention.csv \\"
            echo "     --model_name meta-llama/Llama-2-7b-hf --device a100"
            echo ""
            echo "  # Train with custom tensor parallelism"
            echo "  $0 --compute_dataset_path data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/linear_op.csv \\"
            echo "     --layer_dataset_path data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/attention.csv \\"
            echo "     --model_name meta-llama/Llama-2-7b-hf --tensor_parallel_size 2"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
# Note: --compute_dataset_path is now OPTIONAL
if [ -z "$LAYER_DATASET_PATH" ]; then
    echo "Error: --layer_dataset_path is required"
    echo "Run with --help for usage information"
    exit 1
fi

# Validate compute dataset if provided
if [ -n "$COMPUTE_DATASET_PATH" ] && [ ! -f "$COMPUTE_DATASET_PATH" ]; then
    echo "Error: Compute dataset file not found: $COMPUTE_DATASET_PATH"
    exit 1
fi

if [ ! -f "$LAYER_DATASET_PATH" ]; then
    echo "Error: Layer dataset file not found: $LAYER_DATASET_PATH"
    exit 1
fi

# Display configuration
echo "=========================================="
echo "Attention Model Training Configuration"
echo "=========================================="
if [ -n "$COMPUTE_DATASET_PATH" ]; then
    echo "Compute Dataset:      $COMPUTE_DATASET_PATH"
    echo "Training Mode:        Full (all models)"
else
    echo "Compute Dataset:      (not provided)"
    echo "Training Mode:        Attention-only (skipping compute-dependent models)"
fi
echo "Layer Dataset:        $LAYER_DATASET_PATH"
echo "Output Directory:     $OUTPUT_DIR"
echo "Model Name:           $MODEL_NAME"
echo "Device:               $DEVICE"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Block Size:           $BLOCK_SIZE"
echo "Predictor Type:       $PREDICTOR_TYPE"
echo "Measurement Type:     $MEASUREMENT_TYPE"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
echo "Starting Attention model training..."
echo ""

if [ -z "$MODEL_NAME" ]; then
    echo "Error: --model_name is required"
    exit 1
fi

# Build command based on whether compute_dataset_path is provided
if [ -n "$COMPUTE_DATASET_PATH" ]; then
    # Train with compute dataset (all models)
    "$PYTHON_BIN" -m frontier.training.cli attention \
        --compute_dataset_path "$COMPUTE_DATASET_PATH" \
        --layer_dataset_path "$LAYER_DATASET_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --model_name "$MODEL_NAME" \
        --device "$DEVICE" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --block_size "$BLOCK_SIZE" \
        --predictor_type "$PREDICTOR_TYPE" \
        --measurement_type "$MEASUREMENT_TYPE"
else
    # Train without compute dataset (attention-only models)
    "$PYTHON_BIN" -m frontier.training.cli attention \
        --layer_dataset_path "$LAYER_DATASET_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --model_name "$MODEL_NAME" \
        --device "$DEVICE" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --block_size "$BLOCK_SIZE" \
        --predictor_type "$PREDICTOR_TYPE" \
        --measurement_type "$MEASUREMENT_TYPE"
fi

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Trained models saved to: $OUTPUT_DIR"
echo ""
echo "Models trained:"
if [ -n "$COMPUTE_DATASET_PATH" ]; then
    echo "  Compute-dependent models (6):"
    echo "    - attn_pre_proj"
    echo "    - attn_post_proj"
    echo "    - attn_rope"
    echo "    - input_layernorm"
    echo "    - post_attention_layernorm"
    echo "    - add"
fi
echo "  Layer models (3-4):"
echo "    - attn_kv_cache_save"
echo "    - attn_prefill"
echo "    - attn_decode"
echo "    - attn_prefill_mixed (if mixed-batch data available)"
echo ""
echo "You can now use these models in simulations by setting:"
echo "  execution_time_predictor_config.cache_dir=$OUTPUT_DIR"
echo "=========================================="

