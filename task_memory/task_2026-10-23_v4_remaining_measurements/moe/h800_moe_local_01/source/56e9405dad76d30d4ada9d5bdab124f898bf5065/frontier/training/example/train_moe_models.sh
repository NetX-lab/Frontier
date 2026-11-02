#!/bin/bash

# Shell script for training MoE execution time prediction models
# This script provides a convenient way to train MoE models with common configurations

set -e


# Use Python from the active simulator/training environment. Override with PYTHON_BIN=/path/to/python.
PYTHON_BIN="${PYTHON_BIN:-python}"


# Default values
DATASET_PATH="data/profiling/compute/a800/mixtral_8x7b_moe/moe.csv"
OUTPUT_DIR="cache"
MODEL_NAME="mixtral_8x7b_moe"
DEVICE="a800"
MOE_TP_SIZE=2
EP_SIZE=2
PREDICTOR_TYPE="random_forest"
MEASUREMENT_TYPE="CUDA_EVENT"
ROUTING_RUNTIME_PATH="standard_fused_topk"
GATING_RUNTIME_CONTEXT="standalone_legacy"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_path)
            DATASET_PATH="$2"
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
        --moe_tensor_parallel_size)
            MOE_TP_SIZE="$2"
            shift 2
            ;;
        --expert_parallel_size)
            EP_SIZE="$2"
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
        --routing_runtime_path|--routing-runtime-path)
            ROUTING_RUNTIME_PATH="$2"
            shift 2
            ;;
        --gating_runtime_context|--gating-runtime-context)
            GATING_RUNTIME_CONTEXT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset_path PATH          Path to MoE profiling dataset CSV (required)"
            echo "  --output_dir DIR             Output directory for trained models (default: cache)"
            echo "  --model_name NAME            Model name (e.g., mixtral_8x7b_moe)"
            echo "  --device DEVICE              Device SKU (default: a100)"
            echo "  --moe_tensor_parallel_size N MoE tensor parallel size (default: 1)"
            echo "  --expert_parallel_size N     Expert parallel size (default: 1)"
            echo "  --predictor_type TYPE        Predictor type: random_forest or linear_regression (default: random_forest)"
            echo "  --measurement_type TYPE      Measurement type: CUDA_EVENT or KERNEL_ONLY (default: CUDA_EVENT)"
            echo "  --routing_runtime_path PATH  MoE routing path: standard_fused_topk or uniform_topk (default: standard_fused_topk)"
            echo "  --gating_runtime_context CTX MoE gating context: standalone_legacy or prefill_hot (default: standalone_legacy)"
            echo "  -h, --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Train with model configuration"
            echo "  $0 --dataset_path data/profiling/compute/a100/mixtral_8x7b_moe/moe.csv \\"
            echo "     --model_name mixtral_8x7b_moe --device a100"
            echo ""
            echo "  # Train with custom parallelism"
            echo "  $0 --dataset_path data/profiling/compute/a100/mixtral_8x7b_moe/moe.csv \\"
            echo "     --model_name mixtral_8x7b_moe --moe_tensor_parallel_size 2 --expert_parallel_size 2"
            echo ""
            echo "  # Train Qwen3 RTX PRO 6000 prefill-hot uniform-topk rows"
            echo "  $0 --dataset_path data/profiling/compute/rtx_pro_6000/Qwen3-30B-A3B-tiny/moe.csv \\"
            echo "     --model_name Qwen3-30B-A3B-tiny --device rtx_pro_6000 \\"
            echo "     --routing_runtime_path uniform_topk --gating_runtime_context prefill_hot"
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
if [ -z "$DATASET_PATH" ]; then
    echo "Error: --dataset_path is required"
    echo "Run with --help for usage information"
    exit 1
fi

if [ -z "$MODEL_NAME" ]; then
    echo "Error: --model_name is required"
    echo "Run with --help for usage information"
    exit 1
fi

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset not found: $DATASET_PATH"
    exit 1
fi


# Ensure Frontier is importable from the repository root.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Display configuration
echo "=========================================="
echo "MoE Model Training Configuration"
echo "=========================================="
echo "Dataset Path:              $DATASET_PATH"
echo "Output Directory:          $OUTPUT_DIR"
echo "Model Name:                $MODEL_NAME"
echo "Device:                    $DEVICE"
echo "MoE Tensor Parallel Size:  $MOE_TP_SIZE"
echo "Expert Parallel Size:      $EP_SIZE"
echo "Predictor Type:            $PREDICTOR_TYPE"
echo "Measurement Type:          $MEASUREMENT_TYPE"
echo "Routing Runtime Path:     $ROUTING_RUNTIME_PATH"
echo "Gating Runtime Context:   $GATING_RUNTIME_CONTEXT"
echo "=========================================="
echo ""

# Run training
echo "Starting MoE model training..."
echo ""

"$PYTHON_BIN" -m frontier.training.cli moe \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --device "$DEVICE" \
    --moe_tensor_parallel_size "$MOE_TP_SIZE" \
    --expert_parallel_size "$EP_SIZE" \
    --predictor_type "$PREDICTOR_TYPE" \
    --measurement_type "$MEASUREMENT_TYPE" \
    --routing_runtime_path "$ROUTING_RUNTIME_PATH" \
    --gating_runtime_context "$GATING_RUNTIME_CONTEXT"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="
    echo "Trained models saved to: $OUTPUT_DIR"
    echo ""
    echo "You can now use these models in simulations by ensuring:"
    echo "1. The output directory matches your simulation's cache_dir"
    echo "2. The model configuration matches your simulation parameters"
    exit 0
else
    echo ""
    echo "=========================================="
    echo "Training failed!"
    echo "=========================================="
    echo "Please check the error messages above"
    exit 1
fi
