#!/bin/bash
# Run the sequential PD-AF offline dense CUDA Graph example.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export MODEL_NAME="${MODEL_NAME:-llama2_7b_dense_example}"
export TOTAL_EXPERTS="${TOTAL_EXPERTS:-1}"
export ROUTER_TOPK="${ROUTER_TOPK:-1}"
export PREFILL_MOE_TP="${PREFILL_MOE_TP:-1}"
export PREFILL_MOE_EP="${PREFILL_MOE_EP:-1}"
export DECODE_FFN_MOE_TP="${DECODE_FFN_MOE_TP:-1}"
export DECODE_FFN_MOE_EP="${DECODE_FFN_MOE_EP:-1}"
export ENABLE_CUDA_GRAPH=true
export EXAMPLE_LABEL="${EXAMPLE_LABEL:-Dense CUDA Graph}"
export RUN_ID="${RUN_ID:-dense_cuda_graph}"

if [ "$#" -gt 0 ] && [ "$1" = "--" ]; then
  shift
fi

exec bash "$SCRIPT_DIR/moe_model_basic.sh" -- "$@"
