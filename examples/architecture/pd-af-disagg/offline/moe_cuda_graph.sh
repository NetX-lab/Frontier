#!/bin/bash
# Run the sequential PD-AF offline MoE CUDA Graph example.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export ENABLE_CUDA_GRAPH=true
export EXAMPLE_LABEL="${EXAMPLE_LABEL:-MoE CUDA Graph (EP=1)}"
export RUN_ID="${RUN_ID:-moe_cuda_graph}"

if [ "$#" -gt 0 ] && [ "$1" = "--" ]; then
  shift
fi

exec bash "$SCRIPT_DIR/moe_model_basic.sh" -- "$@"
