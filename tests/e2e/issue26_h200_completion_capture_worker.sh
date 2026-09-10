#!/usr/bin/env bash
# Capture CUDA completion timing for the standard H200 diagnostic batch.
set -euo pipefail

OUT_ROOT="${1:?Provide a fresh persistent output directory.}"
SOURCE="${ISSUE26_DIAGNOSTIC_VLLM_SOURCE:-/data/ycfeng/tmp/issue26-vllm-diagnostics-ar-ab}"
COMMIT="${ISSUE26_DIAGNOSTIC_VLLM_COMMIT:-f35033e05a5fa5632ac4a4952981cb8c14e9f346}"

export VLLM_MOE_UNIFORM_ROUTING=1
export ISSUE26_DIAGNOSTIC_VLLM_SOURCE="$SOURCE"
export ISSUE26_DIAGNOSTIC_VLLM_COMMIT="$COMMIT"
export VLLM_FRONTIER_MOE_COMPLETION_LOG_PATH="$OUT_ROOT/batch/runtime/batch/server.moe_completion"

bash "$(dirname "$0")/issue26_h200_diagnostics_worker.sh" \
  "$OUT_ROOT/batch/runtime" batch

echo "H200_COMPLETION_CAPTURE_EXECUTION_COMPLETE"
