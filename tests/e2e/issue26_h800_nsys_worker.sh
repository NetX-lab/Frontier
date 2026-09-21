#!/usr/bin/env bash
# Run the canonical H800 replay chain with a gated first-formal profiler.
set -euo pipefail
RUN_ROOT="${1:?Provide a fresh output directory under the active task directory.}"
SOURCE="${ISSUE26_DIAGNOSTIC_VLLM_SOURCE:?Set the diagnostic vLLM source path.}"
COMMIT="${ISSUE26_DIAGNOSTIC_VLLM_COMMIT:?Set the diagnostic vLLM commit.}"
WORKERS="$(cd "$(dirname "$0")" && pwd)"
source /data/ycfeng/tmp/issue26_h800_environment_probe.sh "$RUN_ROOT/preflight"
source /data/ycfeng/tmp/issue26-h200-network/company-proxy.sh
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" ALL_PROXY="$all_proxy"
export NO_PROXY="$no_proxy,127.0.0.1,localhost,::1" no_proxy="$NO_PROXY"
git config --global --add safe.directory "$SOURCE"
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$SOURCE" status --porcelain)"
export ISSUE26_DIAGNOSTIC_VLLM_SOURCE="$SOURCE" ISSUE26_DIAGNOSTIC_VLLM_COMMIT="$COMMIT"
export VLLM_MOE_UNIFORM_ROUTING=1
export VLLM_FRONTIER_TORCH_PROFILER_CAPTURE=1
export VLLM_FRONTIER_TORCH_PROFILER_CAPTURE_PREFIX=cmpl-pf4096_dc1024:
export VLLM_FRONTIER_CUDA_PROFILER_CAPTURE_LIMIT=1
export VLLM_FRONTIER_TORCH_PROFILER_CAPTURE_DIR="$RUN_ROOT/replay/batch/runtime/batch/torch-profiler"
unset VLLM_FRONTIER_CUDA_PROFILER_CAPTURE VLLM_FRONTIER_MOE_BOUNDARY_LOG_PATH VLLM_FRONTIER_DIAG_MOE_AR_MODE
unset VLLM_FRONTIER_CUDA_EVENT_OP_LOG_PATH VLLM_FRONTIER_MOE_ROUTING_LOG_PATH
bash "$WORKERS/issue26_h800_replay_worker.sh" "$RUN_ROOT/replay"
"$PY" "$REPO_ROOT/tests/e2e/issue26_diagnostic_identity_analysis.py" \
  --run "$RUN_ROOT/replay/batch/runtime/batch" \
  --output "$RUN_ROOT/replay/batch/runtime/batch/identity_validation.json" --mode batch
test -n "$(find "$RUN_ROOT/replay/batch/runtime/batch/torch-profiler" -name 'forward*.json' -print -quit)"
echo TORCH_PROFILER_STANDARD_REPLAY_COMPLETE
