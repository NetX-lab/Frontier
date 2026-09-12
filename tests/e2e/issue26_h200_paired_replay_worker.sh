#!/usr/bin/env bash
# Run two opposite-order normal/skip pairs within one H200 allocation.
set -euo pipefail
RUN_ROOT="${1:?Provide a fresh run directory.}"
WORKERS="$(cd "$(dirname "$0")" && pwd)"
PY=/local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python
mkdir -p "$RUN_ROOT"
exec > >(tee "$RUN_ROOT/worker.log") 2>&1
trap 'code=$?; printf "worker_exit_code=%s\n" "$code" > "$RUN_ROOT/worker_exit.txt"' EXIT
hostname > "$RUN_ROOT/node.txt"
export VLLM_MOE_UNIFORM_ROUTING=1
export ISSUE26_WARMUPS="${ISSUE26_WARMUPS:-10}"
unset VLLM_FRONTIER_MOE_BOUNDARY_LOG_PATH VLLM_FRONTIER_DIAG_MOE_AR_MODE
unset VLLM_FRONTIER_CUDA_PROFILER_CAPTURE VLLM_FRONTIER_TORCH_PROFILER_CAPTURE
unset VLLM_FRONTIER_CUDA_EVENT_OP_LOG_PATH VLLM_FRONTIER_MOE_ROUTING_LOG_PATH
for arm in normal_1 skip_1 skip_2 normal_2; do
  case "$arm" in
    normal_*)
      export ISSUE26_DIAGNOSTIC_VLLM_SOURCE=/data/ycfeng/tmp/issue26-vllm-diagnostics-ar-ab
      export ISSUE26_DIAGNOSTIC_VLLM_COMMIT=0d633a946e6600c77a251bef8b5553ec7f43f7e7 ;;
    skip_*)
      export ISSUE26_DIAGNOSTIC_VLLM_SOURCE=/data/ycfeng/tmp/issue26-vllm-post-moe-bypass-20260910
      export ISSUE26_DIAGNOSTIC_VLLM_COMMIT=e29a8f925216d517bcbf7ffad47b93970d72d1f5 ;;
  esac
  printf 'ARM_START %s %s\n' "$arm" "$(date -u +%FT%TZ)"
  bash "$WORKERS/issue26_h200_replay_worker.sh" "$RUN_ROOT/$arm"
  "$PY" "$WORKERS/issue26_diagnostic_identity_analysis.py" \
    --run "$RUN_ROOT/$arm/batch/runtime/batch" \
    --output "$RUN_ROOT/$arm/identity_validation.json" --mode batch \
    --warmups "$ISSUE26_WARMUPS" \
    > "$RUN_ROOT/$arm/identity_validation.log" 2>&1
  "$PY" "$WORKERS/issue26_standard_bypass_result.py" \
    --run "$RUN_ROOT/$arm" --source "$ISSUE26_DIAGNOSTIC_VLLM_SOURCE" \
    --no-historical-reference --warmups "$ISSUE26_WARMUPS" --output "$RUN_ROOT/$arm/standard_result.json"
  printf 'ARM_COMPLETE %s %s\n' "$arm" "$(date -u +%FT%TZ)"
done
