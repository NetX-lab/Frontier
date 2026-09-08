#!/usr/bin/env bash
# Run the approved clean replay and isolated batch diagnostics on one H200 pod.
set -euo pipefail
REPLAY_ROOT="${1:?Provide a fresh run parent directory.}"
WORKERS="$(cd "$(dirname "$0")" && pwd)"
export VLLM_MOE_UNIFORM_ROUTING=1
bash "$WORKERS/issue26_h200_uniform_groundtruth_worker.sh" "$REPLAY_ROOT/clean"
bash "$WORKERS/issue26_h200_diagnostics_worker.sh" "$REPLAY_ROOT/batch/runtime" batch
echo HISTORICAL_CONTROL_REPLAY_EXECUTION_COMPLETE
