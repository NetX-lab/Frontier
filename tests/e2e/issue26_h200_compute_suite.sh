#!/usr/bin/env bash
# Bracket low-density operator probes with the same batch-only control.
set -euo pipefail
OUT="${1:?Provide a fresh suite output directory.}"
: "${ISSUE26_DIAGNOSTIC_VLLM_COMMIT:?Pin the reviewed diagnostic checkout.}"
test ! -e "$OUT"
mkdir -p "$OUT"
export VLLM_MOE_UNIFORM_ROUTING=1
WORKER="$(dirname "$0")/issue26_h200_diagnostics_worker.sh"
for MODE in batch_before compute_attention compute_moe compute_detail batch_after; do
  SELECTION="$MODE"
  if [[ "$MODE" == batch_* ]]; then SELECTION=batch; fi
  bash "$WORKER" "$OUT/$MODE" "$SELECTION"
done
echo COMPUTE_SUITE_EXECUTION_COMPLETE
