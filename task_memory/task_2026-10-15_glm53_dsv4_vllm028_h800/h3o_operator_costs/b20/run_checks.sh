#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${B20_REPO:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}
B20_SCRATCH=${B20_SCRATCH:-/data/ycfeng/tmp/codex-implement/4bcc70f2-4440-43f7-9489-12ff832c399a/b20}
PYTHON_BIN=${PYTHON_BIN:-/data/ycfeng/envs/frontier-py310/bin/python}
mkdir -p "$B20_SCRATCH/tmp" "$B20_SCRATCH/checks-cwd"
if [ ! -e "$B20_SCRATCH/checks-cwd/data" ]; then
  ln -s "$REPO/data" "$B20_SCRATCH/checks-cwd/data"
fi
cd "$B20_SCRATCH/checks-cwd"
ulimit -v 62914560
export PYTHONPATH="$REPO:$REPO/tests/unit:$REPO/tests/comparison/calibration:$REPO/tests/analysis"
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$B20_SCRATCH/tmp" FRONTIER_TMP_ROOT="$B20_SCRATCH/tmp"
export WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "$#" -gt 0 ]; then
  exec timeout 3600s "$PYTHON_BIN" -m pytest -q -p no:cacheprovider --basetemp="${B20_TEST_TMP:-$B20_SCRATCH/final-check-tmp}" "$@"
fi
exec timeout 3600s "$PYTHON_BIN" -m pytest -q -p no:cacheprovider --basetemp="$B20_SCRATCH/regression-tmp" \
  "$REPO/tests/unit/test_v4_prediction_memo.py" \
  "$REPO/tests/unit/test_dsv4_operator_consumer.py" "$REPO/tests/unit/test_dsv4_runner_consumer.py" \
  "$REPO/tests/unit/test_dsv4_runner_ledger.py" "$REPO/tests/unit/test_dsv4_kv_block_table_writes.py" \
  "$REPO/tests/unit/test_dsv4_predictor.py" "$REPO/tests/unit/test_v4_attention_coverage.py" \
  "$REPO/tests/unit/test_balanced_routing.py" "$REPO/tests/unit/test_moe_rank_plan.py" "$REPO/tests/unit/test_moe_rank_routing.py" \
  "$REPO/tests/unit/test_dsv4_dp_sync.py" "$REPO/tests/unit/test_dsv4_structural_pd.py" "$REPO/tests/unit/test_dsv4_structural_execution.py" \
  "$REPO/tests/unit/test_dsv4_source_execution.py" "$REPO/tests/unit/test_dsv4_attention_runtime.py" \
  "$REPO/tests/unit/test_dsv4_attention_workload.py" "$REPO/tests/unit/test_dsv4_features.py" \
  "$REPO/tests/unit/test_dsv4_graph_dispatch.py" "$REPO/tests/unit/test_dsv4_a10_tools.py"
