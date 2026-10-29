#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${B20_REPO:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}
B20_SCRATCH=${B20_SCRATCH:-/data/ycfeng/tmp/codex-implement/4bcc70f2-4440-43f7-9489-12ff832c399a/b20}
export PYTHON_BIN=${PYTHON_BIN:-/data/ycfeng/envs/frontier-py310/bin/python}
export PYTHONPATH="$REPO" PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$B20_SCRATCH/tmp" FRONTIER_TMP_ROOT="$B20_SCRATCH/tmp"
export WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export METRICS_OUTPUT_DIR="$B20_SCRATCH/smoke/metrics" RUN_ID_PREFIX=b20_dummy_smoke
mkdir -p "$TMPDIR" "$B20_SCRATCH/smoke"
cd "$REPO"
ulimit -v 62914560
exec timeout 1800s bash examples/architecture/run_dummy_smoke_matrix.sh -- \
    --metrics_config_cache_dir "$B20_SCRATCH/smoke/cache" \
    --measured_cc_backend_config_cache_dir "$B20_SCRATCH/smoke/cc_cache"
