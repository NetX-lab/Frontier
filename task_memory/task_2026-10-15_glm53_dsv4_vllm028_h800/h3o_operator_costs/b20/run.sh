#!/usr/bin/env bash
set -euo pipefail
REPO=${1:?repo}
RUN_ROOT=${2:?scratch run root}
shift 2
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
B20_SCRATCH=${B20_SCRATCH:-/data/ycfeng/tmp/codex-implement/4bcc70f2-4440-43f7-9489-12ff832c399a/b20}
PYTHON_BIN=${PYTHON_BIN:-/data/ycfeng/envs/frontier-py310/bin/python}
mkdir -p "$B20_SCRATCH/tmp" "$RUN_ROOT"
cd "$REPO"
ulimit -v 62914560
export PYTHONPATH="$REPO" PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$B20_SCRATCH/tmp" FRONTIER_TMP_ROOT="$B20_SCRATCH/tmp"
export WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec timeout "${B20_TIMEOUT:-14400}s" "$PYTHON_BIN" -u "$SCRIPT_DIR/run_replay.py" "$REPO" "$RUN_ROOT" "$@"
