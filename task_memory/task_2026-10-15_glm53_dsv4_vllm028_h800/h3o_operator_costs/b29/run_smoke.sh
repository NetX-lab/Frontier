#!/usr/bin/env bash
set -euo pipefail
ulimit -v 31457280
export PYTHON_BIN=/data/ycfeng/envs/frontier-py310/bin/python
export PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/data/ycfeng/Frontier/.worktrees/b29-ingress-delays
export TMPDIR=/data/ycfeng/tmp/codex-implement/6d8aed77-5e60-463c-87d4-b0f7a492b026/b29/smoke/tmp FRONTIER_TMP_ROOT=/data/ycfeng/tmp/codex-implement/6d8aed77-5e60-463c-87d4-b0f7a492b026/b29/smoke/tmp
export METRICS_OUTPUT_DIR=/data/ycfeng/tmp/codex-implement/6d8aed77-5e60-463c-87d4-b0f7a492b026/b29/smoke/metrics RUN_ID_PREFIX=b29_dummy_smoke
mkdir -p /data/ycfeng/tmp/codex-implement/6d8aed77-5e60-463c-87d4-b0f7a492b026/b29/smoke/tmp
ln -s /data/ycfeng/Frontier/.worktrees/b29-ingress-delays/data /data/ycfeng/tmp/codex-implement/6d8aed77-5e60-463c-87d4-b0f7a492b026/b29/smoke/data
cd /data/ycfeng/tmp/codex-implement/6d8aed77-5e60-463c-87d4-b0f7a492b026/b29/smoke
timeout 1800 bash /data/ycfeng/Frontier/.worktrees/b29-ingress-delays/examples/architecture/run_dummy_smoke_matrix.sh -- \
  --metrics_config_cache_dir /data/ycfeng/tmp/codex-implement/6d8aed77-5e60-463c-87d4-b0f7a492b026/b29/smoke/cache \
  --measured_cc_backend_config_cache_dir /data/ycfeng/tmp/codex-implement/6d8aed77-5e60-463c-87d4-b0f7a492b026/b29/smoke/cc_cache
