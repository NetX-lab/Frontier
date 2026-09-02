#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/data/ycfeng/frontier_profiling_envs/issue2_py312_target_v2}"
CUDA_ROOT="${CUDA_ROOT:-/data/ycfeng/cu128_build/cuda-12.8}"
MODEL="${MODEL:-step3-moe-noquant}"
PROFILE_ROOT="${PROFILE_ROOT:-/data/ycfeng/tmp/frontier_profiling_staging_20260823/h200/formal_step3_moe_noquant_2ecc496b_20260824_retry_localcache_1807/step3-moe-noquant/accepted}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/ycfeng/tmp/frontier_profiling_staging_20260823/h200/e2e_step3_a_20260825/runs}"
REPORT_ROOT="${REPORT_ROOT:-/data/ycfeng/tmp/frontier_profiling_staging_20260823/h200/e2e_step3_a_20260825/reports}"
CACHE_ROOT="${CACHE_ROOT:-/data/ycfeng/tmp/frontier_profiling_staging_20260823/h200/jit_cache_step3_a_20260825}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-step3_h200_a_20260825}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$PATH"
export PYTHONPATH="$ENV_ROOT:$REPO_ROOT"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TMPDIR=/data/ycfeng/tmp
export TEMP=/data/ycfeng/tmp
export TMP=/data/ycfeng/tmp
export SQLITE_TMPDIR=/data/ycfeng/tmp
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export FLASHINFER_WORKSPACE_BASE="$CACHE_ROOT/flashinfer_workspace"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export PYTHONDONTWRITEBYTECODE=1
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="$REPO_ROOT"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:$ENV_ROOT/nvidia/cuda_runtime/lib:$ENV_ROOT/nvidia/cuda_nvrtc/lib:$ENV_ROOT/nvidia/cublas/lib:$ENV_ROOT/nvidia/cudnn/lib:$ENV_ROOT/nvidia/cufft/lib:$ENV_ROOT/nvidia/curand/lib:$ENV_ROOT/nvidia/cusolver/lib:$ENV_ROOT/nvidia/cuda_cupti/lib:$ENV_ROOT/nvidia/cufile/lib:$ENV_ROOT/nvidia/cusparselt/lib:$ENV_ROOT/nvidia:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUTPUT_ROOT" "$REPORT_ROOT" "$XDG_CACHE_HOME" \
  "$FLASHINFER_WORKSPACE_BASE" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
cd "$REPO_ROOT"

{
  echo "task_date=2026-08-25"
  echo "worker_start=$(date -Is)"
  echo "host=$(hostname)"
  echo "repo=$REPO_ROOT"
  echo "head=$(git rev-parse HEAD)"
  echo "manifest=$(sha256sum task_memory/task_2026-08-23_profiling_dataset_governance/h200_exact_manifest_frozen_v3.json)"
  echo "python=$(command -v "$PYTHON_BIN")"
  "$PYTHON_BIN" --version
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
} > "$REPORT_ROOT/environment.log" 2>&1

PROFILE_ROOT="$PROFILE_ROOT" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
REPORT_ROOT="$REPORT_ROOT" \
RUN_ID_PREFIX="$RUN_ID_PREFIX" \
PYTHON_BIN="$PYTHON_BIN" \
bash tests/performance/profiling/run_h200_six_model_non_dummy_e2e.sh \
  --model "$MODEL" > "$REPORT_ROOT/launcher.log" 2>&1

printf 'exit_code=0\nworker_end=%s\n' "$(date -Is)" > "$REPORT_ROOT/launcher_status.txt"
