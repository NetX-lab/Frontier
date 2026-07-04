#!/usr/bin/env bash
# Generate staged H800 true-mixed attention profiling rows for operator parity.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/ycfeng/Frontier/worktrees/op-wide-refactor}"
TASK_DIR="${TASK_DIR:-/data/ycfeng/Frontier/task_memory/task_2026-07-01_op_wide_refactor}"
STAGE_ROOT="${STAGE_ROOT:-$TASK_DIR/h800_true_mixed_attention_stage_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$STAGE_ROOT/profiling}"
PREWARM_OUTPUT_ROOT="${PREWARM_OUTPUT_ROOT:-$STAGE_ROOT/prewarm/profiling}"
DEVICE="${DEVICE:-h800}"
NUM_GPUS="${NUM_GPUS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

TARGET="${TARGET:-/data/ycfeng/frontier_profiling_envs/issue2_py312_target_v2}"
CUDA_HOME="${CUDA_HOME:-/data/ycfeng/frontier_profiling_envs/cuda-12.4/cuda/cuda-12.4/cuda}"

MAX_SEQ_LEN="${MAX_SEQ_LEN:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-128}"
PREWARM_MAX_SEQ_LEN="${PREWARM_MAX_SEQ_LEN:-64}"
PREWARM_MAX_MODEL_LEN="${PREWARM_MAX_MODEL_LEN:-64}"
PREWARM_CHUNK_SIZE="${PREWARM_CHUNK_SIZE:-16}"
TP_SIZES="${TP_SIZES:-1 2 4 8}"
PROFILE_METHODS="${PROFILE_METHODS:-cuda_event record_function}"
ATTENTION_BATCH_SIZES="${ATTENTION_BATCH_SIZES:-1 2 4 8}"
ATTENTION_DECODE_KV_SIZES="${ATTENTION_DECODE_KV_SIZES:-1 8 16 32 64 96}"
ATTENTION_CHUNK_SIZE="${ATTENTION_CHUNK_SIZE:-16}"
TRUE_MIXED_PREFILL_BATCH_SIZES="${TRUE_MIXED_PREFILL_BATCH_SIZES:-1 2}"
TRUE_MIXED_PREFILL_CHUNK_SIZES="${TRUE_MIXED_PREFILL_CHUNK_SIZES:-8 16 32 64}"
TRUE_MIXED_DECODE_BATCH_SIZES="${TRUE_MIXED_DECODE_BATCH_SIZES:-1 2}"
TRUE_MIXED_DECODE_KV_CACHE_SIZES="${TRUE_MIXED_DECODE_KV_CACHE_SIZES:-1 8 16 32 64 96}"
TRUE_MIXED_PREFILL_KV_CACHE_SIZE="${TRUE_MIXED_PREFILL_KV_CACHE_SIZE:-0}"

LOG_DIR="${LOG_DIR:-$STAGE_ROOT/logs}"
mkdir -p "$LOG_DIR"

if [ ! -d "$REPO_ROOT" ]; then
  echo "ERROR: REPO_ROOT does not exist: $REPO_ROOT" >&2
  exit 2
fi

if [ ! -d "$TARGET" ]; then
  echo "ERROR: profiling TARGET does not exist: $TARGET" >&2
  exit 2
fi

if [ ! -d "$CUDA_HOME" ]; then
  echo "ERROR: CUDA_HOME does not exist: $CUDA_HOME" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$TARGET/nvidia/cuda_nvcc/bin:$PATH"
export PYTHONPATH="$TARGET:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$TARGET/nvidia/cuda_runtime/lib:$TARGET/nvidia/cublas/lib:$TARGET/nvidia/cuda_nvrtc/lib:$TARGET/nvidia/cudnn/lib:$TARGET/nvidia/cufft/lib:$TARGET/nvidia/curand/lib:$TARGET/nvidia/cusolver/lib:$TARGET/nvidia/cusparse/lib:$TARGET/nvidia/cusparselt/lib:$TARGET/nvidia/nccl/lib:$TARGET/nvidia/nvjitlink/lib:$TARGET/nvidia/nvtx/lib:$TARGET/nvidia/cufile/lib:$TARGET/torch/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export WANDB_DISABLED=true
export VIDUR_DISABLE_WANDB=1
export KINETO_LOG_LEVEL=5
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$STAGE_ROOT/flashinfer_workspace}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-1}"
export MAX_JOBS="${MAX_JOBS:-1}"
export OUTPUT_ROOT
export PREWARM_OUTPUT_ROOT
export DEVICE
export PROFILE_METHODS

run_logged() {
  local name="$1"
  shift
  local log_file="$LOG_DIR/${name}.log"
  echo "===== RUN $name =====" | tee "$log_file"
  printf 'COMMAND:' | tee -a "$log_file"
  printf ' %q' "$@" | tee -a "$log_file"
  printf '\n' | tee -a "$log_file"
  "$@" 2>&1 | tee -a "$log_file"
}

read -r -a DENSE_MODELS <<< "llama2_7b_dense_example"
read -r -a MOE_MODELS <<< "Phi-tiny-MoE-instruct Step2Mini-tiny step-moe-noquant-small Qwen3-30B-A3B-tiny qwen3-next-80b-a3b-instruct-reduced-l2"
read -r -a ALL_MODELS <<< "${DENSE_MODELS[*]} ${MOE_MODELS[*]}"
read -r -a TP_ARGS <<< "$TP_SIZES"
read -r -a METHOD_ARGS <<< "$PROFILE_METHODS"
read -r -a ATTENTION_BATCH_ARGS <<< "$ATTENTION_BATCH_SIZES"
read -r -a ATTENTION_DECODE_KV_ARGS <<< "$ATTENTION_DECODE_KV_SIZES"
read -r -a TRUE_MIXED_PREFILL_BATCH_ARGS <<< "$TRUE_MIXED_PREFILL_BATCH_SIZES"
read -r -a TRUE_MIXED_PREFILL_CHUNK_ARGS <<< "$TRUE_MIXED_PREFILL_CHUNK_SIZES"
read -r -a TRUE_MIXED_DECODE_BATCH_ARGS <<< "$TRUE_MIXED_DECODE_BATCH_SIZES"
read -r -a TRUE_MIXED_DECODE_KV_ARGS <<< "$TRUE_MIXED_DECODE_KV_CACHE_SIZES"

cd "$REPO_ROOT"

run_logged "00_environment" \
  "$PYTHON_BIN" - <<'PY'
import importlib
import os
import subprocess
import sys

print("python", sys.executable, sys.version.split()[0])
print("cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("cuda_home", os.environ.get("CUDA_HOME"))
print("pythonpath_head", os.environ.get("PYTHONPATH", "").split(":")[:3])
for module_name in ("torch", "vllm", "flashinfer", "triton", "pandas"):
    module = importlib.import_module(module_name)
    print(module_name, getattr(module, "__version__", "unknown"))
import torch
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.device_count() < 1:
    raise SystemExit("No CUDA devices visible")
subprocess.run(["nvidia-smi", "-L"], check=True)
PY

run_logged "01_flashinfer_page_jit_prewarm" \
  "$PYTHON_BIN" - <<'PY'
import os

import torch

torch.cuda.set_device(0)

import flashinfer
from flashinfer.jit import env as jit_env
from flashinfer.page import get_page_module

print("flashinfer", getattr(flashinfer, "__version__", "unknown"))
print("flashinfer_workspace_base", os.environ.get("FLASHINFER_WORKSPACE_BASE"))
print("flashinfer_workspace_dir", jit_env.FLASHINFER_WORKSPACE_DIR)
print("flashinfer_jit_dir", jit_env.FLASHINFER_JIT_DIR)
print("flashinfer_nvcc_threads", os.environ.get("FLASHINFER_NVCC_THREADS"))
print("max_jobs", os.environ.get("MAX_JOBS"))
get_page_module()
print("page_jit_prewarm_ok")
PY

run_attention_jit_prewarm() {
  local method="$1"
  local model="$2"
  local method_slug="${method//[^A-Za-z0-9_]/_}"
  local model_slug="${model//[^A-Za-z0-9_]/_}"

  run_logged "02_flashinfer_attention_jit_prewarm_${method_slug}_${model_slug}" \
    "$PYTHON_BIN" -m frontier.profiling.attention.main \
      --disable_ray \
      --num_gpus 1 \
      --models "$model" \
      --num_tensor_parallel_workers "${TP_ARGS[@]}" \
      --max_model_len "$PREWARM_MAX_MODEL_LEN" \
      --max_seq_len "$PREWARM_MAX_SEQ_LEN" \
      --min_batch_size 1 \
      --max_batch_size 1 \
      --batch_size_list 1 \
      --fixed_chunked_prefill_size "$PREWARM_CHUNK_SIZE" \
      --enable_chunked_prefill_grid_search \
      --profile_only_prefill \
      --device "$DEVICE" \
      --profile_method "$method" \
      --output_dir "$PREWARM_OUTPUT_ROOT" \
      --yes
}

for method in "${METHOD_ARGS[@]}"; do
  method_slug="${method//[^A-Za-z0-9_]/_}"
  for model in "${ALL_MODELS[@]}"; do
    model_slug="${model//[^A-Za-z0-9_]/_}"
    run_attention_jit_prewarm "$method" "$model"
    run_logged "attention_true_mixed_${method_slug}_${model_slug}" \
      "$PYTHON_BIN" -m frontier.profiling.attention.main \
        --disable_ray \
        --num_gpus "$NUM_GPUS" \
        --models "$model" \
        --num_tensor_parallel_workers "${TP_ARGS[@]}" \
        --max_model_len "$MAX_MODEL_LEN" \
        --max_seq_len "$MAX_SEQ_LEN" \
        --min_batch_size 1 \
        --max_batch_size 8 \
        --batch_size_list "${ATTENTION_BATCH_ARGS[@]}" \
        --decode_kv_cache_size_list "${ATTENTION_DECODE_KV_ARGS[@]}" \
        --fixed_chunked_prefill_size "$ATTENTION_CHUNK_SIZE" \
        --enable_chunked_prefill_grid_search \
        --enable_true_mixed \
        --true_mixed_prefill_batch_sizes "${TRUE_MIXED_PREFILL_BATCH_ARGS[@]}" \
        --true_mixed_prefill_chunk_sizes "${TRUE_MIXED_PREFILL_CHUNK_ARGS[@]}" \
        --true_mixed_decode_batch_sizes "${TRUE_MIXED_DECODE_BATCH_ARGS[@]}" \
        --true_mixed_decode_kv_cache_sizes "${TRUE_MIXED_DECODE_KV_ARGS[@]}" \
        --true_mixed_prefill_kv_cache_size "$TRUE_MIXED_PREFILL_KV_CACHE_SIZE" \
        --device "$DEVICE" \
        --profile_method "$method" \
        --output_dir "$OUTPUT_ROOT" \
        --yes
  done
done

run_logged "99_stage_manifest" \
  "$PYTHON_BIN" - <<'PY'
import csv
import hashlib
import json
import math
import os
from pathlib import Path

stage_root = Path(os.environ["OUTPUT_ROOT"]) / "compute" / os.environ["DEVICE"]
models = [
    "llama2_7b_dense_example",
    "Phi-tiny-MoE-instruct",
    "Step2Mini-tiny",
    "step-moe-noquant-small",
    "Qwen3-30B-A3B-tiny",
    "qwen3-next-80b-a3b-instruct-reduced-l2",
]
required_numeric_columns = (
    "batch_composition_ratio",
    "decode_avg_kv_cache_size",
    "decode_batch_size",
    "num_prefill_seqs",
    "total_batch_size",
    "total_prefill_tokens",
    "total_tokens",
)

def valid_number(value: object) -> bool:
    if str(value).strip().lower() in {"", "nan", "none", "null"}:
        return False
    try:
        return math.isfinite(float(str(value).strip()))
    except (TypeError, ValueError):
        return False

def audit_csv(path: Path) -> dict:
    data = path.read_bytes()
    reader = csv.DictReader(data.decode("utf-8").splitlines())
    fields = reader.fieldnames or []
    rows = list(reader)
    true_rows = [
        row for row in rows
        if str(row.get("is_true_mixed_batch", "")).strip().lower()
        in {"1", "true", "t", "yes", "y"}
    ]
    valid_decode = [
        row for row in true_rows
        if valid_number(row.get("time_stats.attn_decode.median", ""))
    ]
    invalid_numeric_columns = {}
    required_numeric_valid_rows = 0
    required_numeric_invalid_cells = 0
    for row in true_rows:
        row_valid = True
        for column in required_numeric_columns:
            if valid_number(row.get(column, "")):
                continue
            row_valid = False
            required_numeric_invalid_cells += 1
            invalid_numeric_columns[column] = invalid_numeric_columns.get(column, 0) + 1
        if row_valid:
            required_numeric_valid_rows += 1
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "row_count": len(rows),
        "column_count": len(fields),
        "true_mixed_row_count": len(true_rows),
        "true_mixed_attn_decode_valid_count": len(valid_decode),
        "true_mixed_required_numeric_valid_row_count": required_numeric_valid_rows,
        "true_mixed_required_numeric_invalid_cell_count": required_numeric_invalid_cells,
        "true_mixed_required_numeric_invalid_columns": invalid_numeric_columns,
    }

profile_methods = os.environ["PROFILE_METHODS"].split()
expected_files = []
for method in profile_methods:
    normalized = method.strip().lower()
    if normalized in {"record_function", "kernel_only"}:
        expected_files.extend(
            ("attention_true_mixed_kernel_only.csv", "attention_combined_kernel_only.csv")
        )
    elif normalized in {"cuda_event", "cuda"}:
        expected_files.extend(("attention_true_mixed.csv", "attention_combined.csv"))
    else:
        raise ValueError(f"Unsupported profile method in manifest audit: {method}")

manifest = {
    "stage_root": str(stage_root),
    "expected_files": expected_files,
    "models": {},
}
for model in models:
    model_stats = {}
    for filename in expected_files:
        path = stage_root / model / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = audit_csv(path)
        if stat["true_mixed_row_count"] <= 0:
            raise ValueError(f"No true mixed rows in {path}")
        if stat["true_mixed_attn_decode_valid_count"] != stat["true_mixed_row_count"]:
            raise ValueError(
                f"Invalid true mixed attn decode timings in {path}: "
                f"{stat['true_mixed_attn_decode_valid_count']}/"
                f"{stat['true_mixed_row_count']} rows valid"
            )
        if stat["true_mixed_required_numeric_valid_row_count"] != stat["true_mixed_row_count"]:
            raise ValueError(
                f"Invalid true mixed numeric feature values in {path}: "
                f"{stat['true_mixed_required_numeric_valid_row_count']}/"
                f"{stat['true_mixed_row_count']} rows valid, "
                f"invalid_columns={stat['true_mixed_required_numeric_invalid_columns']}"
            )
        model_stats[filename] = stat
    manifest["models"][model] = model_stats
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

echo "True-mixed H800 attention profiling staged at: $STAGE_ROOT"
