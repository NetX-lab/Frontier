#!/usr/bin/env bash
# Generate the B0 H800 golden profiling CSV prerequisites for operator parity.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/ycfeng/Frontier/worktrees/op-wide-refactor}"
TASK_DIR="${TASK_DIR:-/data/ycfeng/Frontier/task_memory/task_2026-07-01_op_wide_refactor}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/data/profiling}"
DEVICE="${DEVICE:-h800}"
NUM_GPUS="${NUM_GPUS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

TARGET="${TARGET:-/data/ycfeng/frontier_profiling_envs/issue2_py312_target_v2}"
CUDA_HOME="${CUDA_HOME:-/data/ycfeng/frontier_profiling_envs/cuda-12.4/cuda/cuda-12.4/cuda}"

MAX_SEQ_LEN="${MAX_SEQ_LEN:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-128}"
ATTENTION_BATCH_SIZES="${ATTENTION_BATCH_SIZES:-1 2 4 8}"
ATTENTION_DECODE_KV_SIZES="${ATTENTION_DECODE_KV_SIZES:-1 8 32 64 96}"
ATTENTION_CHUNK_SIZE="${ATTENTION_CHUNK_SIZE:-64}"
LINEAR_NUM_TOKENS="${LINEAR_NUM_TOKENS:-1 2 4 8 16 32 64 128}"
MOE_NUM_TOKENS="${MOE_NUM_TOKENS:-1 2 4 8 16 32 64}"
TP_SIZES="${TP_SIZES:-1 2 4 8}"
EP_SIZES="${EP_SIZES:-1 2 4 8}"
PROFILE_METHODS="${PROFILE_METHODS:-cuda_event record_function}"

LOG_DIR="${LOG_DIR:-$TASK_DIR/b0_h800_profiling_logs_$(date +%Y%m%d_%H%M%S)}"
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
read -r -a EP_ARGS <<< "$EP_SIZES"
read -r -a ATTENTION_BATCH_ARGS <<< "$ATTENTION_BATCH_SIZES"
read -r -a ATTENTION_DECODE_KV_ARGS <<< "$ATTENTION_DECODE_KV_SIZES"
read -r -a LINEAR_TOKEN_ARGS <<< "$LINEAR_NUM_TOKENS"
read -r -a MOE_TOKEN_ARGS <<< "$MOE_NUM_TOKENS"
read -r -a METHOD_ARGS <<< "$PROFILE_METHODS"

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

for method in "${METHOD_ARGS[@]}"; do
  method_slug="${method//[^A-Za-z0-9_]/_}"

  if [ "$method" = "record_function" ]; then
    for model in "${ALL_MODELS[@]}"; do
      model_slug="${model//[^A-Za-z0-9_]/_}"
      run_logged "attention_${method_slug}_${model_slug}" \
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
          --device "$DEVICE" \
          --profile_method "$method" \
          --output_dir "$OUTPUT_ROOT" \
          --yes
    done
  else
    run_logged "attention_${method_slug}" \
      "$PYTHON_BIN" -m frontier.profiling.attention.main \
        --disable_ray \
        --num_gpus "$NUM_GPUS" \
        --models "${ALL_MODELS[@]}" \
        --num_tensor_parallel_workers "${TP_ARGS[@]}" \
        --max_model_len "$MAX_MODEL_LEN" \
        --max_seq_len "$MAX_SEQ_LEN" \
        --min_batch_size 1 \
        --max_batch_size 8 \
        --batch_size_list "${ATTENTION_BATCH_ARGS[@]}" \
        --decode_kv_cache_size_list "${ATTENTION_DECODE_KV_ARGS[@]}" \
        --fixed_chunked_prefill_size "$ATTENTION_CHUNK_SIZE" \
        --enable_chunked_prefill_grid_search \
        --device "$DEVICE" \
        --profile_method "$method" \
        --output_dir "$OUTPUT_ROOT" \
        --yes
  fi

  if [ "$method" = "record_function" ]; then
    for model in "${DENSE_MODELS[@]}"; do
      model_slug="${model//[^A-Za-z0-9_]/_}"
      run_logged "linear_dense_${method_slug}_${model_slug}" \
        "$PYTHON_BIN" -m frontier.profiling.linear_op.main \
          --disable_ray \
          --num_gpus "$NUM_GPUS" \
          --models "$model" \
          --num_tensor_parallel_workers "${TP_ARGS[@]}" \
          --num_tokens_list "${LINEAR_TOKEN_ARGS[@]}" \
          --max_tokens 128 \
          --device "$DEVICE" \
          --profile_method "$method" \
          --output_dir "$OUTPUT_ROOT" \
          --yes
    done

    for model in "${MOE_MODELS[@]}"; do
      model_slug="${model//[^A-Za-z0-9_]/_}"
      run_logged "linear_moe_${method_slug}_${model_slug}" \
        "$PYTHON_BIN" -m frontier.profiling.linear_op.main \
          --disable_ray \
          --num_gpus "$NUM_GPUS" \
          --models "$model" \
          --num_tensor_parallel_workers "${TP_ARGS[@]}" \
          --num_tokens_list "${LINEAR_TOKEN_ARGS[@]}" \
          --max_tokens 128 \
          --is_moe \
          --device "$DEVICE" \
          --profile_method "$method" \
          --output_dir "$OUTPUT_ROOT" \
          --yes
    done

    for model in "${MOE_MODELS[@]}"; do
      model_slug="${model//[^A-Za-z0-9_]/_}"
      run_logged "moe_${method_slug}_${model_slug}" \
        "$PYTHON_BIN" -m frontier.profiling.moe.main \
          --disable_ray \
          --num_gpus "$NUM_GPUS" \
          --models "$model" \
          --num_tensor_parallel_workers "${TP_ARGS[@]}" \
          --expert_parallel_sizes "${EP_ARGS[@]}" \
          --num_tokens_list "${MOE_TOKEN_ARGS[@]}" \
          --max_tokens 64 \
          --enable_load_imbalance \
          --load_distributions uniform \
          --num_samples_per_distribution 1 \
          --routing_runtime_path uniform_topk \
          --gating_runtime_context prefill_hot \
          --device "$DEVICE" \
          --profile_method "$method" \
          --output_dir "$OUTPUT_ROOT" \
          --yes
    done
  else
    run_logged "linear_dense_${method_slug}" \
      "$PYTHON_BIN" -m frontier.profiling.linear_op.main \
        --disable_ray \
        --num_gpus "$NUM_GPUS" \
        --models "${DENSE_MODELS[@]}" \
        --num_tensor_parallel_workers "${TP_ARGS[@]}" \
        --num_tokens_list "${LINEAR_TOKEN_ARGS[@]}" \
        --max_tokens 128 \
        --device "$DEVICE" \
        --profile_method "$method" \
        --output_dir "$OUTPUT_ROOT" \
        --yes

    run_logged "linear_moe_${method_slug}" \
      "$PYTHON_BIN" -m frontier.profiling.linear_op.main \
        --disable_ray \
        --num_gpus "$NUM_GPUS" \
        --models "${MOE_MODELS[@]}" \
        --num_tensor_parallel_workers "${TP_ARGS[@]}" \
        --num_tokens_list "${LINEAR_TOKEN_ARGS[@]}" \
        --max_tokens 128 \
        --is_moe \
        --device "$DEVICE" \
        --profile_method "$method" \
        --output_dir "$OUTPUT_ROOT" \
        --yes

    run_logged "moe_${method_slug}" \
      "$PYTHON_BIN" -m frontier.profiling.moe.main \
        --disable_ray \
        --num_gpus "$NUM_GPUS" \
        --models "${MOE_MODELS[@]}" \
        --num_tensor_parallel_workers "${TP_ARGS[@]}" \
        --expert_parallel_sizes "${EP_ARGS[@]}" \
        --num_tokens_list "${MOE_TOKEN_ARGS[@]}" \
        --max_tokens 64 \
        --enable_load_imbalance \
        --load_distributions uniform \
        --num_samples_per_distribution 1 \
        --routing_runtime_path uniform_topk \
        --gating_runtime_context prefill_hot \
        --device "$DEVICE" \
        --profile_method "$method" \
        --output_dir "$OUTPUT_ROOT" \
        --yes
  fi
done

run_logged "99_manifest" \
  "$PYTHON_BIN" - <<'PY'
import csv
import hashlib
import json
import math
from pathlib import Path

repo = Path("/data/ycfeng/Frontier/worktrees/op-wide-refactor")
root = repo / "data/profiling/compute/h800"
models = {
    "llama2_7b_dense_example": ["attention.csv", "attention_kernel_only.csv", "linear_op.csv", "linear_op_kernel_only.csv"],
    "Phi-tiny-MoE-instruct": ["attention.csv", "attention_kernel_only.csv", "linear_op.csv", "linear_op_kernel_only.csv", "moe.csv", "moe_kernel_only.csv"],
    "Step2Mini-tiny": ["attention.csv", "attention_kernel_only.csv", "linear_op.csv", "linear_op_kernel_only.csv", "moe.csv", "moe_kernel_only.csv"],
    "step-moe-noquant-small": ["attention.csv", "attention_kernel_only.csv", "linear_op.csv", "linear_op_kernel_only.csv", "moe.csv", "moe_kernel_only.csv"],
    "Qwen3-30B-A3B-tiny": ["attention.csv", "attention_kernel_only.csv", "linear_op.csv", "linear_op_kernel_only.csv", "moe.csv", "moe_kernel_only.csv"],
    "qwen3-next-80b-a3b-instruct-reduced-l2": ["attention.csv", "attention_kernel_only.csv", "linear_op.csv", "linear_op_kernel_only.csv", "moe.csv", "moe_kernel_only.csv"],
}

def audit_csv(path: Path) -> dict:
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    text = data.decode("utf-8")
    reader = csv.DictReader(text.splitlines())
    fields = reader.fieldnames or []
    time_cols = [c for c in fields if c.startswith("time_stats.")]
    row_count = 0
    invalid = 0
    valid = 0
    empty_row_count = 0
    valid_by_column = {col: 0 for col in time_cols}
    measurement_types = set()
    for row in reader:
        row_count += 1
        row_valid = 0
        if row.get("measurement_type"):
            measurement_types.add(row["measurement_type"])
        for col in time_cols:
            raw = str(row.get(col, "")).strip().lower()
            if raw in {"", "nan", "none", "null"}:
                invalid += 1
                continue
            valid += 1
            row_valid += 1
            valid_by_column[col] += 1
        if time_cols and row_valid == 0:
            empty_row_count += 1
    return {
        "path": str(path),
        "sha256": sha,
        "row_count": row_count,
        "time_stats_columns": len(time_cols),
        "time_stats_valid_count": valid,
        "time_stats_sparse_nan_count": invalid,
        "time_stats_empty_row_count": empty_row_count,
        "time_stats_empty_column_count": sum(
            1 for count in valid_by_column.values() if count == 0
        ),
        "measurement_type_values": sorted(measurement_types),
    }

manifest = {"root": str(root), "models": {}}
for model, filenames in models.items():
    model_rows = {}
    for filename in filenames:
        path = root / model / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = audit_csv(path)
        if stat["row_count"] <= 0:
            raise ValueError(f"Empty profiling CSV: {path}")
        if stat["time_stats_columns"] <= 0:
            raise ValueError(f"Profiling CSV has no time_stats columns: {path}")
        if stat["time_stats_valid_count"] <= 0:
            raise ValueError(f"Profiling CSV has no valid time_stats values: {path}")
        if stat["time_stats_empty_row_count"] > 0:
            raise ValueError(
                f"Profiling CSV has rows with no valid time_stats values: "
                f"{path}: {stat['time_stats_empty_row_count']}"
            )
        if stat["time_stats_empty_column_count"] > 0:
            raise ValueError(
                f"Profiling CSV has time_stats columns with no valid values: "
                f"{path}: {stat['time_stats_empty_column_count']}"
            )
        model_rows[filename] = stat
    manifest["models"][model] = model_rows
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

echo "B0 H800 profiling completed. Logs: $LOG_DIR"
