#!/usr/bin/env bash
# Generate staged H800 MoE gating-context profiling rows for B0 operator parity.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/ycfeng/Frontier/worktrees/op-wide-refactor}"
TASK_DIR="${TASK_DIR:-/data/ycfeng/Frontier/task_memory/task_2026-07-01_op_wide_refactor}"
STAGE_ROOT="${STAGE_ROOT:-$TASK_DIR/b0_h800_standalone_gating_stage_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$STAGE_ROOT/profiling}"
DEVICE="${DEVICE:-h800}"
NUM_GPUS="${NUM_GPUS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

TARGET="${TARGET:-/data/ycfeng/frontier_profiling_envs/issue2_py312_target_v2}"
CUDA_HOME="${CUDA_HOME:-/data/ycfeng/frontier_profiling_envs/cuda-12.4/cuda/cuda-12.4/cuda}"

MOE_GATING_CONTEXT="${MOE_GATING_CONTEXT:-standalone_legacy}"
MOE_NUM_TOKENS="${MOE_NUM_TOKENS:-1 2 4 8 16 32 64}"
TP_SIZES="${TP_SIZES:-1 2 4 8}"
EP_SIZES="${EP_SIZES:-1 2 4 8}"
PROFILE_METHODS="${PROFILE_METHODS:-cuda_event record_function}"

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
export OUTPUT_ROOT
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

read -r -a MOE_MODELS <<< "Phi-tiny-MoE-instruct Step2Mini-tiny step-moe-noquant-small Qwen3-30B-A3B-tiny qwen3-next-80b-a3b-instruct-reduced-l2"
read -r -a TP_ARGS <<< "$TP_SIZES"
read -r -a EP_ARGS <<< "$EP_SIZES"
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
  for model in "${MOE_MODELS[@]}"; do
    model_slug="${model//[^A-Za-z0-9_]/_}"
    run_logged "moe_${method_slug}_${MOE_GATING_CONTEXT}_${model_slug}" \
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
        --gating_runtime_context "$MOE_GATING_CONTEXT" \
        --device "$DEVICE" \
        --profile_method "$method" \
        --output_dir "$OUTPUT_ROOT" \
        --yes
  done
done

run_logged "99_manifest" \
  "$PYTHON_BIN" - <<'PY'
import csv
import hashlib
import json
from pathlib import Path

stage_root = Path(__import__("os").environ["OUTPUT_ROOT"]) / "compute" / "h800"
models = [
    "Phi-tiny-MoE-instruct",
    "Step2Mini-tiny",
    "step-moe-noquant-small",
    "Qwen3-30B-A3B-tiny",
    "qwen3-next-80b-a3b-instruct-reduced-l2",
]
filenames = ["moe.csv", "moe_kernel_only.csv"]
manifest = {"stage_device_root": str(stage_root), "models": {}}
for model in models:
    model_manifest = {}
    for filename in filenames:
        path = stage_root / model / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        contexts = sorted({row.get("gating_runtime_context", "") for row in rows})
        data = path.read_bytes()
        model_manifest[filename] = {
            "path": str(path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "row_count": len(rows),
            "gating_runtime_context_values": contexts,
        }
    manifest["models"][model] = model_manifest
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

echo "B0 H800 MoE gating backfill staged at: $STAGE_ROOT"
echo "Merge with tests/e2e/operator_parity/merge_profile_csv_contexts.py after auditing staged files."
