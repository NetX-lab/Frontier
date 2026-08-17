#!/usr/bin/env bash
# Collect the exact H800 profiling supplements required by the 200-case matrix.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROFILE_ENV_ROOT="${PROFILE_ENV_ROOT:-/data/ycfeng/frontier_profiling_envs/issue2_py312_target_v2}"
CUDA12="${CUDA12:-/data/ycfeng/cu128_build/cuda-12.8}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
DEVICE="${DEVICE:-h800}"
NUM_GPUS="${NUM_GPUS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
STAGE_ROOT="${STAGE_ROOT:-/data/ycfeng/tmp/frontier_moe_ep_profile_backfill_20260817}"
BASE_REF="${BASE_REF:-HEAD}"
BASE_COMMIT=""
DRY_RUN=false

TOKENS=(1 2 4 8 16 32 64)
QWEN_MTP_TOKENS=(1 2 4 8 16 32 64 128)
MTP_COLUMNS=(
  time_stats.mtp_fusion_proj.min
  time_stats.mtp_fusion_proj.max
  time_stats.mtp_fusion_proj.mean
  time_stats.mtp_fusion_proj.median
  time_stats.mtp_fusion_proj.std
  time_stats.mtp_fusion_proj.count
  time_stats.lm_head_linear.min
  time_stats.lm_head_linear.max
  time_stats.lm_head_linear.mean
  time_stats.lm_head_linear.median
  time_stats.lm_head_linear.std
  time_stats.lm_head_linear.count
)

RAW_ROOT="$STAGE_ROOT/raw"
MERGED_ROOT="$STAGE_ROOT/merged"
BASELINE_ROOT="$STAGE_ROOT/baseline"
BASELINE_CANONICAL_ROOT="$BASELINE_ROOT/data/profiling/compute/$DEVICE"
LOG_ROOT="$STAGE_ROOT/logs"
AUDIT_JSON="$STAGE_ROOT/profile_audit.json"
MERGE_TOOL="$REPO_ROOT/tests/e2e/operator_parity/merge_profile_csv_contexts.py"

usage() {
  cat <<'EOF'
Usage: run_moe_ep_h800_profile_backfill.sh [--stage-root PATH] [--base-ref REF] [--dry-run]

The default mode collects into a new staging root and never writes canonical
profiling CSVs. The merge baseline is exported from the explicit Git base ref,
not copied from the working tree. Publish only after inspecting
profile_audit.json and running the task's explicit merge command.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage-root)
      [[ $# -ge 2 ]] || { echo "--stage-root requires a value" >&2; exit 2; }
      STAGE_ROOT="$2"
      RAW_ROOT="$STAGE_ROOT/raw"
      MERGED_ROOT="$STAGE_ROOT/merged"
      BASELINE_ROOT="$STAGE_ROOT/baseline"
      BASELINE_CANONICAL_ROOT="$BASELINE_ROOT/data/profiling/compute/$DEVICE"
      LOG_ROOT="$STAGE_ROOT/logs"
      AUDIT_JSON="$STAGE_ROOT/profile_audit.json"
      shift 2
      ;;
    --base-ref)
      [[ $# -ge 2 ]] || { echo "--base-ref requires a value" >&2; exit 2; }
      BASE_REF="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! BASE_COMMIT="$(git -C "$REPO_ROOT" rev-parse --verify "${BASE_REF}^{commit}")"; then
  echo "Invalid Git base ref: $BASE_REF" >&2
  exit 2
fi
printf 'base_ref=%s base_commit=%s\n' "$BASE_REF" "$BASE_COMMIT"

if [[ "$DRY_RUN" != true && -e "$STAGE_ROOT" ]]; then
  echo "Refusing to reuse existing staging root: $STAGE_ROOT" >&2
  echo "Choose a new --stage-root; no files were changed." >&2
  exit 2
fi

if [[ "$DRY_RUN" != true ]]; then
  mkdir -p \
    "$RAW_ROOT" \
    "$MERGED_ROOT/compute/$DEVICE" \
    "$BASELINE_ROOT" \
    "$LOG_ROOT"
fi

export CUDA_VISIBLE_DEVICES
export CUDA_HOME="$CUDA12"
export PATH="$CUDA12/bin:$PROFILE_ENV_ROOT/nvidia/cuda_nvcc/bin:$PATH"
export PYTHONPATH="$PROFILE_ENV_ROOT:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CUDA12/lib64:$PROFILE_ENV_ROOT/nvidia/cuda_runtime/lib:$PROFILE_ENV_ROOT/nvidia/cuda_nvrtc/lib:$PROFILE_ENV_ROOT/nvidia/cublas/lib:$PROFILE_ENV_ROOT/nvidia/cudnn/lib:$PROFILE_ENV_ROOT/nvidia/cufft/lib:$PROFILE_ENV_ROOT/nvidia/curand/lib:$PROFILE_ENV_ROOT/nvidia/cusolver/lib:$PROFILE_ENV_ROOT/nvidia/cuda_cupti/lib:$PROFILE_ENV_ROOT/nvidia/cufile/lib:$PROFILE_ENV_ROOT/nvidia/cusparselt/lib:$PROFILE_ENV_ROOT/nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export WANDB_DISABLED=true
export VIDUR_DISABLE_WANDB=1
export KINETO_LOG_LEVEL=5

run_logged() {
  local name="$1"
  shift
  printf 'COMMAND'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" == true ]]; then
    return 0
  fi
  local log_file="$LOG_ROOT/${name}.log"
  {
    printf 'COMMAND'
    printf ' %q' "$@"
    printf '\n'
  } >"$log_file"
  "$@" 2>&1 | tee -a "$log_file"
}

profile_moe() {
  local model="$1"
  local method="$2"
  local context="$3"
  local tp_sizes="$4"
  local ep_sizes="$5"
  local model_slug="${model//[^A-Za-z0-9_.-]/_}"
  local run_slug="${model_slug}_${method}_${context}"
  local output_root="$RAW_ROOT/$run_slug"
  local output_file="moe.csv"
  [[ "$method" == "record_function" ]] && output_file="moe_kernel_only.csv"
  read -r -a tp_args <<<"$tp_sizes"
  read -r -a ep_args <<<"$ep_sizes"

  run_logged "profile_${run_slug}" \
    "$PYTHON_BIN" -m frontier.profiling.moe.main \
    --disable_ray \
    --num_gpus "$NUM_GPUS" \
    --models "$model" \
    --device "$DEVICE" \
    --output_dir "$output_root" \
    --num_tensor_parallel_workers "${tp_args[@]}" \
    --expert_parallel_sizes "${ep_args[@]}" \
    --num_tokens_list "${TOKENS[@]}" \
    --profile_method "$method" \
    --routing_runtime_path standard_fused_topk \
    --gating_runtime_context "$context" \
    --enable_load_imbalance \
    --load_distributions uniform \
    --num_samples_per_distribution 1 \
    --yes

  if [[ "$DRY_RUN" != true ]]; then
    [[ -f "$output_root/compute/$DEVICE/$model/$output_file" ]] || {
      echo "Missing expected staged MoE output: $output_root/compute/$DEVICE/$model/$output_file" >&2
      exit 1
    }
  fi
}

profile_qwen_mtp() {
  local model="qwen3-next-80b-a3b-instruct-reduced-l2"
  local output_root="$RAW_ROOT/qwen3_next_mtp_linear"
  run_logged "profile_qwen3_next_mtp_linear" \
    "$PYTHON_BIN" -m frontier.profiling.linear_op.main \
    --disable_ray \
    --num_gpus "$NUM_GPUS" \
    --models "$model" \
    --device "$DEVICE" \
    --output_dir "$output_root" \
    --num_tensor_parallel_workers 4 \
    --attn_tp 4 \
    --ffn_tp 4 \
    --num_tokens_list "${QWEN_MTP_TOKENS[@]}" \
    --max_tokens 128 \
    --is_moe \
    --include_target_embedded_mtp \
    --profile_method cuda_event \
    --yes

  if [[ "$DRY_RUN" != true ]]; then
    [[ -f "$output_root/compute/$DEVICE/$model/linear_op.csv" ]] || {
      echo "Missing expected staged Qwen3-Next MTP output: $output_root/compute/$DEVICE/$model/linear_op.csv" >&2
      exit 1
    }
  fi
}

snapshot_canonical_to_staging() {
  local profile_paths=(
    "data/profiling/compute/$DEVICE/Phi-tiny-MoE-instruct"
    "data/profiling/compute/$DEVICE/step-moe-noquant-small"
    "data/profiling/compute/$DEVICE/qwen3-next-80b-a3b-instruct-reduced-l2"
  )
  printf 'COMMAND git -C %q archive --format=tar %q --' \
    "$REPO_ROOT" \
    "$BASE_COMMIT"
  printf ' %q' "${profile_paths[@]}"
  printf ' | tar -xf - -C %q\n' "$BASELINE_ROOT"
  if [[ "$DRY_RUN" != true ]]; then
    {
      git -C "$REPO_ROOT" archive \
        --format=tar \
        "$BASE_COMMIT" \
        -- \
        "${profile_paths[@]}" |
        tar -xf - -C "$BASELINE_ROOT"
    } 2>&1 | tee "$LOG_ROOT/snapshot_canonical_from_git.log"
  fi

  local model
  for model in Phi-tiny-MoE-instruct step-moe-noquant-small qwen3-next-80b-a3b-instruct-reduced-l2; do
    run_logged "copy_${model}" \
      cp -a \
      "$BASELINE_CANONICAL_ROOT/$model" \
      "$MERGED_ROOT/compute/$DEVICE/"
  done
}

merge_moe_supplements() {
  local model="$1"
  local method="$2"
  local context="$3"
  local model_slug="${model//[^A-Za-z0-9_.-]/_}"
  local run_slug="${model_slug}_${method}_${context}"
  local filename="moe.csv"
  [[ "$method" == "record_function" ]] && filename="moe_kernel_only.csv"
  run_logged "merge_${run_slug}" \
    "$PYTHON_BIN" "$MERGE_TOOL" \
    --canonical-root "$MERGED_ROOT/compute/$DEVICE" \
    --supplement-root "$RAW_ROOT/$run_slug/compute/$DEVICE" \
    --allow-in-place \
    --models "$model" \
    --filenames "$filename"
}

merge_qwen_mtp_columns() {
  run_logged "merge_qwen3_next_mtp_columns" \
    "$PYTHON_BIN" "$MERGE_TOOL" \
    --canonical-root "$MERGED_ROOT/compute/$DEVICE" \
    --supplement-root "$RAW_ROOT/qwen3_next_mtp_linear/compute/$DEVICE" \
    --allow-in-place \
    --models qwen3-next-80b-a3b-instruct-reduced-l2 \
    --filenames linear_op.csv \
    --enrich-columns "${MTP_COLUMNS[@]}" \
    --drop-canonical-key is_step2_mini=False \
    --supplement-key num_tensor_parallel_workers=4
}

audit_staging() {
  run_logged "audit_staging" \
    "$PYTHON_BIN" - \
    "$MERGED_ROOT/compute/$DEVICE" \
    "$BASELINE_CANONICAL_ROOT" \
    "$AUDIT_JSON" \
    "$BASE_REF" \
    "$BASE_COMMIT" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
canonical_root = Path(sys.argv[2])
report_path = Path(sys.argv[3])
base_ref = sys.argv[4]
base_commit = sys.argv[5]
tokens = {1, 2, 4, 8, 16, 32, 64}
qwen_tokens = {1, 2, 4, 8, 16, 32, 64, 128}
mtp_columns = (
    "time_stats.mtp_fusion_proj.min",
    "time_stats.mtp_fusion_proj.max",
    "time_stats.mtp_fusion_proj.mean",
    "time_stats.mtp_fusion_proj.median",
    "time_stats.mtp_fusion_proj.std",
    "time_stats.mtp_fusion_proj.count",
    "time_stats.lm_head_linear.min",
    "time_stats.lm_head_linear.max",
    "time_stats.lm_head_linear.mean",
    "time_stats.lm_head_linear.median",
    "time_stats.lm_head_linear.std",
    "time_stats.lm_head_linear.count",
)


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def finite(value):
    return bool(str(value).strip()) and math.isfinite(float(str(value).strip()))


def audit_file(path, expected_measurement):
    fields, rows = read(path)
    if not rows:
        raise ValueError(f"empty profiling CSV: {path}")
    if {row.get("measurement_type", "") for row in rows} != {expected_measurement}:
        raise ValueError(f"unexpected measurement family in {path}")
    timing = [field for field in fields if field.startswith("time_stats.")]
    invalid = [
        (index, field)
        for index, row in enumerate(rows)
        for field in timing
        if not finite(row.get(field, ""))
    ]
    if invalid:
        raise ValueError(f"non-finite timing values in {path}: {invalid[:5]}")
    return {"path": str(path), "rows": len(rows), "timing_columns": len(timing)}


def audit_moe(model, expected_tp_ep):
    results = []
    for filename, measurement in (("moe.csv", "CUDA_EVENT"), ("moe_kernel_only.csv", "KERNEL_ONLY")):
        path = root / model / filename
        result = audit_file(path, measurement)
        fields, rows = read(path)
        observed = {}
        for row in rows:
            key = (
                int(row["num_tensor_parallel_workers"]),
                int(row["expert_parallel_size"]),
                row["routing_runtime_path"],
                row["gating_runtime_context"],
                int(row["num_tokens"]),
            )
            observed[key] = observed.get(key, 0) + 1
        required = 0
        for tp, ep in expected_tp_ep:
            for context in ("standalone_legacy", "prefill_hot"):
                for token in tokens:
                    key = (tp, ep, "standard_fused_topk", context, token)
                    if observed.get(key) != 1:
                        raise ValueError(f"missing/duplicate MoE key {key} in {path}")
                    required += 1
        result["required_standard_fused_topk_keys"] = required
        results.append(result)
    return results


audit = {
    "provenance": {
        "base_ref": base_ref,
        "base_commit": base_commit,
        "baseline_root": str(canonical_root),
    },
    "moe": {
        "Phi-tiny-MoE-instruct": audit_moe(
            "Phi-tiny-MoE-instruct",
            {(1, 1), (1, 2), (1, 4), (1, 8)},
        ),
        "step-moe-noquant-small": audit_moe(
            "step-moe-noquant-small",
            {(4, 1), (4, 2)},
        ),
    }
}

model = "qwen3-next-80b-a3b-instruct-reduced-l2"
qwen = root / model / "linear_op.csv"
canonical_qwen = canonical_root / model / "linear_op.csv"
qwen_fields, qwen_rows = read(qwen)
canonical_fields, canonical_rows = read(canonical_qwen)
if "is_step2_mini" not in canonical_fields:
    raise ValueError(f"expected legacy is_step2_mini key in {canonical_qwen}")
if {row.get("is_step2_mini", "") for row in canonical_rows} != {"False"}:
    raise ValueError(f"unexpected legacy key values in {canonical_qwen}")
if "is_step2_mini" in qwen_fields:
    raise ValueError(f"legacy is_step2_mini key remains in {qwen}")
for column in mtp_columns:
    if column not in qwen_fields:
        raise ValueError(f"missing MTP column {column} in {qwen}")

canonical_comparable_fields = [
    field for field in canonical_fields if field != "is_step2_mini"
]
canonical_key_fields = [
    field
    for field in canonical_comparable_fields
    if not field.startswith("time_stats.")
]
staged_key_fields = [
    field for field in qwen_fields if not field.startswith("time_stats.")
]
if set(canonical_key_fields) != set(staged_key_fields):
    raise ValueError(
        "Qwen key schema changed beyond legacy migration: "
        f"canonical={canonical_key_fields}, staged={staged_key_fields}"
    )
canonical_by_key = {}
for row in canonical_rows:
    key = tuple(row.get(field, "") for field in canonical_key_fields)
    if key in canonical_by_key:
        raise ValueError(f"duplicate canonical Qwen key: {key}")
    canonical_by_key[key] = row
staged_by_key = {}
for row in qwen_rows:
    key = tuple(row.get(field, "") for field in canonical_key_fields)
    if key in staged_by_key:
        raise ValueError(f"duplicate staged Qwen key: {key}")
    staged_by_key[key] = row
if set(canonical_by_key) != set(staged_by_key):
    raise ValueError("Qwen row keys changed during MTP enrichment")

preserved_timing_fields = [
    field
    for field in canonical_fields
    if field.startswith("time_stats.") and field not in mtp_columns
]
preserved_cells = 0
for key, canonical_row in canonical_by_key.items():
    staged_row = staged_by_key[key]
    for field in preserved_timing_fields:
        if canonical_row.get(field, "") != staged_row.get(field, ""):
            raise ValueError(
                f"Qwen canonical timing changed for {field}, key={key}"
            )
        preserved_cells += 1

qwen_tp4 = [
    row for row in qwen_rows
    if int(row["num_tensor_parallel_workers"]) == 4
]
if {int(row["num_tokens"]) for row in qwen_tp4} != qwen_tokens:
    raise ValueError(f"Qwen3-Next TP=4 token coverage mismatch in {qwen}")
mtp_populated_cells = 0
for row in qwen_tp4:
    for column in mtp_columns:
        if not finite(row.get(column, "")):
            raise ValueError(f"invalid Qwen3-Next MTP value {column} for key {row}")
        mtp_populated_cells += 1
qwen_non_tp4 = [
    row for row in qwen_rows
    if int(row["num_tensor_parallel_workers"]) != 4
]
mtp_empty_nonselected_cells = 0
for row in qwen_non_tp4:
    for column in mtp_columns:
        if str(row.get(column, "")).strip():
            raise ValueError(
                f"unexpected non-TP4 MTP value {column} for key {row}"
            )
        mtp_empty_nonselected_cells += 1
audit["qwen3_next_mtp"] = {
    "path": str(qwen),
    "canonical_path": str(canonical_qwen),
    "row_count": len(qwen_rows),
    "tp4_rows": len(qwen_tp4),
    "non_tp4_rows": len(qwen_non_tp4),
    "token_points": sorted(qwen_tokens),
    "columns": list(mtp_columns),
    "legacy_key_removed": True,
    "preserved_existing_timing_fields": len(preserved_timing_fields),
    "preserved_existing_timing_cells": preserved_cells,
    "mtp_populated_cells": mtp_populated_cells,
    "mtp_empty_nonselected_cells": mtp_empty_nonselected_cells,
}

report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(audit, indent=2, sort_keys=True))
PY
}

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run: no GPU or filesystem mutation will occur."
fi

run_logged environment \
  "$PYTHON_BIN" - <<'PY'
import importlib
import os
import subprocess
import sys

print("python", sys.executable, sys.version.split()[0])
print("cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("cuda_home", os.environ.get("CUDA_HOME"))
for module_name in ("torch", "vllm", "flashinfer", "triton", "pandas"):
    module = importlib.import_module(module_name)
    print(module_name, getattr(module, "__version__", "unknown"))
import torch
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
    raise SystemExit("Expected at least 8 visible CUDA devices")
subprocess.run(["nvidia-smi", "-L"], check=True)
PY

profile_moe Phi-tiny-MoE-instruct cuda_event standalone_legacy "1" "1 2 4 8"
profile_moe Phi-tiny-MoE-instruct cuda_event prefill_hot "1" "1 2 4 8"
profile_moe Phi-tiny-MoE-instruct record_function standalone_legacy "1" "1 2 4 8"
profile_moe Phi-tiny-MoE-instruct record_function prefill_hot "1" "1 2 4 8"
profile_moe step-moe-noquant-small cuda_event standalone_legacy "4" "1 2"
profile_moe step-moe-noquant-small cuda_event prefill_hot "4" "1 2"
profile_moe step-moe-noquant-small record_function standalone_legacy "4" "1 2"
profile_moe step-moe-noquant-small record_function prefill_hot "4" "1 2"
profile_qwen_mtp

snapshot_canonical_to_staging
merge_moe_supplements Phi-tiny-MoE-instruct cuda_event standalone_legacy
merge_moe_supplements Phi-tiny-MoE-instruct cuda_event prefill_hot
merge_moe_supplements Phi-tiny-MoE-instruct record_function standalone_legacy
merge_moe_supplements Phi-tiny-MoE-instruct record_function prefill_hot
merge_moe_supplements step-moe-noquant-small cuda_event standalone_legacy
merge_moe_supplements step-moe-noquant-small cuda_event prefill_hot
merge_moe_supplements step-moe-noquant-small record_function standalone_legacy
merge_moe_supplements step-moe-noquant-small record_function prefill_hot
merge_qwen_mtp_columns
audit_staging

echo "Staging profile backfill completed: $STAGE_ROOT"
echo "Merged, audited profiles: $MERGED_ROOT/compute/$DEVICE"
echo "Audit report: $AUDIT_JSON"
