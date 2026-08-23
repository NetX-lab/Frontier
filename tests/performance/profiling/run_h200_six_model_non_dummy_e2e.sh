#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DISABLED=true
export VIDUR_DISABLE_WANDB=1

VALIDATOR_MODULE="tests.performance.profiling.validate_h200_six_model_non_dummy_e2e"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PROFILE_ROOT="${PROFILE_ROOT:-/data/ycfeng/tmp/frontier_profiling_staging_20260823/h200/e2e_fixture}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/ycfeng/tmp/frontier_profiling_staging_20260823/h200/e2e_runs}"
REPORT_ROOT="${REPORT_ROOT:-/data/ycfeng/tmp/frontier_profiling_staging_20260823/h200/e2e_reports}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-h200_six_model_non_dummy_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN=false
PREFLIGHT_ONLY=false

SUPPORTED_MODELS=()
SELECTED_MODELS=()

usage() {
  cat <<'EOF'
Usage:
  run_h200_six_model_non_dummy_e2e.sh [options]

Options:
  --model MODEL          Run one supported model. Repeat to select multiple models.
  --profile-root PATH    Root containing one profiling directory per model.
  --output-root PATH     Frontier metrics output root.
  --report-root PATH     Preflight, runtime log, and validation report root.
  --run-id-prefix TEXT   Prefix used for each model's metrics run ID.
  --python-bin PATH      Python executable.
  --preflight-only       Validate profiling CSVs without running Frontier.
  --dry-run              Print resolved commands without executing them.
  -h, --help             Show this help.

When --model is omitted, the six confirmed H200 models run serially.
EOF
}

require_cli_value() {
  local option="$1"
  local value="${2-}"
  if [ -z "$value" ] || [[ "$value" == --* ]]; then
    echo "ERROR: $option requires a value." >&2
    exit 2
  fi
}

is_supported_model() {
  local candidate="$1"
  local supported
  for supported in "${SUPPORTED_MODELS[@]}"; do
    if [ "$candidate" = "$supported" ]; then
      return 0
    fi
  done
  return 1
}

print_command() {
  printf 'Command:'
  printf ' %q' "$@"
  printf '\n'
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --model)
      require_cli_value "$1" "${2-}"
      SELECTED_MODELS+=("$2")
      shift 2
      ;;
    --profile-root)
      require_cli_value "$1" "${2-}"
      PROFILE_ROOT="$2"
      shift 2
      ;;
    --output-root)
      require_cli_value "$1" "${2-}"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --report-root)
      require_cli_value "$1" "${2-}"
      REPORT_ROOT="$2"
      shift 2
      ;;
    --run-id-prefix)
      require_cli_value "$1" "${2-}"
      RUN_ID_PREFIX="$2"
      shift 2
      ;;
    --python-bin)
      require_cli_value "$1" "${2-}"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --preflight-only)
      PREFLIGHT_ONLY=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ "$#" -gt 0 ]; then
  echo "ERROR: unexpected positional arguments: $*" >&2
  exit 2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python executable is unavailable: $PYTHON_BIN" >&2
  exit 2
fi

supported_model_output="$(
  "$PYTHON_BIN" - <<'PY'
from tests.performance.profiling.validate_h200_six_model_non_dummy_e2e import (
    SUPPORTED_MODELS,
)

print("\n".join(SUPPORTED_MODELS))
PY
)" || {
  echo "ERROR: failed to load the H200 model registry." >&2
  exit 1
}
mapfile -t SUPPORTED_MODELS <<<"$supported_model_output"
if [ "${#SUPPORTED_MODELS[@]}" -eq 0 ]; then
  echo "ERROR: the H200 model registry is empty." >&2
  exit 1
fi

if [[ "$RUN_ID_PREFIX" == *"/"* ]] \
  || [[ "$RUN_ID_PREFIX" == *"\\"* ]] \
  || [[ "$RUN_ID_PREFIX" == *".."* ]]; then
  echo "ERROR: --run-id-prefix must be a single safe path component." >&2
  exit 2
fi

if [ "${#SELECTED_MODELS[@]}" -eq 0 ]; then
  SELECTED_MODELS=("${SUPPORTED_MODELS[@]}")
fi

for model in "${SELECTED_MODELS[@]}"; do
  if ! is_supported_model "$model"; then
    echo "ERROR: unsupported model: $model" >&2
    exit 2
  fi
done

PROFILE_ROOT="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$PROFILE_ROOT")"
OUTPUT_ROOT="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$OUTPUT_ROOT")"
REPORT_ROOT="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$REPORT_ROOT")"

cd "$REPO_ROOT"

for model in "${SELECTED_MODELS[@]}"; do
  profile_dir="$PROFILE_ROOT/$model"
  if [ ! -d "$profile_dir" ]; then
    echo "ERROR: profiling directory is missing: $profile_dir" >&2
    exit 1
  fi

  contract_values="$(
    "$PYTHON_BIN" - "$model" <<'PY'
import sys

from tests.performance.profiling.validate_h200_six_model_non_dummy_e2e import (
    build_model_contract,
)

contract = build_model_contract(sys.argv[1])
print(
    int(contract.is_moe),
    contract.num_experts,
    contract.router_topk,
    contract.num_pipeline_stages,
)
PY
  )"
  read -r is_moe num_experts router_topk num_pipeline_stages \
    <<<"$(tail -n 1 <<<"$contract_values")"
  if [[ ! "$is_moe" =~ ^[01]$ ]] \
    || [[ ! "$num_experts" =~ ^[0-9]+$ ]] \
    || [[ ! "$router_topk" =~ ^[0-9]+$ ]] \
    || [[ ! "$num_pipeline_stages" =~ ^[0-9]+$ ]] \
    || [ "$num_pipeline_stages" -lt 1 ]; then
    echo "ERROR: invalid registry-derived contract values for $model." >&2
    printf '%s\n' "$contract_values" >&2
    exit 1
  fi

  if [ "$is_moe" = "1" ]; then
    attn_dp=2
    moe_ep=2
  else
    attn_dp=1
    moe_ep=1
  fi

  model_report_dir="$REPORT_ROOT/$model"
  preflight_report="$model_report_dir/preflight.json"
  runtime_report="$model_report_dir/runtime.json"
  runtime_log="$model_report_dir/runtime.log"
  run_id="${RUN_ID_PREFIX}_${model}"

  run_dir="$("$PYTHON_BIN" - "$OUTPUT_ROOT" "$model" "$run_id" <<'PY'
import sys

from frontier.utils.output_paths import build_metrics_run_output_dir

print(
    build_metrics_run_output_dir(
        output_root=sys.argv[1],
        model_type=sys.argv[2],
        workload_type="offline_batch",
        run_id=sys.argv[3],
    )
)
PY
  )"

  preflight_cmd=(
    "$PYTHON_BIN" -m "$VALIDATOR_MODULE"
    preflight
    --model "$model"
    --profile-dir "$profile_dir"
    --report-json "$preflight_report"
  )

  frontier_cmd=(
    "$PYTHON_BIN" -m frontier.main
    --simulation_mode offline
    --sys_arch co-location
    --decode_cuda_graph_mode full_decode_only
    --cluster_config_num_replicas 1
    --replica_config_model_name "$model"
    --replica_config_device h200
    --replica_config_num_pipeline_stages "$num_pipeline_stages"
    --replica_config_attn_tensor_parallel_size 1
    --replica_config_attn_data_parallel_size "$attn_dp"
    --replica_config_moe_tensor_parallel_size 1
    --replica_config_moe_expert_parallel_size "$moe_ep"
    --cc_backend_config_type analytical
    --replica_scheduler_config_type vllm_v1
    --vllm_v1_scheduler_config_max_tokens_in_batch 2
    --vllm_v1_scheduler_config_long_prefill_token_threshold 2
    --vllm_v1_scheduler_config_enable_chunked_prefill
    --request_generator_config_type synthetic
    --synthetic_request_generator_config_num_requests 1
    --length_generator_config_type fixed
    --fixed_request_length_generator_config_prefill_tokens 2
    --fixed_request_length_generator_config_decode_tokens 2
    --interval_generator_config_type poisson
    --poisson_request_interval_generator_config_qps 1.0
    --no-random_forrest_execution_time_predictor_config_enable_dummy_mode
    --random_forrest_execution_time_predictor_config_no_cache
    --random_forrest_execution_time_predictor_config_k_fold_cv_splits 2
    --random_forrest_execution_time_predictor_config_num_training_job_threads 1
    --random_forrest_execution_time_predictor_config_num_estimators 8
    --random_forrest_execution_time_predictor_config_max_depth 8
    --random_forrest_execution_time_predictor_config_min_samples_split 2
    --random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 2
    --random_forrest_execution_time_predictor_config_prediction_max_batch_size 2
    --random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 4
    --random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling
    --random_forrest_execution_time_predictor_config_linear_op_input_file "$profile_dir/linear_op.csv"
    --random_forrest_execution_time_predictor_config_linear_op_kernel_only_input_file "$profile_dir/linear_op_kernel_only.csv"
    --random_forrest_execution_time_predictor_config_atten_input_file "$profile_dir/attention.csv"
    --random_forrest_execution_time_predictor_config_atten_kernel_only_input_file "$profile_dir/attention_kernel_only.csv"
    --metrics_config_output_dir "$OUTPUT_ROOT"
    --metrics_config_run_id "$run_id"
    --metrics_config_write_metrics
    --metrics_config_store_request_metrics
    --metrics_config_store_frontier_stage_batch_ledger
    --metrics_config_enable_op_level_tracing
    --no-metrics_config_store_plots
    --no-metrics_config_enable_chrome_trace
    --no-metrics_config_write_json_trace
  )

  if [ "$is_moe" = "1" ]; then
    frontier_cmd+=(
      --replica_config_total_expert_num "$num_experts"
      --replica_config_router_topk "$router_topk"
      --replica_config_moe_routing_mode simulation
      --replica_config_moe_routing_seed 42
      --random_forrest_execution_time_predictor_config_moe_input_file "$profile_dir/moe.csv"
      --random_forrest_execution_time_predictor_config_moe_kernel_only_input_file "$profile_dir/moe_kernel_only.csv"
    )
  fi

  validate_cmd=(
    "$PYTHON_BIN" -m "$VALIDATOR_MODULE"
    validate-run
    --model "$model"
    --profile-dir "$profile_dir"
    --run-dir "$run_dir"
    --report-json "$runtime_report"
  )

  printf '\nModel: %s\n' "$model"
  printf 'Profile directory: %s\n' "$profile_dir"
  printf 'Resolved run directory: %s\n' "$run_dir"
  print_command "${preflight_cmd[@]}"

  if [ "$DRY_RUN" = "true" ]; then
    if [ "$PREFLIGHT_ONLY" = "false" ]; then
      print_command "${frontier_cmd[@]}"
      print_command "${validate_cmd[@]}"
    fi
    continue
  fi

  mkdir -p "$model_report_dir"
  "${preflight_cmd[@]}"

  if [ "$PREFLIGHT_ONLY" = "true" ]; then
    echo "Preflight PASS: $model"
    continue
  fi

  if [ -e "$run_dir" ]; then
    echo "ERROR: runtime output already exists: $run_dir" >&2
    exit 1
  fi

  print_command "${frontier_cmd[@]}"
  if ! "${frontier_cmd[@]}" >"$runtime_log" 2>&1; then
    echo "ERROR: Frontier runtime failed for $model. Log: $runtime_log" >&2
    tail -n 200 "$runtime_log" >&2
    exit 1
  fi

  "${validate_cmd[@]}"
  echo "Runtime validation PASS: $model"
done

if [ "$DRY_RUN" = "true" ]; then
  echo "Dry run completed; no preflight or Frontier command was executed."
elif [ "$PREFLIGHT_ONLY" = "true" ]; then
  echo "All selected profile preflights passed."
else
  echo "All selected H200 non-dummy E2E runs passed."
fi
