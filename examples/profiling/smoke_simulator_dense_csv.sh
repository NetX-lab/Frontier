#!/bin/bash
# =============================================================================
# Profiling Example - Dense CSV Downstream Simulator Smoke
# =============================================================================
# Feeds checked-in linear_op.csv and attention.csv profiles directly into the
# simulator without dummy predictor mode.
#
# Default checked-in profiles:
#   data/profiling/compute/rtx_pro_6000/qwen2_dense_test/linear_op.csv
#   data/profiling/compute/rtx_pro_6000/qwen2_dense_test/attention.csv
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DISABLED=true
export VIDUR_DISABLE_WANDB=1
PYTHON_BIN="${PYTHON_BIN:-python3}"
CC_BACKEND_CONFIG_TYPE="${CC_BACKEND_CONFIG_TYPE:-astra_sim_analytical}"
DEVICE="${DEVICE:-rtx_pro_6000}"
MODEL="${MODEL:-qwen2_dense_test}"
DATA_DIR_BASE="${DATA_DIR_BASE:-$REPO_ROOT/data/profiling}"
LINEAR_OP_CSV="${LINEAR_OP_CSV:-}"
ATTENTION_CSV="${ATTENTION_CSV:-}"
METRICS_OUTPUT_DIR="${METRICS_OUTPUT_DIR:-$REPO_ROOT/outputs/examples/profiling-simulator}"
RUN_ID="${RUN_ID:-profiling_dense_csv_smoke}"

require_cli_value() {
  local option="$1"
  local value="${2-}"
  if [ -z "$value" ] || [[ "$value" == --* ]]; then
    echo "ERROR: $option requires a value" >&2
    exit 2
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --python-bin)
      require_cli_value "$1" "${2-}"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --device)
      require_cli_value "$1" "${2-}"
      DEVICE="$2"
      shift 2
      ;;
    --model)
      require_cli_value "$1" "${2-}"
      MODEL="$2"
      shift 2
      ;;
    --output-root|--data-dir-base)
      require_cli_value "$1" "${2-}"
      DATA_DIR_BASE="$2"
      shift 2
      ;;
    --linear-op-csv)
      require_cli_value "$1" "${2-}"
      LINEAR_OP_CSV="$2"
      shift 2
      ;;
    --attention-csv)
      require_cli_value "$1" "${2-}"
      ATTENTION_CSV="$2"
      shift 2
      ;;
    --cc-backend-config-type)
      require_cli_value "$1" "${2-}"
      CC_BACKEND_CONFIG_TYPE="$2"
      shift 2
      ;;
    --metrics-output-dir)
      require_cli_value "$1" "${2-}"
      METRICS_OUTPUT_DIR="$2"
      shift 2
      ;;
    --run-id)
      require_cli_value "$1" "${2-}"
      RUN_ID="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [ "$#" -gt 0 ]; then
  echo "ERROR: unexpected positional arguments: $*" >&2
  exit 2
fi

LINEAR_OP_CSV="${LINEAR_OP_CSV:-$DATA_DIR_BASE/compute/$DEVICE/$MODEL/linear_op.csv}"
ATTENTION_CSV="${ATTENTION_CSV:-$DATA_DIR_BASE/compute/$DEVICE/$MODEL/attention.csv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: PYTHON_BIN is not executable or not on PATH: $PYTHON_BIN" >&2
  exit 2
fi

for csv_path in "$LINEAR_OP_CSV" "$ATTENTION_CSV"; do
  if [ ! -f "$csv_path" ]; then
    echo "ERROR: required profiling CSV is missing: $csv_path" >&2
    echo "Expected profiling taxonomy: data/profiling/compute/<device>/<model>/..." >&2
    exit 2
  fi
done

CMD=(
  "$PYTHON_BIN" -m frontier.main
  --simulation_mode offline
  --sys_arch co-location
  --cluster_config_num_replicas 1
  --replica_config_model_name "$MODEL"
  --replica_config_attn_tensor_parallel_size 1
  --replica_config_num_pipeline_stages 1
  --replica_config_attn_data_parallel_size 1
  --cc_backend_config_type "$CC_BACKEND_CONFIG_TYPE"
  --replica_scheduler_config_type vllm_v1
  --request_generator_config_type synthetic
  --synthetic_request_generator_config_num_requests 1
  --length_generator_config_type fixed
  --fixed_request_length_generator_config_prefill_tokens 8
  --fixed_request_length_generator_config_decode_tokens 2
  --interval_generator_config_type poisson
  --poisson_request_interval_generator_config_qps 1.0
  --no-random_forrest_execution_time_predictor_config_enable_dummy_mode
  --random_forrest_execution_time_predictor_config_linear_op_input_file "$LINEAR_OP_CSV"
  --random_forrest_execution_time_predictor_config_atten_input_file "$ATTENTION_CSV"
  --random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 1024
  --random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling
  --metrics_config_output_dir "$METRICS_OUTPUT_DIR"
  --metrics_config_run_id "$RUN_ID"
  --metrics_config_write_metrics
  --metrics_config_store_request_metrics
  --no-metrics_config_store_plots
  --no-metrics_config_enable_chrome_trace
  --no-metrics_config_write_json_trace
)

cat <<EOF
============================================
  Dense CSV Downstream Simulator Smoke
============================================
Taxonomy: data/profiling/compute/<device>/<model>/...
Linear profile: $LINEAR_OP_CSV
Attention profile: $ATTENTION_CSV
CC backend: $CC_BACKEND_CONFIG_TYPE
Metrics run: $METRICS_OUTPUT_DIR / $RUN_ID
============================================
EOF

cd "$REPO_ROOT"
"${CMD[@]}"
