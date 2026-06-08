#!/bin/bash
# =============================================================================
# Co-location (Monolithic) Online Mode - Dense Thinking Mode Example
# =============================================================================
# This script demonstrates a minimal Thinking Mode workflow on the monolithic
# architecture. It keeps the same runtime optimization defaults as the dense
# baseline: decode CUDA Graph modeling plus Chunked Prefill.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DISABLED=true
export VIDUR_DISABLE_WANDB=1
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-2-7b-hf}"
SYS_ARCH="${SYS_ARCH:-co-location}"
CC_BACKEND="${CC_BACKEND:-analytical}"
NUM_REPLICAS="${NUM_REPLICAS:-2}"
ATTN_TP="${ATTN_TP:-2}"
PP="${PP:-1}"
DP="${DP:-1}"
REPLICA_SCHEDULER="${REPLICA_SCHEDULER:-vllm_v1}"
NUM_REQUESTS="${NUM_REQUESTS:-1}"
PREFILL_TOKENS="${PREFILL_TOKENS:-8}"
DECODE_TOKENS="${DECODE_TOKENS:-2}"
THINKING_DEPTH="${THINKING_DEPTH:-2}"
TOOL_CALL_LATENCY="${TOOL_CALL_LATENCY:-0.001}"
THINKING_ROUND_PREFILL_TOKENS="${THINKING_ROUND_PREFILL_TOKENS:-3}"
THINKING_ROUND_DECODE_TOKENS="${THINKING_ROUND_DECODE_TOKENS:-1}"
ENABLE_DUMMY_MODE="${ENABLE_DUMMY_MODE:-true}"
DUMMY_EXEC_TIME_MS="${DUMMY_EXEC_TIME_MS:-1.0}"
DECODE_CUDA_GRAPH_MODE="${DECODE_CUDA_GRAPH_MODE:-full_decode_only}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-true}"
MAX_TOKENS_IN_BATCH="${MAX_TOKENS_IN_BATCH:-128}"
LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-4}"
METRICS_OUTPUT_DIR="${METRICS_OUTPUT_DIR:-$REPO_ROOT/outputs/examples/co-location/online}"
RUN_ID="${RUN_ID:-thinking_mode_basic_online}"

require_bool() {
  local name="$1"
  local value="$2"
  if [ "$value" != "true" ] && [ "$value" != "false" ]; then
    echo "ERROR: $name must be true or false; got $value" >&2
    exit 2
  fi
}

require_bool "ENABLE_DUMMY_MODE" "$ENABLE_DUMMY_MODE"
require_bool "ENABLE_CHUNKED_PREFILL" "$ENABLE_CHUNKED_PREFILL"


if [ "$SYS_ARCH" != "co-location" ]; then
  echo "ERROR: this example only supports SYS_ARCH=co-location; got SYS_ARCH=$SYS_ARCH" >&2
  exit 2
fi


if [ "$ENABLE_CHUNKED_PREFILL" = "false" ] && [ "$LONG_PREFILL_TOKEN_THRESHOLD" != "0" ]; then
  echo "ERROR: LONG_PREFILL_TOKEN_THRESHOLD must be 0 when ENABLE_CHUNKED_PREFILL=false" >&2
  exit 2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: PYTHON_BIN is not executable or not on PATH: $PYTHON_BIN" >&2
  exit 2
fi

CMD=(
  "$PYTHON_BIN" -m frontier.main
  --simulation_mode online
  --sys_arch "$SYS_ARCH"
  --cluster_config_num_replicas "$NUM_REPLICAS"
  --replica_config_model_name "$MODEL_NAME"
  --replica_config_attn_tensor_parallel_size "$ATTN_TP"
  --replica_config_num_pipeline_stages "$PP"
  --replica_config_attn_data_parallel_size "$DP"
  --cc_backend_config_type "$CC_BACKEND"
  --replica_scheduler_config_type "$REPLICA_SCHEDULER"
  --decode_cuda_graph_mode "$DECODE_CUDA_GRAPH_MODE"
  --vllm_v1_scheduler_config_max_tokens_in_batch "$MAX_TOKENS_IN_BATCH"
  --vllm_v1_scheduler_config_long_prefill_token_threshold "$LONG_PREFILL_TOKEN_THRESHOLD"
  --request_generator_config_type synthetic
  --synthetic_request_generator_config_num_requests "$NUM_REQUESTS"
  --length_generator_config_type fixed
  --fixed_request_length_generator_config_prefill_tokens "$PREFILL_TOKENS"
  --fixed_request_length_generator_config_decode_tokens "$DECODE_TOKENS"
  --interval_generator_config_type poisson
  --poisson_request_interval_generator_config_qps 1.0
  --metrics_config_output_dir "$METRICS_OUTPUT_DIR"
  --metrics_config_run_id "$RUN_ID"
  --metrics_config_write_metrics
  --metrics_config_store_request_metrics
  --metrics_config_store_batch_metrics
  --metrics_config_store_token_completion_metrics
  --metrics_config_store_utilization_metrics
  --no-metrics_config_store_plots
  --no-metrics_config_enable_chrome_trace
  --no-metrics_config_write_json_trace
  --enable_thinking_mode
  --thinking_depth "$THINKING_DEPTH"
  --tool_call_latency "$TOOL_CALL_LATENCY"
  --thinking_round_prefill_tokens "$THINKING_ROUND_PREFILL_TOKENS"
  --thinking_round_decode_tokens "$THINKING_ROUND_DECODE_TOKENS"
)

if [ "$ENABLE_CHUNKED_PREFILL" = "true" ]; then
  CMD+=(--vllm_v1_scheduler_config_enable_chunked_prefill)
else
  CMD+=(--no-vllm_v1_scheduler_config_enable_chunked_prefill)
fi

if [ "$ENABLE_DUMMY_MODE" = "true" ]; then
  CMD+=(
    --random_forrest_execution_time_predictor_config_enable_dummy_mode
    --random_forrest_execution_time_predictor_config_dummy_execution_time_ms "$DUMMY_EXEC_TIME_MS"
  )
fi

if [ "$#" -gt 0 ]; then
  if [ "$1" = "--" ]; then
    shift
  fi
  CMD+=("$@")
fi

cat <<EOF
============================================
  Co-location Mode - Dense Thinking Mode
============================================
PYTHONPATH: $PYTHONPATH
Model: $MODEL_NAME
Architecture: $SYS_ARCH
Backend: $CC_BACKEND
Simulation Mode: online
Replicas: $NUM_REPLICAS
Parallelism: TP=$ATTN_TP, PP=$PP, DP=$DP
Scheduler: $REPLICA_SCHEDULER
Requests: $NUM_REQUESTS (prefill=$PREFILL_TOKENS, decode=$DECODE_TOKENS)
Thinking Mode: enabled (depth=$THINKING_DEPTH, hidden prefill=$THINKING_ROUND_PREFILL_TOKENS, hidden decode=$THINKING_ROUND_DECODE_TOKENS)
Runtime Optimizations: decode_cuda_graph_mode=$DECODE_CUDA_GRAPH_MODE, chunked_prefill=$ENABLE_CHUNKED_PREFILL
Metrics: output_dir=$METRICS_OUTPUT_DIR, run_id=$RUN_ID
Dummy Mode: $ENABLE_DUMMY_MODE
============================================
EOF

echo "Running online simulation..."
if "${CMD[@]}"; then
  echo "Online simulation completed successfully."
else
  exit_code=$?
  echo "Simulation failed (exit code: $exit_code)" >&2
  exit "$exit_code"
fi
