#!/bin/bash
# =============================================================================
# Co-location (Monolithic) Mode - MoE Prefix Caching Recipe
# =============================================================================
# This advanced recipe enables vLLM V1 Prefix Caching. It uses the public
# shared-session fixture under examples/fixtures/ by default; that trace repeats
# the same block_hash_ids across two requests so the second request has a cache hit.
#
# Prefix caching and speculative/MTP decoding have separate runtime contracts, so
# this recipe intentionally does not enable speculative decoding.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DISABLED=true
export VIDUR_DISABLE_WANDB=1
PYTHON_BIN="${PYTHON_BIN:-python3}"
CC_BACKEND_CONFIG_TYPE="${CC_BACKEND_CONFIG_TYPE:-astra_sim_analytical}"

MODEL_NAME="${MODEL_NAME:-Phi-tiny-MoE-instruct}"
SYS_ARCH="${SYS_ARCH:-co-location}"
NUM_REPLICAS="${NUM_REPLICAS:-4}"
ATTN_TP="${ATTN_TP:-4}"
MOE_TP="${MOE_TP:-2}"
MOE_EP="${MOE_EP:-2}"
PP="${PP:-1}"
DP="${DP:-1}"
TOTAL_EXPERTS="${TOTAL_EXPERTS:-8}"
ROUTER_TOPK="${ROUTER_TOPK:-2}"
MOE_ROUTING_MODE="${MOE_ROUTING_MODE:-simulation}"
MOE_ROUTING_SEED="${MOE_ROUTING_SEED:-42}"
REPLICA_SCHEDULER="${REPLICA_SCHEDULER:-vllm_v1}"
TRACE_FILE="${TRACE_FILE:-$REPO_ROOT/examples/fixtures/prefix_cache_shared_session_trace.csv}"
MAX_TOKENS="${MAX_TOKENS:-128}"
EXPECTED_TRACE_REQUESTS="${EXPECTED_TRACE_REQUESTS:-2}"
ENABLE_DUMMY_MODE="${ENABLE_DUMMY_MODE:-true}"
DUMMY_EXEC_TIME_MS="${DUMMY_EXEC_TIME_MS:-1.0}"
DECODE_CUDA_GRAPH_MODE="${DECODE_CUDA_GRAPH_MODE:-full_decode_only}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-true}"
MAX_TOKENS_IN_BATCH="${MAX_TOKENS_IN_BATCH:-128}"
LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-16}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
NUM_BLOCKS="${NUM_BLOCKS:-128}"
METRICS_OUTPUT_DIR="${METRICS_OUTPUT_DIR:-$REPO_ROOT/outputs/examples/co-location}"
RUN_ID="${RUN_ID:-moe_prefix_caching}"

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

if [ ! -f "$TRACE_FILE" ]; then
  echo "ERROR: TRACE_FILE does not exist: $TRACE_FILE" >&2
  exit 2
fi


if (( ATTN_TP * DP != MOE_TP * MOE_EP )); then
  echo "ERROR: shared-domain MoE requires ATTN_TP * DP == MOE_TP * MOE_EP" >&2
  echo "       got ATTN_TP=$ATTN_TP, DP=$DP, MOE_TP=$MOE_TP, MOE_EP=$MOE_EP" >&2
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
  --simulation_mode offline
  --sys_arch "$SYS_ARCH"
  --cluster_config_num_replicas "$NUM_REPLICAS"
  --cluster_scheduler_config_type sticky_round_robin
  --replica_config_model_name "$MODEL_NAME"
  --replica_config_attn_tensor_parallel_size "$ATTN_TP"
  --replica_config_moe_tensor_parallel_size "$MOE_TP"
  --replica_config_moe_expert_parallel_size "$MOE_EP"
  --replica_config_num_pipeline_stages "$PP"
  --replica_config_attn_data_parallel_size "$DP"
  --cc_backend_config_type "$CC_BACKEND_CONFIG_TYPE"
  --replica_config_total_expert_num "$TOTAL_EXPERTS"
  --replica_config_router_topk "$ROUTER_TOPK"
  --replica_config_moe_routing_mode "$MOE_ROUTING_MODE"
  --replica_config_moe_routing_seed "$MOE_ROUTING_SEED"
  --replica_scheduler_config_type "$REPLICA_SCHEDULER"
  --decode_cuda_graph_mode "$DECODE_CUDA_GRAPH_MODE"
  --vllm_v1_scheduler_config_max_tokens_in_batch "$MAX_TOKENS_IN_BATCH"
  --vllm_v1_scheduler_config_long_prefill_token_threshold "$LONG_PREFILL_TOKEN_THRESHOLD"
  --vllm_v1_scheduler_config_block_size "$BLOCK_SIZE"
  --vllm_v1_scheduler_config_num_blocks "$NUM_BLOCKS"
  --vllm_v1_scheduler_config_enable_prefix_caching
  --request_generator_config_type trace_replay
  --trace_request_generator_config_trace_file "$TRACE_FILE"
  --trace_request_generator_config_max_tokens "$MAX_TOKENS"
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
  Co-location Mode - MoE Prefix Caching
============================================
Model: $MODEL_NAME
Trace: $TRACE_FILE
CC Backend: $CC_BACKEND_CONFIG_TYPE
Expected Trace Shape: requests=$EXPECTED_TRACE_REQUESTS from TRACE_FILE, repeated block_hash_ids produce cache-hit blocks
Parallelism: Attn_TP=$ATTN_TP, MoE_TP=$MOE_TP, MoE_EP=$MOE_EP, PP=$PP, DP=$DP
Runtime Optimizations: decode_cuda_graph_mode=$DECODE_CUDA_GRAPH_MODE, chunked_prefill=$ENABLE_CHUNKED_PREFILL, prefix_caching=true
Metrics: output_dir=$METRICS_OUTPUT_DIR, run_id=$RUN_ID
============================================
EOF

echo "Running simulation..."
if "${CMD[@]}"; then
  echo "Simulation completed successfully."
else
  exit_code=$?
  echo "Simulation failed (exit code: $exit_code)" >&2
  exit "$exit_code"
fi
