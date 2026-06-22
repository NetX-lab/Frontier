#!/bin/bash
# =============================================================================
# PDD / pd-disaggregation Offline Mode - MoE Prefix Caching Recipe
# =============================================================================
# This script mirrors the co-location example surface while using the
# pre-release-v0.2 PDD / pd-disaggregation architecture: prefill runs in the PREFILL cluster,
# decode runs in the DECODE cluster, and KV cache is transferred between them.
#
# This recipe enables vLLM V1 Prefix Caching with a public shared-session trace
# fixture. Prefix Caching and Speculative Decoding are intentionally separate;
# this script does not enable speculative decoding.
## Override any uppercase variable from the shell, and append extra Frontier CLI
# flags after "--" if you need to customize the run.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DISABLED=true
export VIDUR_DISABLE_WANDB=1
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_NAME="${MODEL_NAME:-Phi-tiny-MoE-instruct}"
SYS_ARCH="${SYS_ARCH:-pd-disaggregation}"
PREFILL_REPLICAS="${PREFILL_REPLICAS:-2}"
DECODE_REPLICAS="${DECODE_REPLICAS:-2}"
PREFILL_ATTN_TP="${PREFILL_ATTN_TP:-2}"
PREFILL_ATTN_DP="${PREFILL_ATTN_DP:-1}"
PREFILL_MOE_TP="${PREFILL_MOE_TP:-1}"
PREFILL_MOE_EP="${PREFILL_MOE_EP:-2}"
PREFILL_PP="${PREFILL_PP:-1}"
PREFILL_DEVICE="${PREFILL_DEVICE:-a800}"
PREFILL_MEMORY_MARGIN_FRACTION="${PREFILL_MEMORY_MARGIN_FRACTION:-0.2}"
DECODE_ATTN_TP="${DECODE_ATTN_TP:-2}"
DECODE_ATTN_DP="${DECODE_ATTN_DP:-1}"
DECODE_MOE_TP="${DECODE_MOE_TP:-1}"
DECODE_MOE_EP="${DECODE_MOE_EP:-2}"
DECODE_PP="${DECODE_PP:-1}"
DECODE_DEVICE="${DECODE_DEVICE:-a800}"
DECODE_MEMORY_MARGIN_FRACTION="${DECODE_MEMORY_MARGIN_FRACTION:-0.2}"
TOTAL_EXPERTS="${TOTAL_EXPERTS:-8}"
ROUTER_TOPK="${ROUTER_TOPK:-2}"
MOE_ROUTING_MODE="${MOE_ROUTING_MODE:-simulation}"
MOE_ROUTING_SEED="${MOE_ROUTING_SEED:-42}"
REPLICA_SCHEDULER="${REPLICA_SCHEDULER:-vllm_v1}"
NUM_REQUESTS="${NUM_REQUESTS:-2}"
PREFILL_TOKENS="${PREFILL_TOKENS:-32}"
DECODE_TOKENS="${DECODE_TOKENS:-8}"
QPS="${QPS:-1.0}"
ENABLE_DUMMY_MODE="${ENABLE_DUMMY_MODE:-true}"
DUMMY_EXEC_TIME_MS="${DUMMY_EXEC_TIME_MS:-1.0}"
DECODE_CUDA_GRAPH_MODE="${DECODE_CUDA_GRAPH_MODE:-none}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-true}"
MAX_TOKENS_IN_BATCH="${MAX_TOKENS_IN_BATCH:-1024}"
LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-64}"
KV_TRANSFER_BANDWIDTH_GBPS="${KV_TRANSFER_BANDWIDTH_GBPS:-200.0}"
KV_TRANSFER_LATENCY_MS="${KV_TRANSFER_LATENCY_MS:-0.5}"
TRACE_FILE="${TRACE_FILE:-$REPO_ROOT/examples/fixtures/prefix_cache_shared_session_trace.csv}"
MAX_TOKENS="${MAX_TOKENS:-128}"
EXPECTED_TRACE_REQUESTS="${EXPECTED_TRACE_REQUESTS:-2}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
NUM_BLOCKS="${NUM_BLOCKS:-128}"
METRICS_OUTPUT_DIR="${METRICS_OUTPUT_DIR:-$REPO_ROOT/outputs/examples/pdd/offline}"
RUN_ID="${RUN_ID:-moe_prefix_caching}"

require_bool() {
  local name="$1"
  local value="$2"
  if [ "$value" != "true" ] && [ "$value" != "false" ]; then
    echo "ERROR: $name must be true or false; got $value" >&2
    exit 2
  fi
}

require_non_negative_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $name must be a non-negative integer; got $value" >&2
    exit 2
  fi
}

require_positive_integer() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]]
}

require_bool "ENABLE_DUMMY_MODE" "$ENABLE_DUMMY_MODE"
require_bool "ENABLE_CHUNKED_PREFILL" "$ENABLE_CHUNKED_PREFILL"

if [ "$SYS_ARCH" != "pd-disaggregation" ]; then
  echo "ERROR: this example only supports SYS_ARCH=pd-disaggregation; got SYS_ARCH=$SYS_ARCH" >&2
  exit 2
fi

if [ "$DECODE_CUDA_GRAPH_MODE" = "none" ]; then
  echo "INFO: Decode CUDA Graph modeling is disabled by DECODE_CUDA_GRAPH_MODE=none."
elif [ "$DECODE_CUDA_GRAPH_MODE" != "full_decode_only" ] && [ "$DECODE_CUDA_GRAPH_MODE" != "piecewise" ]; then
  echo "ERROR: DECODE_CUDA_GRAPH_MODE must be none, full_decode_only, or piecewise; got $DECODE_CUDA_GRAPH_MODE" >&2
  exit 2
fi

if [ "$ENABLE_CHUNKED_PREFILL" = "false" ] && [ "$LONG_PREFILL_TOKEN_THRESHOLD" != "0" ]; then
  echo "ERROR: LONG_PREFILL_TOKEN_THRESHOLD must be 0 when ENABLE_CHUNKED_PREFILL=false" >&2
  exit 2
fi

if (( PREFILL_ATTN_TP * PREFILL_ATTN_DP != PREFILL_MOE_TP * PREFILL_MOE_EP )); then
  echo "ERROR: shared-domain prefill MoE requires PREFILL_ATTN_TP * PREFILL_ATTN_DP == PREFILL_MOE_TP * PREFILL_MOE_EP" >&2
  echo "       got PREFILL_ATTN_TP=$PREFILL_ATTN_TP, PREFILL_ATTN_DP=$PREFILL_ATTN_DP, PREFILL_MOE_TP=$PREFILL_MOE_TP, PREFILL_MOE_EP=$PREFILL_MOE_EP" >&2
  exit 2
fi

if (( DECODE_ATTN_TP * DECODE_ATTN_DP != DECODE_MOE_TP * DECODE_MOE_EP )); then
  echo "ERROR: shared-domain decode MoE requires DECODE_ATTN_TP * DECODE_ATTN_DP == DECODE_MOE_TP * DECODE_MOE_EP" >&2
  echo "       got DECODE_ATTN_TP=$DECODE_ATTN_TP, DECODE_ATTN_DP=$DECODE_ATTN_DP, DECODE_MOE_TP=$DECODE_MOE_TP, DECODE_MOE_EP=$DECODE_MOE_EP" >&2
  exit 2
fi

if [ ! -f "$TRACE_FILE" ]; then
  echo "ERROR: TRACE_FILE does not exist: $TRACE_FILE" >&2
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
  --no-enable_parallel_clusters
  --cluster_config_prefill_cluster_num_replicas "$PREFILL_REPLICAS"
  --cluster_config_decode_cluster_num_replicas "$DECODE_REPLICAS"
  --cluster_scheduler_config_type sticky_round_robin
  --cluster_config_prefill_replica_config_num_pipeline_stages "$PREFILL_PP"
  --cluster_config_prefill_replica_config_attn_tensor_parallel_size "$PREFILL_ATTN_TP"
  --cluster_config_prefill_replica_config_attn_data_parallel_size "$PREFILL_ATTN_DP"
  --cluster_config_prefill_replica_config_moe_tensor_parallel_size "$PREFILL_MOE_TP"
  --cluster_config_prefill_replica_config_moe_expert_parallel_size "$PREFILL_MOE_EP"
  --cluster_config_prefill_replica_config_total_expert_num "$TOTAL_EXPERTS"
  --cluster_config_prefill_replica_config_router_topk "$ROUTER_TOPK"
  --cluster_config_prefill_replica_config_device "$PREFILL_DEVICE"
  --cluster_config_prefill_replica_config_memory_margin_fraction "$PREFILL_MEMORY_MARGIN_FRACTION"
  --cluster_config_decode_replica_config_num_pipeline_stages "$DECODE_PP"
  --cluster_config_decode_replica_config_attn_tensor_parallel_size "$DECODE_ATTN_TP"
  --cluster_config_decode_replica_config_attn_data_parallel_size "$DECODE_ATTN_DP"
  --cluster_config_decode_replica_config_moe_tensor_parallel_size "$DECODE_MOE_TP"
  --cluster_config_decode_replica_config_moe_expert_parallel_size "$DECODE_MOE_EP"
  --cluster_config_decode_replica_config_total_expert_num "$TOTAL_EXPERTS"
  --cluster_config_decode_replica_config_router_topk "$ROUTER_TOPK"
  --cluster_config_decode_replica_config_device "$DECODE_DEVICE"
  --cluster_config_decode_replica_config_memory_margin_fraction "$DECODE_MEMORY_MARGIN_FRACTION"
  --cc_backend_config_type analytical
  --replica_config_model_name "$MODEL_NAME"
  --replica_config_moe_routing_mode "$MOE_ROUTING_MODE"
  --replica_config_moe_routing_seed "$MOE_ROUTING_SEED"
  --replica_scheduler_config_type "$REPLICA_SCHEDULER"
  --decode_cuda_graph_mode "$DECODE_CUDA_GRAPH_MODE"
  --vllm_v1_scheduler_config_max_tokens_in_batch "$MAX_TOKENS_IN_BATCH"
  --vllm_v1_scheduler_config_long_prefill_token_threshold "$LONG_PREFILL_TOKEN_THRESHOLD"
  --vllm_v1_scheduler_config_block_size "${BLOCK_SIZE:-16}"
  --vllm_v1_scheduler_config_num_blocks "${NUM_BLOCKS:-128}"
  --request_generator_config_type trace_replay
  --trace_request_generator_config_trace_file "$TRACE_FILE"
  --trace_request_generator_config_max_tokens "$MAX_TOKENS"
  --analytical_kv_cache_transfer_config_network_bandwidth_gbps "$KV_TRANSFER_BANDWIDTH_GBPS"
  --analytical_kv_cache_transfer_config_network_latency_ms "$KV_TRANSFER_LATENCY_MS"
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

CMD+=(--vllm_v1_scheduler_config_enable_prefix_caching)

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
=========================================================
  PDD / pd-disaggregation Mode - MoE Prefix Caching Recipe
=========================================================
PYTHONPATH: $PYTHONPATH
Model: $MODEL_NAME
Architecture: $SYS_ARCH
Simulation Mode: offline
Prefill cluster replicas: $PREFILL_REPLICAS
Decode cluster replicas: $DECODE_REPLICAS
Prefill parallelism: Attn_TP=$PREFILL_ATTN_TP, MoE_TP=$PREFILL_MOE_TP, MoE_EP=$PREFILL_MOE_EP, PP=$PREFILL_PP, DP=$PREFILL_ATTN_DP
Decode parallelism: Attn_TP=$DECODE_ATTN_TP, MoE_TP=$DECODE_MOE_TP, MoE_EP=$DECODE_MOE_EP, PP=$DECODE_PP, DP=$DECODE_ATTN_DP
MoE: total_experts=$TOTAL_EXPERTS, router_topk=$ROUTER_TOPK, routing=$MOE_ROUTING_MODE, seed=$MOE_ROUTING_SEED
Scheduler: $REPLICA_SCHEDULER
Trace: $TRACE_FILE
Expected Trace Shape: requests=$EXPECTED_TRACE_REQUESTS from TRACE_FILE, repeated block_hash_ids produce cache-hit blocks
Prefix Caching: enabled, block_size=$BLOCK_SIZE, num_blocks=$NUM_BLOCKS
Requests: $NUM_REQUESTS (prefill=$PREFILL_TOKENS, decode=$DECODE_TOKENS, qps=$QPS)
Runtime Optimizations: decode_cuda_graph_mode=$DECODE_CUDA_GRAPH_MODE, chunked_prefill=$ENABLE_CHUNKED_PREFILL
KV transfer: bandwidth_gbps=$KV_TRANSFER_BANDWIDTH_GBPS, latency_ms=$KV_TRANSFER_LATENCY_MS
Metrics: output_dir=$METRICS_OUTPUT_DIR, run_id=$RUN_ID
Dummy Mode: $ENABLE_DUMMY_MODE
=========================================================
EOF

echo "Running simulation..."
if "${CMD[@]}"; then
  echo "Simulation completed successfully."
else
  exit_code=$?
  echo "Simulation failed (exit code: $exit_code)" >&2
  exit "$exit_code"
fi
