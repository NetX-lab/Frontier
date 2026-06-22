#!/bin/bash
# =============================================================================
# PDD / pd-disaggregation Offline Mode - MoE Speculative Decoding / MTP Recipe
# =============================================================================
# This script mirrors the co-location example surface while using the
# pre-release-v0.2 PDD / pd-disaggregation architecture: prefill runs in the PREFILL cluster,
# decode runs in the DECODE cluster, and KV cache is transferred between them.
#
# Speculative decoding and Prefix Caching have separate runtime contracts.
# This recipe enables speculative decoding and intentionally leaves Prefix
# Caching disabled. It also defaults DECODE_CUDA_GRAPH_MODE to "none" because
# production speculative decoding requires eager decode scheduling.
#
# For MTP-style methods, set SPEC_METHOD to an MTP method and keep
# MTP_N_PREDICT / MTP_NUM_LAYERS positive.
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
PREFILL_REPLICAS="${PREFILL_REPLICAS:-1}"
DECODE_REPLICAS="${DECODE_REPLICAS:-1}"
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
NUM_REQUESTS="${NUM_REQUESTS:-8}"
PREFILL_TOKENS="${PREFILL_TOKENS:-256}"
DECODE_TOKENS="${DECODE_TOKENS:-32}"
QPS="${QPS:-1.0}"
ENABLE_DUMMY_MODE="${ENABLE_DUMMY_MODE:-true}"
DUMMY_EXEC_TIME_MS="${DUMMY_EXEC_TIME_MS:-1.0}"
DECODE_CUDA_GRAPH_MODE="${DECODE_CUDA_GRAPH_MODE:-none}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-true}"
MAX_TOKENS_IN_BATCH="${MAX_TOKENS_IN_BATCH:-1024}"
LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-64}"
KV_TRANSFER_BANDWIDTH_GBPS="${KV_TRANSFER_BANDWIDTH_GBPS:-200.0}"
KV_TRANSFER_LATENCY_MS="${KV_TRANSFER_LATENCY_MS:-0.5}"
SPEC_METHOD="${SPEC_METHOD:-ngram}"
SPEC_MODEL_NAME="${SPEC_MODEL_NAME:-}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-2}"
COMMITTED_TOKENS_PER_ITERATION="${COMMITTED_TOKENS_PER_ITERATION:-2}"
PROPOSER_OVERHEAD_MS_BY_METHOD="${PROPOSER_OVERHEAD_MS_BY_METHOD:-{\"ngram\":0.0,\"qwen3_next_mtp\":0.0,\"deepseek_mtp\":0.0,\"ernie_mtp\":0.0}}"
MTP_N_PREDICT="${MTP_N_PREDICT:-0}"
MTP_NUM_LAYERS="${MTP_NUM_LAYERS:-0}"
METRICS_OUTPUT_DIR="${METRICS_OUTPUT_DIR:-$REPO_ROOT/outputs/examples/pdd/offline}"
RUN_ID="${RUN_ID:-moe_spec_dec}"

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

require_non_negative_integer "MTP_N_PREDICT" "$MTP_N_PREDICT"
require_non_negative_integer "MTP_NUM_LAYERS" "$MTP_NUM_LAYERS"

if [ "$DECODE_CUDA_GRAPH_MODE" != "none" ]; then
  echo "ERROR: speculative decoding currently requires DECODE_CUDA_GRAPH_MODE=none in production recipes; got $DECODE_CUDA_GRAPH_MODE" >&2
  exit 2
fi

case "$SPEC_METHOD" in
  deepseek_mtp|ernie_mtp|qwen3_moe_mtp|qwen3_next_mtp)
    if ! require_positive_integer "MTP_N_PREDICT" "$MTP_N_PREDICT" || ! require_positive_integer "MTP_NUM_LAYERS" "$MTP_NUM_LAYERS"; then
      echo "ERROR: SPEC_METHOD=$SPEC_METHOD requires MTP_N_PREDICT>0 and MTP_NUM_LAYERS>0" >&2
      exit 2
    fi
    ;;
  *)
    if [ "$MTP_N_PREDICT" -ne 0 ] || [ "$MTP_NUM_LAYERS" -ne 0 ]; then
      echo "ERROR: MTP_N_PREDICT/MTP_NUM_LAYERS are only valid for MTP SPEC_METHOD values" >&2
      exit 2
    fi
    ;;
esac

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
  --request_generator_config_type synthetic
  --synthetic_request_generator_config_num_requests "$NUM_REQUESTS"
  --length_generator_config_type fixed
  --fixed_request_length_generator_config_prefill_tokens "$PREFILL_TOKENS"
  --fixed_request_length_generator_config_decode_tokens "$DECODE_TOKENS"
  --interval_generator_config_type poisson
  --poisson_request_interval_generator_config_qps "$QPS"
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

CMD+=(
  --speculative_decoding_config_enabled
  --speculative_decoding_config_method "$SPEC_METHOD"
  --speculative_decoding_config_spec_model_name "$SPEC_MODEL_NAME"
  --speculative_decoding_config_num_speculative_tokens "$NUM_SPECULATIVE_TOKENS"
  --speculative_decoding_config_committed_tokens_per_iteration "$COMMITTED_TOKENS_PER_ITERATION"
  --speculative_decoding_config_proposer_overhead_ms_by_method "$PROPOSER_OVERHEAD_MS_BY_METHOD"
)

if [ "$MTP_N_PREDICT" -gt 0 ]; then
  CMD+=(--speculative_decoding_config_mtp_n_predict "$MTP_N_PREDICT")
fi

if [ "$MTP_NUM_LAYERS" -gt 0 ]; then
  CMD+=(--speculative_decoding_config_mtp_num_layers "$MTP_NUM_LAYERS")
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
=========================================================
  PDD / pd-disaggregation Mode - MoE Speculative Decoding / MTP Recipe
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
Requests: $NUM_REQUESTS (prefill=$PREFILL_TOKENS, decode=$DECODE_TOKENS, qps=$QPS)
Speculative Decoding: method=$SPEC_METHOD, num_speculative_tokens=$NUM_SPECULATIVE_TOKENS, committed_tokens_per_iteration=$COMMITTED_TOKENS_PER_ITERATION
MTP: n_predict=$MTP_N_PREDICT, num_layers=$MTP_NUM_LAYERS
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
