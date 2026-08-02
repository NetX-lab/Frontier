#!/bin/bash
# =============================================================================
# PD-AF / pd-af-disaggregation Online Mode - MoE Model Basic Example
# =============================================================================
# This script demonstrates the pd-af-disaggregation architecture with three
# disaggregated clusters: PREFILL, DECODE_ATTN, and DECODE_FFN.
#
# Architecture:
#   Prefill cluster  -> processes prefill phase (attention + MoE together)
#   Decode-Attn cluster -> processes decode attention phase
#   Decode-FFN cluster  -> processes decode MoE/FFN phase
#
# KV cache is transferred from PREFILL -> DECODE_ATTN.
# M2N (activation) data is transferred between DECODE_ATTN <-> DECODE_FFN.
#
# This example uses EP=1 (no expert parallelism) with dummy mode enabled.
# Override any uppercase variable from the shell, and append extra Frontier CLI
# flags after "--" if you need to customize the run.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DISABLED=true
export VIDUR_DISABLE_WANDB=1
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_NAME="${MODEL_NAME:-Phi-tiny-MoE-instruct}"
SYS_ARCH="${SYS_ARCH:-pd-af-disaggregation}"
EXAMPLE_LABEL="${EXAMPLE_LABEL:-MoE Basic Online (EP=1)}"

# --- Cluster replica counts ---
PREFILL_REPLICAS="${PREFILL_REPLICAS:-1}"
DECODE_ATTN_REPLICAS="${DECODE_ATTN_REPLICAS:-1}"
DECODE_FFN_REPLICAS="${DECODE_FFN_REPLICAS:-1}"

# --- AF Pipeline configuration ---
DECODE_ATTN_AF_MICRO_BATCH="${DECODE_ATTN_AF_MICRO_BATCH:-1}"
DECODE_FFN_AF_MICRO_BATCH="${DECODE_FFN_AF_MICRO_BATCH:-1}"
DECODE_ATTN_MICRO_BATCH_SIZE="${DECODE_ATTN_MICRO_BATCH_SIZE:-8}"

# --- Prefill cluster parallelism ---
PREFILL_PP="${PREFILL_PP:-1}"
PREFILL_ATTN_TP="${PREFILL_ATTN_TP:-1}"
PREFILL_ATTN_DP="${PREFILL_ATTN_DP:-1}"
PREFILL_MOE_TP="${PREFILL_MOE_TP:-1}"
PREFILL_MOE_EP="${PREFILL_MOE_EP:-1}"
PREFILL_DEVICE="${PREFILL_DEVICE:-a800}"
PREFILL_MEMORY_MARGIN_FRACTION="${PREFILL_MEMORY_MARGIN_FRACTION:-0.2}"

# --- Decode-Attn cluster parallelism ---
DECODE_ATTN_PP="${DECODE_ATTN_PP:-1}"
DECODE_ATTN_TP="${DECODE_ATTN_TP:-1}"
DECODE_ATTN_DP="${DECODE_ATTN_DP:-1}"
DECODE_ATTN_DEVICE="${DECODE_ATTN_DEVICE:-a800}"
DECODE_ATTN_MEMORY_MARGIN_FRACTION="${DECODE_ATTN_MEMORY_MARGIN_FRACTION:-0.2}"

# --- Decode-FFN cluster parallelism ---
DECODE_FFN_PP="${DECODE_FFN_PP:-1}"
DECODE_FFN_MOE_TP="${DECODE_FFN_MOE_TP:-1}"
DECODE_FFN_MOE_EP="${DECODE_FFN_MOE_EP:-1}"
DECODE_FFN_DEVICE="${DECODE_FFN_DEVICE:-a800}"
DECODE_FFN_MEMORY_MARGIN_FRACTION="${DECODE_FFN_MEMORY_MARGIN_FRACTION:-0.2}"

# --- MoE configuration ---
TOTAL_EXPERTS="${TOTAL_EXPERTS:-16}"
ROUTER_TOPK="${ROUTER_TOPK:-2}"
MOE_ROUTING_MODE="${MOE_ROUTING_MODE:-simulation}"
MOE_ROUTING_SEED="${MOE_ROUTING_SEED:-42}"

# --- Per-cluster scheduler types ---
PREFILL_SCHEDULER="${PREFILL_SCHEDULER:-vllm_v1}"
DECODE_ATTN_SCHEDULER="${DECODE_ATTN_SCHEDULER:-vllm_v1}"
DECODE_FFN_SCHEDULER="${DECODE_FFN_SCHEDULER:-orca}"

# --- Workload configuration ---
NUM_REQUESTS="${NUM_REQUESTS:-8}"
PREFILL_TOKENS="${PREFILL_TOKENS:-256}"
DECODE_TOKENS="${DECODE_TOKENS:-32}"
QPS="${QPS:-1.0}"

# --- Runtime configuration ---
ENABLE_DUMMY_MODE="${ENABLE_DUMMY_MODE:-true}"
DUMMY_EXEC_TIME_MS="${DUMMY_EXEC_TIME_MS:-1.0}"
ENABLE_CUDA_GRAPH="${ENABLE_CUDA_GRAPH:-false}"
MAX_TOKENS_IN_BATCH="${MAX_TOKENS_IN_BATCH:-1024}"
LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-64}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-true}"

# --- KV transfer (prefill -> decode_attn) ---
KV_TRANSFER_BANDWIDTH_GBPS="${KV_TRANSFER_BANDWIDTH_GBPS:-200.0}"
KV_TRANSFER_LATENCY_MS="${KV_TRANSFER_LATENCY_MS:-0.5}"

# --- M2N transfer (decode_attn <-> decode_ffn) ---
M2N_BANDWIDTH_GBPS="${M2N_BANDWIDTH_GBPS:-200.0}"
M2N_LATENCY_MS="${M2N_LATENCY_MS:-0.05}"

# --- Metrics ---
METRICS_OUTPUT_DIR="${METRICS_OUTPUT_DIR:-$REPO_ROOT/outputs/examples/pd-af-disagg/online}"
RUN_ID="${RUN_ID:-moe_model_basic_online}"

# =============================================================================
# Validation helpers
# =============================================================================

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

require_bool "ENABLE_DUMMY_MODE" "$ENABLE_DUMMY_MODE"
require_bool "ENABLE_CHUNKED_PREFILL" "$ENABLE_CHUNKED_PREFILL"
require_bool "ENABLE_CUDA_GRAPH" "$ENABLE_CUDA_GRAPH"

if [ "$SYS_ARCH" != "pd-af-disaggregation" ]; then
  echo "ERROR: this example only supports SYS_ARCH=pd-af-disaggregation; got SYS_ARCH=$SYS_ARCH" >&2
  exit 2
fi

if (( PREFILL_ATTN_TP * PREFILL_ATTN_DP != PREFILL_MOE_TP * PREFILL_MOE_EP )); then
  echo "ERROR: prefill cluster requires PREFILL_ATTN_TP * PREFILL_ATTN_DP == PREFILL_MOE_TP * PREFILL_MOE_EP" >&2
  echo "       got PREFILL_ATTN_TP=$PREFILL_ATTN_TP, PREFILL_ATTN_DP=$PREFILL_ATTN_DP, PREFILL_MOE_TP=$PREFILL_MOE_TP, PREFILL_MOE_EP=$PREFILL_MOE_EP" >&2
  exit 2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: PYTHON_BIN is not executable or not on PATH: $PYTHON_BIN" >&2
  exit 2
fi

# =============================================================================
# Build command
# =============================================================================

CMD=(
  "$PYTHON_BIN" -m frontier.main
  --simulation_mode online
  --sys_arch "$SYS_ARCH"
  --no-enable_parallel_clusters

  # Cluster replica counts
  --cluster_config_prefill_cluster_num_replicas "$PREFILL_REPLICAS"
  --cluster_config_decode_attn_cluster_num_replicas "$DECODE_ATTN_REPLICAS"
  --cluster_config_decode_ffn_cluster_num_replicas "$DECODE_FFN_REPLICAS"

  # AF Pipeline micro-batch
  --cluster_config_decode_attn_af_pipeline_num_micro_batch "$DECODE_ATTN_AF_MICRO_BATCH"
  --cluster_config_decode_ffn_af_pipeline_num_micro_batch "$DECODE_FFN_AF_MICRO_BATCH"
  --cluster_config_decode_attn_micro_batch_size "$DECODE_ATTN_MICRO_BATCH_SIZE"

  # Prefill cluster replica config
  --cluster_config_prefill_replica_config_num_pipeline_stages "$PREFILL_PP"
  --cluster_config_prefill_replica_config_attn_tensor_parallel_size "$PREFILL_ATTN_TP"
  --cluster_config_prefill_replica_config_attn_data_parallel_size "$PREFILL_ATTN_DP"
  --cluster_config_prefill_replica_config_moe_tensor_parallel_size "$PREFILL_MOE_TP"
  --cluster_config_prefill_replica_config_moe_expert_parallel_size "$PREFILL_MOE_EP"
  --cluster_config_prefill_replica_config_total_expert_num "$TOTAL_EXPERTS"
  --cluster_config_prefill_replica_config_router_topk "$ROUTER_TOPK"
  --cluster_config_prefill_replica_config_device "$PREFILL_DEVICE"
  --cluster_config_prefill_replica_config_memory_margin_fraction "$PREFILL_MEMORY_MARGIN_FRACTION"

  # Decode-Attn cluster replica config
  --cluster_config_decode_attn_replica_config_num_pipeline_stages "$DECODE_ATTN_PP"
  --cluster_config_decode_attn_replica_config_attn_tensor_parallel_size "$DECODE_ATTN_TP"
  --cluster_config_decode_attn_replica_config_attn_data_parallel_size "$DECODE_ATTN_DP"
  --cluster_config_decode_attn_replica_config_device "$DECODE_ATTN_DEVICE"
  --cluster_config_decode_attn_replica_config_memory_margin_fraction "$DECODE_ATTN_MEMORY_MARGIN_FRACTION"

  # Decode-FFN cluster replica config
  --cluster_config_decode_ffn_replica_config_num_pipeline_stages "$DECODE_FFN_PP"
  --cluster_config_decode_ffn_replica_config_moe_tensor_parallel_size "$DECODE_FFN_MOE_TP"
  --cluster_config_decode_ffn_replica_config_moe_expert_parallel_size "$DECODE_FFN_MOE_EP"
  --cluster_config_decode_ffn_replica_config_total_expert_num "$TOTAL_EXPERTS"
  --cluster_config_decode_ffn_replica_config_router_topk "$ROUTER_TOPK"
  --cluster_config_decode_ffn_replica_config_device "$DECODE_FFN_DEVICE"
  --cluster_config_decode_ffn_replica_config_memory_margin_fraction "$DECODE_FFN_MEMORY_MARGIN_FRACTION"

  # Per-cluster scheduler types
  --cluster_config_prefill_replica_scheduler_config_type "$PREFILL_SCHEDULER"
  --cluster_config_decode_attn_replica_scheduler_config_type "$DECODE_ATTN_SCHEDULER"
  --cluster_config_decode_ffn_replica_scheduler_config_type "$DECODE_FFN_SCHEDULER"

  # Backend config
  --cc_backend_config_type analytical
  --m2n_transfer_config_type analytical

  # Model / MoE routing
  --replica_config_model_name "$MODEL_NAME"
  --replica_config_moe_routing_mode "$MOE_ROUTING_MODE"
  --replica_config_moe_routing_seed "$MOE_ROUTING_SEED"

  # Scheduler parameters
  --vllm_v1_scheduler_config_max_tokens_in_batch "$MAX_TOKENS_IN_BATCH"
  --vllm_v1_scheduler_config_long_prefill_token_threshold "$LONG_PREFILL_TOKEN_THRESHOLD"
  --vllm_v1_scheduler_config_block_size "${BLOCK_SIZE:-16}"
  --vllm_v1_scheduler_config_num_blocks "${NUM_BLOCKS:-128}"

  # Workload
  --request_generator_config_type synthetic
  --synthetic_request_generator_config_num_requests "$NUM_REQUESTS"
  --length_generator_config_type fixed
  --fixed_request_length_generator_config_prefill_tokens "$PREFILL_TOKENS"
  --fixed_request_length_generator_config_decode_tokens "$DECODE_TOKENS"
  --interval_generator_config_type poisson
  --poisson_request_interval_generator_config_qps "$QPS"

  # KV transfer (prefill -> decode_attn)
  --analytical_kv_cache_transfer_config_network_bandwidth_gbps "$KV_TRANSFER_BANDWIDTH_GBPS"
  --analytical_kv_cache_transfer_config_network_latency_ms "$KV_TRANSFER_LATENCY_MS"

  # M2N transfer (decode_attn <-> decode_ffn)
  --analytical_m2_n_transfer_config_memory_bandwidth_gbps "$M2N_BANDWIDTH_GBPS"
  --analytical_m2_n_transfer_config_network_latency_ms "$M2N_LATENCY_MS"

  # Metrics
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

if [ "$ENABLE_CUDA_GRAPH" = "true" ]; then
  CMD+=(
    --use_cuda_graph
    --cudagraph_capture_sizes 8 16 32 64
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
  PD-AF / pd-af-disaggregation Mode - $EXAMPLE_LABEL
=========================================================
PYTHONPATH: $PYTHONPATH
Model: $MODEL_NAME
Architecture: $SYS_ARCH
Simulation Mode: online
Prefill cluster replicas: $PREFILL_REPLICAS
Decode-Attn cluster replicas: $DECODE_ATTN_REPLICAS
Decode-FFN cluster replicas: $DECODE_FFN_REPLICAS
AF Pipeline: decode_attn_micro_batch=$DECODE_ATTN_AF_MICRO_BATCH, decode_ffn_micro_batch=$DECODE_FFN_AF_MICRO_BATCH
Prefill parallelism: PP=$PREFILL_PP, Attn_TP=$PREFILL_ATTN_TP, Attn_DP=$PREFILL_ATTN_DP, MoE_TP=$PREFILL_MOE_TP, MoE_EP=$PREFILL_MOE_EP
Decode-Attn parallelism: PP=$DECODE_ATTN_PP, Attn_TP=$DECODE_ATTN_TP, Attn_DP=$DECODE_ATTN_DP
Decode-FFN parallelism: PP=$DECODE_FFN_PP, MoE_TP=$DECODE_FFN_MOE_TP, MoE_EP=$DECODE_FFN_MOE_EP
Model topology: total_experts=$TOTAL_EXPERTS, router_topk=$ROUTER_TOPK, routing=$MOE_ROUTING_MODE
Schedulers: prefill=$PREFILL_SCHEDULER, decode_attn=$DECODE_ATTN_SCHEDULER, decode_ffn=$DECODE_FFN_SCHEDULER
Requests: $NUM_REQUESTS (prefill=$PREFILL_TOKENS, decode=$DECODE_TOKENS, qps=$QPS)
KV transfer: bandwidth_gbps=$KV_TRANSFER_BANDWIDTH_GBPS, latency_ms=$KV_TRANSFER_LATENCY_MS
M2N transfer: bandwidth_gbps=$M2N_BANDWIDTH_GBPS, latency_ms=$M2N_LATENCY_MS
Metrics: output_dir=$METRICS_OUTPUT_DIR, run_id=$RUN_ID
Dummy Mode: $ENABLE_DUMMY_MODE
CUDA Graph: $ENABLE_CUDA_GRAPH
=========================================================
EOF

echo "Running online simulation..."
if "${CMD[@]}"; then
  echo "Online simulation completed successfully."
else
  exit_code=$?
  echo "Simulation failed (exit code: $exit_code)" >&2
  exit "$exit_code"
fi
