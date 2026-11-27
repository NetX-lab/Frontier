#!/bin/bash
# Qwen3-4B PD-AF simulation with the filled profiling package.
#
# Data layout expected under data/profiling:
#   compute/hpp2/Qwen3-4B/linear_op_kernel_only.csv
#   compute/h200/Qwen3-4B/{linear_op.csv,linear_op_kernel_only.csv,
#                          attention.csv,attention_kernel_only.csv}
#   cpu_overhead/hpp2_16rank/Qwen3-4B/cpu_overheads_kernel_only.csv
#   network/a2f_h200_to_hpp2/m2n_transfer_lut.csv
#   network/hpp2_16rank/{all_reduce.csv,weight_stream_lut.csv}
#   network/h200_prefill_to_decode_attn/kv_cache_transfer_lut.csv
#
# The analytical M2N/KV/CC settings below are fitted from those LUTs:
#   M2N Direct: 366 Gbps, 0 latency, no 1.2x p2p speedup
#   KV NVLink:  3172.5 Gbps, 0.0208 ms latency
#   HPP2 ring:  2984 Gbps intra-node (373 GB/s), 0 latency

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DISABLED=true
export VIDUR_DISABLE_WANDB=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ID="${RUN_ID:-qwen3_4b_pd_af_baseline}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/outputs/qwen3-4b-pd-af}"
NUM_REQUESTS="${NUM_REQUESTS:-8}"
PREFILL_TOKENS="${PREFILL_TOKENS:-512}"
DECODE_TOKENS="${DECODE_TOKENS:-64}"
ARRIVAL_PATTERN="${ARRIVAL_PATTERN:-burst}"
WEIGHT_STREAM_LAYERS="${WEIGHT_STREAM_LAYERS:-0}"
AF_MICRO_BATCH="${AF_MICRO_BATCH:-1}"
DECODE_ATTN_MICRO_BATCH_SIZE="${DECODE_ATTN_MICRO_BATCH_SIZE:-8}"

cd "$REPO_ROOT"

MODEL_JSON="data/config/models/Qwen3-4B.json"
REQUIRED=(
  "$MODEL_JSON"
  "data/profiling/compute/hpp2/Qwen3-4B/linear_op_kernel_only.csv"
  "data/profiling/compute/h200/Qwen3-4B/linear_op.csv"
  "data/profiling/compute/h200/Qwen3-4B/linear_op_kernel_only.csv"
  "data/profiling/compute/h200/Qwen3-4B/attention.csv"
  "data/profiling/compute/h200/Qwen3-4B/attention_kernel_only.csv"
  "data/profiling/cpu_overhead/hpp2_16rank/Qwen3-4B/cpu_overheads_kernel_only.csv"
  "data/profiling/network/a2f_h200_to_hpp2/m2n_transfer_lut.csv"
  "data/profiling/network/hpp2_16rank/all_reduce.csv"
  "data/profiling/network/hpp2_16rank/weight_stream_lut.csv"
  "data/profiling/network/h200_prefill_to_decode_attn/kv_cache_transfer_lut.csv"
)
for path in "${REQUIRED[@]}"; do
  if [ ! -f "$path" ]; then
    echo "ERROR: missing required profiling file: $path" >&2
    exit 2
  fi
done

case "$WEIGHT_STREAM_LAYERS" in
  0) CPU_OVERHEAD_TEMPLATE="" ;;
  1|2|3)
    STREAM_CPU_OVERHEAD_FILE="data/profiling/cpu_overhead/hpp2_16rank/Qwen3-4B/cpu_overheads_kernel_only_stream${WEIGHT_STREAM_LAYERS}.csv"
    if [ ! -f "$STREAM_CPU_OVERHEAD_FILE" ]; then
      echo "ERROR: missing stream CPU-overhead table: $STREAM_CPU_OVERHEAD_FILE" >&2
      exit 2
    fi
    # Keep {NETWORK_DEVICE}/{MODEL} placeholders so only hpp2_16rank resolves
    # the HPP2 streaming table while h200_dgx stays unpopulated (host term = 0).
    CPU_OVERHEAD_TEMPLATE="./data/profiling/cpu_overhead/{NETWORK_DEVICE}/{MODEL}/cpu_overheads_kernel_only_stream${WEIGHT_STREAM_LAYERS}.csv"
    ;;
  *)
    echo "ERROR: WEIGHT_STREAM_LAYERS must be 0, 1, 2, or 3; got $WEIGHT_STREAM_LAYERS" >&2
    exit 2
    ;;
esac

case "$ARRIVAL_PATTERN" in
  burst)
    ARRIVAL_ARGS=(--interval_generator_config_type static)
    ARRIVAL_DESCRIPTION="all requests arrive at time 0"
    if [ -n "${QPS:-}" ]; then
      echo "Note: QPS is ignored for a burst; set ARRIVAL_PATTERN=poisson to use QPS." >&2
    fi
    ;;
  poisson)
    QPS="${QPS:-1.0}"
    ARRIVAL_ARGS=(--interval_generator_config_type poisson
                  --poisson_request_interval_generator_config_qps "$QPS"
                  --offline_use_generated_request_arrivals)
    ARRIVAL_DESCRIPTION="Poisson arrivals, offered QPS=$QPS"
    ;;
  *)
    echo "ERROR: ARRIVAL_PATTERN must be burst or poisson; got $ARRIVAL_PATTERN" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT_DIR"

CMD=(
  "$PYTHON_BIN" -m frontier.main
  --simulation_mode offline
  --sys_arch pd-af-disaggregation
  --no-enable_parallel_clusters

  --cluster_config_prefill_cluster_num_replicas 1
  --cluster_config_decode_attn_cluster_num_replicas 1
  --cluster_config_decode_ffn_cluster_num_replicas 1

  --cluster_config_decode_attn_af_pipeline_num_micro_batch "$AF_MICRO_BATCH"
  --cluster_config_decode_ffn_af_pipeline_num_micro_batch "$AF_MICRO_BATCH"
  --cluster_config_decode_attn_micro_batch_size "$DECODE_ATTN_MICRO_BATCH_SIZE"

  # PREFILL: 8x H200, dense attention TP8.  The dense FFN follows attn TP8;
  # moe_tp=8 only satisfies the shared role contract and is unused by a dense FFN.
  --cluster_config_prefill_replica_config_num_pipeline_stages 1
  --cluster_config_prefill_replica_config_attn_tensor_parallel_size 8
  --cluster_config_prefill_replica_config_moe_tensor_parallel_size 8
  --cluster_config_prefill_replica_config_moe_expert_parallel_size 1
  --cluster_config_prefill_replica_config_total_expert_num 1
  --cluster_config_prefill_replica_config_router_topk 1
  --cluster_config_prefill_replica_config_device h200
  --cluster_config_prefill_replica_config_network_device h200_dgx
  --cluster_config_prefill_replica_config_memory_margin_fraction 0.2

  # DECODE_ATTN: 8x H200, dense attention TP8.
  --cluster_config_decode_attn_replica_config_num_pipeline_stages 1
  --cluster_config_decode_attn_replica_config_attn_tensor_parallel_size 8
  --cluster_config_decode_attn_replica_config_device h200
  --cluster_config_decode_attn_replica_config_network_device h200_dgx
  --cluster_config_decode_attn_replica_config_memory_margin_fraction 0.2

  # DECODE_FFN: 8x HPP2 = 16 dies/ranks, dense FFN TP16.
  # The CPU-overhead loader selects moe_tensor_parallel_size for DECODE_FFN,
  # so attn_tensor_parallel_size stays at its default.
  --cluster_config_decode_ffn_replica_config_num_pipeline_stages 1
  --cluster_config_decode_ffn_replica_config_moe_tensor_parallel_size 16
  --cluster_config_decode_ffn_replica_config_moe_expert_parallel_size 1
  --cluster_config_decode_ffn_replica_config_total_expert_num 1
  --cluster_config_decode_ffn_replica_config_router_topk 1
  --cluster_config_decode_ffn_replica_config_device hpp2
  --cluster_config_decode_ffn_replica_config_streamed_ffn_layers "$WEIGHT_STREAM_LAYERS"
  --cluster_config_decode_ffn_replica_config_network_device hpp2_16rank
  --cluster_config_decode_ffn_replica_config_memory_margin_fraction 0.0

  --cluster_config_prefill_replica_scheduler_config_type vllm_v1
  --cluster_config_decode_attn_replica_scheduler_config_type vllm_v1
  --cluster_config_decode_ffn_replica_scheduler_config_type orca

  --cc_backend_config_type analytical
  --m2n_transfer_config_type analytical
  --kv_cache_transfer_config_type analytical

  # Fitted from the filled M2N / KV / all-reduce LUTs.
  --analytical_m2_n_transfer_config_memory_bandwidth_gbps 366.0
  --analytical_m2_n_transfer_config_network_latency_ms 0.0
  --no-analytical_m2_n_transfer_config_enable_p2p_optimization
  --analytical_kv_cache_transfer_config_network_bandwidth_gbps 3172.5
  --analytical_kv_cache_transfer_config_network_latency_ms 0.0208
  --analytical_cc_backend_config_intra_node_bandwidth_gbps 2984.0
  --analytical_cc_backend_config_network_bandwidth_gbps 292.8
  --analytical_cc_backend_config_network_latency_us 0.0

  --replica_config_model_name Qwen3-4B
  --replica_config_moe_routing_distribution_type balanced
  --replica_config_moe_routing_seed 42

  --vllm_v1_scheduler_config_max_tokens_in_batch 8192
  --vllm_v1_scheduler_config_long_prefill_token_threshold 64
  --vllm_v1_scheduler_config_block_size 16
  --vllm_v1_scheduler_config_num_blocks 4096
  --vllm_v1_scheduler_config_enable_chunked_prefill

  --request_generator_config_type synthetic
  --synthetic_request_generator_config_num_requests "$NUM_REQUESTS"
  --length_generator_config_type fixed
  --fixed_request_length_generator_config_prefill_tokens "$PREFILL_TOKENS"
  --fixed_request_length_generator_config_decode_tokens "$DECODE_TOKENS"
  "${ARRIVAL_ARGS[@]}"

  # Use the filled HPP2 CPU-overhead table.  H200 clusters have no table and
  # will log a warning and price their host terms as 0.
  --no-random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling
  --random_forrest_execution_time_predictor_config_k_fold_cv_splits 3

  --metrics_config_output_dir "$OUT_DIR"
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

if [ -n "$CPU_OVERHEAD_TEMPLATE" ]; then
  CMD+=(
    --random_forrest_execution_time_predictor_config_cpu_overhead_kernel_only_input_file
    "$CPU_OVERHEAD_TEMPLATE"
  )
fi

if [ "$#" -gt 0 ]; then
  CMD+=("$@")
fi

echo "========================================================="
echo " Qwen3-4B PD-AF"
echo " model: Qwen3-4B"
echo " PREFILL:    h200 x8, attn TP8"
echo " DECODE_ATTN:h200 x8, attn TP8"
echo " DECODE_FFN: hpp2 x8/16 ranks, FFN TP16"
echo " requests:   $NUM_REQUESTS, prefill=$PREFILL_TOKENS, decode=$DECODE_TOKENS; $ARRIVAL_DESCRIPTION"
echo " weight stream: $WEIGHT_STREAM_LAYERS layer(s) (0 = all resident)"
echo " metrics:    $OUT_DIR/$RUN_ID"
echo "========================================================="
exec "${CMD[@]}"