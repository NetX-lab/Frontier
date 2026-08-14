#!/usr/bin/env bash
# Run the approved C4 L2 trained configuration against an external profile snapshot.
# BRANCH_MODE=reference selects the historical Reference parser spellings.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 OUTPUT_DIR CACHE_DIR" >&2
  exit 2
fi

OUTPUT_DIR="$1"
CACHE_DIR="$2"
REPO_ROOT="$(pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
BRANCH_MODE="${BRANCH_MODE:-candidate}"
PROFILE_ROOT="${PROFILE_ROOT:?PROFILE_ROOT must point to the immutable external C4 profile root}"
ENABLE_EVENT_LOGGING="${ENABLE_EVENT_LOGGING:-true}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DISABLED=true
export VIDUR_DISABLE_WANDB=1

for file in linear_op.csv linear_op_kernel_only.csv attention.csv attention_kernel_only.csv moe.csv moe_kernel_only.csv; do
  test -f "$PROFILE_ROOT/compute/h800/step-moe-noquant/$file" || {
    echo "missing required profile: $PROFILE_ROOT/compute/h800/step-moe-noquant/$file" >&2
    exit 1
  }
done

CMD=(
  "$PYTHON_BIN" -m frontier.main
  --seed 42
  --simulation_mode offline
  --sys_arch pd-af-disaggregation
  --no-enable_parallel_clusters
  --cluster_config_prefill_cluster_num_replicas 1
  --cluster_config_decode_attn_cluster_num_replicas 1
  --cluster_config_decode_ffn_cluster_num_replicas 1
  --cluster_config_decode_attn_af_pipeline_num_micro_batch 1
  --cluster_config_decode_ffn_af_pipeline_num_micro_batch 1
  --cluster_config_decode_attn_micro_batch_size 8
  --cluster_config_prefill_replica_config_num_pipeline_stages 1
  --cluster_config_prefill_replica_config_attn_tensor_parallel_size 8
  --cluster_config_prefill_replica_config_attn_data_parallel_size 1
  --cluster_config_prefill_replica_config_moe_tensor_parallel_size 1
  --cluster_config_prefill_replica_config_moe_expert_parallel_size 8
  --cluster_config_prefill_replica_config_device h800
  --cluster_config_prefill_replica_config_memory_margin_fraction 0.2
  --cluster_config_decode_attn_replica_config_num_pipeline_stages 1
  --cluster_config_decode_attn_replica_config_attn_tensor_parallel_size 8
  --cluster_config_decode_attn_replica_config_attn_data_parallel_size 1
  --cluster_config_decode_attn_replica_config_device h800
  --cluster_config_decode_attn_replica_config_memory_margin_fraction 0.2
  --cluster_config_decode_ffn_replica_config_num_pipeline_stages 1
  --cluster_config_decode_ffn_replica_config_moe_tensor_parallel_size 1
  --cluster_config_decode_ffn_replica_config_moe_expert_parallel_size 8
  --cluster_config_decode_ffn_replica_config_device h800
  --cluster_config_decode_ffn_replica_config_memory_margin_fraction 0.2
  --cluster_config_prefill_replica_scheduler_config_type vllm_v1
  --cluster_config_decode_attn_replica_scheduler_config_type vllm_v1
  --cluster_config_decode_ffn_replica_scheduler_config_type orca
  --m2n_transfer_config_type analytical
  --replica_config_model_name step-moe-noquant
  --replica_config_moe_routing_distribution_type random
  --replica_config_moe_routing_seed 42
  --request_generator_config_type synthetic
  --synthetic_request_generator_config_num_requests 64
  --synthetic_request_generator_config_seed 42
  --fixed_request_length_generator_config_prefill_tokens 512
  --fixed_request_length_generator_config_decode_tokens 128
  --fixed_request_length_generator_config_seed 42
  --no-random_forrest_execution_time_predictor_config_enable_dummy_mode
  # C4 forms a 32-request prefill batch of 512 tokens each.  Attention and
  # stage predictors use the full effective batch token count (16384) as their
  # lookup key, so retain the measured 16384 point in the prediction table.
  --random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 16384
  --random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling
  --random_forrest_execution_time_predictor_config_linear_op_input_file "$PROFILE_ROOT/compute/h800/step-moe-noquant/linear_op.csv"
  --random_forrest_execution_time_predictor_config_linear_op_kernel_only_input_file "$PROFILE_ROOT/compute/h800/step-moe-noquant/linear_op_kernel_only.csv"
  --random_forrest_execution_time_predictor_config_atten_input_file "$PROFILE_ROOT/compute/h800/step-moe-noquant/attention.csv"
  --random_forrest_execution_time_predictor_config_atten_kernel_only_input_file "$PROFILE_ROOT/compute/h800/step-moe-noquant/attention_kernel_only.csv"
  --random_forrest_execution_time_predictor_config_moe_input_file "$PROFILE_ROOT/compute/h800/step-moe-noquant/moe.csv"
  --random_forrest_execution_time_predictor_config_moe_kernel_only_input_file "$PROFILE_ROOT/compute/h800/step-moe-noquant/moe_kernel_only.csv"
  --analytical_kv_cache_transfer_config_network_bandwidth_gbps 200.0
  --analytical_kv_cache_transfer_config_network_latency_ms 0.5
  --analytical_m2_n_transfer_config_memory_bandwidth_gbps 200.0
  --analytical_m2_n_transfer_config_network_latency_ms 0.05
  --metrics_config_output_dir "$OUTPUT_DIR"
  --metrics_config_cache_dir "$CACHE_DIR"
  --metrics_config_run_id c4
  --metrics_config_write_metrics
  --metrics_config_store_request_metrics
  --metrics_config_store_batch_metrics
  --metrics_config_store_token_completion_metrics
  --metrics_config_store_utilization_metrics
  --no-metrics_config_store_plots
  --no-metrics_config_enable_chrome_trace
  --no-metrics_config_write_json_trace
  --no-vllm_v1_scheduler_config_enable_chunked_prefill
)

if [[ "$ENABLE_EVENT_LOGGING" == "true" ]]; then
  CMD+=(--enable_cluster_event_logging --cluster_event_log_dir "$OUTPUT_DIR/events" --cluster_event_log_level INFO --cluster_log_filter PREFILL,DECODE_ATTN,DECODE_FFN)
elif [[ "$ENABLE_EVENT_LOGGING" != "false" ]]; then
  echo "ENABLE_EVENT_LOGGING must be true or false: $ENABLE_EVENT_LOGGING" >&2
  exit 2
fi

if [[ "$BRANCH_MODE" == "candidate" ]]; then
  CMD+=(--cc_backend_config_type analytical --length_generator_config_type fixed --interval_generator_config_type static --static_request_interval_generator_config_seed 42)
elif [[ "$BRANCH_MODE" == "reference" ]]; then
  CMD+=(--cluster_config_prefill_cc_backend_config_type analytical --cluster_config_decode_attn_cc_backend_config_type analytical --cluster_config_decode_ffn_cc_backend_config_type analytical --synthetic_request_generator_config_length_generator_config_type fixed --synthetic_request_generator_config_interval_generator_config_type static --static_request_interval_generator_config_seed 42)
else
  echo "unsupported BRANCH_MODE: $BRANCH_MODE" >&2
  exit 2
fi

exec "${CMD[@]}"
