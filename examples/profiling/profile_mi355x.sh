#!/bin/bash
# =============================================================================
# MI355X (ROCm) end-to-end pipeline: profile -> train -> simulate
# =============================================================================
#   bash examples/profiling/profile_mi355x.sh attention   # host, needs torch only
#   bash examples/profiling/profile_mi355x.sh linear_moe  # INSIDE the vLLM container
#   bash examples/profiling/profile_mi355x.sh train_sim   # host, CPU only
#
# The stage split is not cosmetic: it is the host/container boundary.
# `attention` uses the TORCH_SDPA backend and attention/main.py only requires
# torch when the backend is not FlashInfer, so it runs on the host. `linear_moe`
# hard-requires an importable vllm and must run in the ROCm vLLM container (see
# MI355X_ROCM_COOKBOOK.md). `train_sim` is sklearn + discrete-event simulation,
# no GPU at all.
#
# Every stage is idempotent: it skips any (model, op) whose CSV already exists,
# so an interrupted run resumes rather than recollecting.
#
# ROCm specifics applied automatically:
#   - CUDA_VISIBLE_DEVICES is what the profiler's GPU discovery actually reads,
#     on AMD too; without it, it shells out to nvidia-smi and finds nothing.
#   - FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK=1 for linear_op: this vLLM's
#     get_rope() takes `rope_parameters`, not `rotary_dim`, so the wrapper call
#     would raise TypeError.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-mi355x}"
NETWORK_DEVICE="${NETWORK_DEVICE:-mi355x_8gpu}"
NUM_GPUS="${NUM_GPUS:-8}"
TP_SIZES="${TP_SIZES:-1 2 4 8}"
EP_SIZES="${EP_SIZES:-1 2 4 8}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/data/profiling}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/mi355x_profiling}"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/cache}"
METRICS_DIR="${METRICS_DIR:-$REPO_ROOT/outputs/mi355x}"

DENSE_MODELS="${DENSE_MODELS:-meta-llama/Llama-2-7b-hf}"
MOE_MODELS="${MOE_MODELS:-qwen3-a3b-30b-moe openai/gpt-oss-20b openai/gpt-oss-120b}"

# Profiling-time and training-time flags must agree on these or training
# silently filters the dataset to zero rows and fails.
ROUTING_RUNTIME_PATH="${ROUTING_RUNTIME_PATH:-standard_fused_topk}"
GATING_RUNTIME_CONTEXT="${GATING_RUNTIME_CONTEXT:-standalone_legacy}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"

# attention grid
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
MIN_BATCH_SIZE="${MIN_BATCH_SIZE:-1}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-16}"
FIXED_CHUNKED_PREFILL_SIZE="${FIXED_CHUNKED_PREFILL_SIZE:-64}"

# linear_op / moe grids
MAX_TOKENS="${MAX_TOKENS:-4096}"
# The MoE grid is TP x EP x num_tokens x load_distribution x samples. All 16
# TP/EP combinations are kept because the MoE predictor is trained per-(TP,EP)
# and never generalises across it. All three load distributions are requested
# explicitly — the profiler defaults to `uniform` alone, but the simulator's
# default routing mode (`simulation`) models expert-load skew, and a
# uniform-only predictor would be extrapolating on exactly that skew. The token
# axis is thinned instead: it feeds a smooth per-token curve, so ~50 points
# carry the same information as the default 259 and keep this at 7,056 rows per
# model instead of 37,296.
MOE_NUM_TOKENS="${MOE_NUM_TOKENS:-1 2 4 8 12 16 20 24 32 40 48 56 64 80 96 112 128 144 160 192 224 256 288 320 384 448 512 576 640 704 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048 2304 2560 2816 3072 3328 3584 3840 4096}"
MOE_LOAD_DISTRIBUTIONS="${MOE_LOAD_DISTRIBUTIONS:-uniform skewed extremely_skewed}"

# Qwen3-30B-A3B at exactly num_tokens=4000 with TP=1 faults the GPU on this
# ROCm/vLLM build ("Memory access fault ... Write access to a read-only page").
# Verified deterministic and narrow: every other point in the 259-value grid
# passes, TP=8 at 4000 passes, and Llama-2-7b and gpt-oss both pass at 4000.
# The fault kills the whole worker pool, so the point is excluded rather than
# retried; neighbours at 3968 and 4032 keep the grid dense there.
QWEN_MODEL_NAME="${QWEN_MODEL_NAME:-qwen3-a3b-30b-moe}"
QWEN_BAD_NUM_TOKENS="${QWEN_BAD_NUM_TOKENS:-4000}"

# simulation workload
TP="${TP:-8}"
EP="${EP:-1}"
NUM_REQUESTS="${NUM_REQUESTS:-32}"
PREFILL_TOKENS="${PREFILL_TOKENS:-512}"
DECODE_TOKENS="${DECODE_TOKENS:-128}"
QPS="${QPS:-1.0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
mkdir -p "$LOG_DIR" "$CACHE_DIR"

tag_of() { echo "$1" | tr '/' '_'; }

already_have() {
  local csv="$OUTPUT_DIR/compute/$DEVICE/$1/$2.csv"
  if [ -s "$csv" ]; then
    echo "  skip $2: $1 already collected"
    return 0
  fi
  return 1
}

# ---------------------------------------------------------------- attention --
stage_attention() {
  for model in $DENSE_MODELS $MOE_MODELS; do
    already_have "$model" attention && continue
    local log="$LOG_DIR/attention_$(tag_of "$model").log"
    echo "  attention: $model -> $log"
    # Invoked directly rather than through profile_attention_chunked_prefill.sh:
    # that wrapper hardcodes --profile_only_prefill, and attention/main.py
    # overwrites its CSV rather than appending, so collecting the two phases as
    # separate runs would destroy the first. Passing neither phase flag profiles
    # both at once.
    "$PYTHON_BIN" -m frontier.profiling.attention.main \
      --disable_ray --yes \
      --models "$model" \
      --num_gpus "$NUM_GPUS" \
      --max_seq_len "$MAX_SEQ_LEN" \
      --num_tensor_parallel_workers $TP_SIZES \
      --max_pipeline_parallel_size 1 \
      --attention_backend TORCH_SDPA \
      --block_size "$BLOCK_SIZE" \
      --min_batch_size "$MIN_BATCH_SIZE" \
      --max_batch_size "$MAX_BATCH_SIZE" \
      --fixed_chunked_prefill_size "$FIXED_CHUNKED_PREFILL_SIZE" \
      --enable_chunked_prefill_grid_search \
      --device "$DEVICE" \
      --profile_method cuda_event \
      --output_dir "$OUTPUT_DIR" > "$log" 2>&1
  done
}

# --------------------------------------------------------------- linear_moe --
qwen_token_grid() {
  local grid="1 2 4" step=8 t
  for ((t = 8; t <= MAX_TOKENS; t += step)); do
    [ "$t" -ne "$QWEN_BAD_NUM_TOKENS" ] && grid="$grid $t"
    if [ "$t" -ge 1024 ] && [ "$step" -eq 8 ]; then step=16
    elif [ "$t" -ge 2048 ] && [ "$step" -eq 16 ]; then step=32
    fi
  done
  echo "$grid"
}

stage_linear_moe() {
  export FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK=1

  for model in $DENSE_MODELS $MOE_MODELS; do
    already_have "$model" linear_op && continue
    local log="$LOG_DIR/linear_op_$(tag_of "$model").log"
    echo "  linear_op: $model -> $log"

    local -a token_args=(--max_tokens "$MAX_TOKENS")
    [ "$model" = "$QWEN_MODEL_NAME" ] && token_args=(--num_tokens_list $(qwen_token_grid))

    local -a moe_flag=()
    case " $MOE_MODELS " in *" $model "*) moe_flag=(--is_moe);; esac

    "$PYTHON_BIN" -m frontier.profiling.linear_op.main \
      --disable_ray --yes \
      --models "$model" \
      --num_gpus "$NUM_GPUS" \
      "${token_args[@]}" \
      --num_tensor_parallel_workers $TP_SIZES \
      --profile_method cuda_event \
      --device "$DEVICE" \
      --output_dir "$OUTPUT_DIR" \
      "${moe_flag[@]}" > "$log" 2>&1
  done

  for model in $MOE_MODELS; do
    already_have "$model" moe && continue
    local log="$LOG_DIR/moe_$(tag_of "$model").log"
    echo "  moe: $model -> $log"
    "$PYTHON_BIN" -m frontier.profiling.moe.main \
      --disable_ray --yes \
      --models "$model" \
      --num_gpus "$NUM_GPUS" \
      --num_tokens_list $MOE_NUM_TOKENS \
      --load_distributions $MOE_LOAD_DISTRIBUTIONS \
      --num_tensor_parallel_workers $TP_SIZES \
      --expert_parallel_sizes $EP_SIZES \
      --routing_runtime_path "$ROUTING_RUNTIME_PATH" \
      --gating_runtime_context "$GATING_RUNTIME_CONTEXT" \
      --profile_method cuda_event \
      --device "$DEVICE" \
      --output_dir "$OUTPUT_DIR" > "$log" 2>&1
  done
}

# ---------------------------------------------------------------- train_sim --
stage_train_sim() {
  local base_root="$OUTPUT_DIR/compute/$DEVICE"

  for model in $DENSE_MODELS $MOE_MODELS; do
    local base="$base_root/$model"
    local log="$LOG_DIR/train_$(tag_of "$model").log"
    local -a moe_flag=()
    case " $MOE_MODELS " in *" $model "*) moe_flag=(--is_moe);; esac
    : > "$log"
    echo "  train: $model"

    "$PYTHON_BIN" -m frontier.training.cli linear_op \
      --dataset_path "$base/linear_op.csv" \
      --output_dir "$CACHE_DIR" --measurement_type CUDA_EVENT \
      --model_name "$model" --device "$DEVICE" \
      --tensor_parallel_size "$TP" "${moe_flag[@]}" >> "$log" 2>&1

    "$PYTHON_BIN" -m frontier.training.cli attention \
      --layer_dataset_path "$base/attention.csv" \
      --compute_dataset_path "$base/linear_op.csv" \
      --output_dir "$CACHE_DIR" --measurement_type CUDA_EVENT \
      --block_size "$BLOCK_SIZE" \
      --model_name "$model" --device "$DEVICE" \
      --tensor_parallel_size "$TP" >> "$log" 2>&1

    if [ "${#moe_flag[@]}" -gt 0 ]; then
      "$PYTHON_BIN" -m frontier.training.cli moe \
        --dataset_path "$base/moe.csv" \
        --output_dir "$CACHE_DIR" --measurement_type CUDA_EVENT \
        --model_name "$model" --device "$DEVICE" \
        --moe_tensor_parallel_size "$TP" --expert_parallel_size "$EP" \
        --routing_runtime_path "$ROUTING_RUNTIME_PATH" \
        --gating_runtime_context "$GATING_RUNTIME_CONTEXT" >> "$log" 2>&1
    fi
  done

  for model in $DENSE_MODELS $MOE_MODELS; do
    local tag; tag=$(tag_of "$model")
    echo "  simulate: $model"
    if "$PYTHON_BIN" -m frontier.main \
      --simulation_mode offline --sys_arch co-location \
      --cc_backend_config_type vidur \
      --cluster_config_num_replicas 1 \
      --replica_config_device "$DEVICE" \
      --replica_config_network_device "$NETWORK_DEVICE" \
      --replica_config_model_name "$model" \
      --replica_config_attn_tensor_parallel_size "$TP" \
      --replica_config_attn_data_parallel_size 1 \
      --replica_config_moe_tensor_parallel_size "$TP" \
      --replica_config_moe_expert_parallel_size "$EP" \
      --replica_config_num_pipeline_stages 1 \
      --replica_scheduler_config_type vllm_v1 \
      --vllm_v1_scheduler_config_block_size "$BLOCK_SIZE" \
      --decode_cuda_graph_mode none \
      --request_generator_config_type synthetic \
      --synthetic_request_generator_config_num_requests "$NUM_REQUESTS" \
      --length_generator_config_type fixed \
      --fixed_request_length_generator_config_prefill_tokens "$PREFILL_TOKENS" \
      --fixed_request_length_generator_config_decode_tokens "$DECODE_TOKENS" \
      --fixed_request_length_generator_config_max_tokens $((PREFILL_TOKENS + DECODE_TOKENS + 384)) \
      --interval_generator_config_type poisson \
      --poisson_request_interval_generator_config_qps "$QPS" \
      --metrics_config_output_dir "$METRICS_DIR" \
      --metrics_config_run_id "${tag}_tp${TP}" \
      --metrics_config_write_metrics --metrics_config_store_request_metrics \
      --no-metrics_config_store_plots --no-metrics_config_enable_chrome_trace \
      --no-metrics_config_write_json_trace > "$LOG_DIR/sim_${tag}.log" 2>&1; then
      echo "    ok"
    else
      echo "    FAILED — see $LOG_DIR/sim_${tag}.log"
    fi
  done
}

case "${1:-}" in
  attention)  stage_attention ;;
  linear_moe) stage_linear_moe ;;
  train_sim)  stage_train_sim ;;
  *)
    echo "usage: $0 {attention|linear_moe|train_sim}" >&2
    echo "  attention   host        (torch only)" >&2
    echo "  linear_moe  container   (needs vllm)" >&2
    echo "  train_sim   host        (CPU only)" >&2
    exit 2 ;;
esac

echo "=== stage ${1} complete ==="
