#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VLLM_ROOT="$PROJECT_ROOT/sota-infer-engine/vllm"

NUM_REQUESTS="${NUM_REQUESTS:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/frontier_mla_deepseek_v2_flashinfer_mla_live_probe}"
MODEL_CONFIG_JSON="${MODEL_CONFIG_JSON:-$OUTPUT_DIR/model/config.json}"
PYTHON_BIN="${PYTHON_BIN:-/local/ycfeng/anaconda3/envs/frontier/bin/python}"
FRONTIER_CUDA_HOME="${FRONTIER_CUDA_HOME:-/usr/local/cuda-13.2}"
FLASHINFER_PYTHON_EXPECTED_VERSION="${FLASHINFER_PYTHON_EXPECTED_VERSION:-0.3.1.post1}"
VLLM_FRONTIER_COMPILED_PACKAGE="${VLLM_FRONTIER_COMPILED_PACKAGE:-/local/ycfeng/anaconda3/envs/frontier/lib/python3.10/site-packages/vllm}"
VLLM_FRONTIER_VLLM_FLASH_ATTN_PACKAGE="${VLLM_FRONTIER_VLLM_FLASH_ATTN_PACKAGE:-/local/ycfeng/anaconda3/envs/frontier/lib/python3.10/site-packages/vllm/vllm_flash_attn}"
FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp/frontier_flashinfer_workspace}"

GPU_DEVICES="${GPU_DEVICES:-0}"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
PREFILL_TOKENS="${PREFILL_TOKENS:-64}"
DECODE_TOKENS="${DECODE_TOKENS:-2}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-256}"
MAX_MODEL_LEN_BUFFER="${MAX_MODEL_LEN_BUFFER:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
ENFORCE_EAGER="${ENFORCE_EAGER:-True}"
CHUNKED_PREFILL_FLAG="${CHUNKED_PREFILL_FLAG:-False}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER_MLA}"
EXPECTED_BACKEND="${EXPECTED_BACKEND:-FLASHINFER_MLA}"
EXPECTED_BLOCK_SIZE="${EXPECTED_BLOCK_SIZE:-64}"
CUDA_EVENT_OP_SCOPES="${CUDA_EVENT_OP_SCOPES:-attn_mla_kv_cache_save,attn_mla_prefill_kv_up_proj,attn_mla_prefill,attn_mla_decode_q_latent_proj,attn_mla_decode,attn_mla_v_up_proj}"

MODEL_RUNTIME_DIR="$OUTPUT_DIR/model"
CUDA_EVENT_OP_LOG_PATH="$OUTPUT_DIR/cuda_ops.jsonl"
BATCH_LOG="$OUTPUT_DIR/batch_log.jsonl"
RESULT_CSV="$OUTPUT_DIR/ttft.csv"
RESULT_JSON="$OUTPUT_DIR/ttft.json"
RUN_LOG="$OUTPUT_DIR/run.log"
VALIDATION_LOG="$OUTPUT_DIR/validation.log"
RUN_FRONTIER_IMPORT_VALIDATION="${RUN_FRONTIER_IMPORT_VALIDATION:-1}"
FRONTIER_IMPORT_OUTPUT_DIR="${FRONTIER_IMPORT_OUTPUT_DIR:-$OUTPUT_DIR/frontier_import}"
FRONTIER_IMPORT_PROFILE_METHOD="${FRONTIER_IMPORT_PROFILE_METHOD:-${VLLM_FRONTIER_OP_TIMING_MODE:-record_function}}"
FRONTIER_IMPORT_MODEL_DIR="$FRONTIER_IMPORT_OUTPUT_DIR/compute/h100/deepseek-ai/DeepSeek-V2-Lite"
case "${VLLM_FRONTIER_OP_TIMING_MODE:-record_function}" in
  record_function)
    VLLM_FRONTIER_TIMING_FAMILY="KERNEL_ONLY"
    ;;
  cuda_event)
    VLLM_FRONTIER_TIMING_FAMILY="CUDA_EVENT"
    ;;
  *)
    echo "ERROR: VLLM_FRONTIER_OP_TIMING_MODE must be one of record_function or cuda_event. Got VLLM_FRONTIER_OP_TIMING_MODE=${VLLM_FRONTIER_OP_TIMING_MODE:-record_function}." >&2
    exit 6
    ;;
esac
case "$FRONTIER_IMPORT_PROFILE_METHOD" in
  record_function|kernel_only)
    FRONTIER_IMPORT_MEASUREMENT_FAMILY="KERNEL_ONLY"
    FRONTIER_IMPORT_PROFILE_CSV="$FRONTIER_IMPORT_MODEL_DIR/attention_kernel_only.csv"
    FRONTIER_IMPORT_SIDECAR="$FRONTIER_IMPORT_MODEL_DIR/attention_kernel_only_vllm_mla_groundtruth_comparison.csv"
    ;;
  cuda|cuda_event)
    FRONTIER_IMPORT_MEASUREMENT_FAMILY="CUDA_EVENT"
    FRONTIER_IMPORT_PROFILE_CSV="$FRONTIER_IMPORT_MODEL_DIR/attention.csv"
    FRONTIER_IMPORT_SIDECAR="$FRONTIER_IMPORT_MODEL_DIR/attention_vllm_mla_groundtruth_comparison.csv"
    ;;
  *)
    echo "ERROR: FRONTIER_IMPORT_PROFILE_METHOD must be one of record_function, kernel_only, cuda, or cuda_event. Got FRONTIER_IMPORT_PROFILE_METHOD=${FRONTIER_IMPORT_PROFILE_METHOD}." >&2
    exit 6
    ;;
esac
if [[ "$FRONTIER_IMPORT_MEASUREMENT_FAMILY" != "$VLLM_FRONTIER_TIMING_FAMILY" ]]; then
  echo "ERROR: FRONTIER_IMPORT_PROFILE_METHOD must match VLLM_FRONTIER_OP_TIMING_MODE measurement semantics. Got FRONTIER_IMPORT_PROFILE_METHOD=${FRONTIER_IMPORT_PROFILE_METHOD} (${FRONTIER_IMPORT_MEASUREMENT_FAMILY}) and VLLM_FRONTIER_OP_TIMING_MODE=${VLLM_FRONTIER_OP_TIMING_MODE:-record_function} (${VLLM_FRONTIER_TIMING_FAMILY})." >&2
  exit 6
fi

if [[ "$VLLM_ATTENTION_BACKEND" != "FLASHINFER_MLA" ]]; then
  echo "ERROR: DeepSeek-V2 MLA Stage 2 requires VLLM_ATTENTION_BACKEND=FLASHINFER_MLA. Got VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}." >&2
  exit 2
fi

if [[ "$TP_SIZE" != "1" ]]; then
  echo "ERROR: DeepSeek-V2 MLA Stage 2 live probe currently requires TP_SIZE=1 to preserve one-worker latent cache semantics. Got TP_SIZE=${TP_SIZE}." >&2
  exit 3
fi

if [[ ! -f "$VLLM_ROOT/examples/offline_inference/measure_ttft.py" ]]; then
  echo "ERROR: vLLM entrypoint not found: $VLLM_ROOT/examples/offline_inference/measure_ttft.py" >&2
  exit 4
fi

mkdir -p "$MODEL_RUNTIME_DIR"
: > "$CUDA_EVENT_OP_LOG_PATH"
: > "$BATCH_LOG"
: > "$RUN_LOG"
: > "$VALIDATION_LOG"

  cat > "$MODEL_RUNTIME_DIR/config.json" <<'JSON'
{
  "architectures": ["DeepseekV2ForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "aux_loss_alpha": 0.001,
  "bos_token_id": 100000,
  "eos_token_id": 100001,
  "ep_size": 1,
  "first_k_dense_replace": 0,
  "hidden_act": "silu",
  "hidden_size": 5120,
  "initializer_range": 0.02,
  "intermediate_size": 12288,
  "kv_lora_rank": 512,
  "max_position_embeddings": 4096,
  "model_type": "deepseek_v2",
  "moe_intermediate_size": 1407,
  "moe_layer_freq": 1,
  "n_group": 1,
  "n_routed_experts": 1,
  "n_shared_experts": 1,
  "norm_topk_prob": false,
  "num_attention_heads": 128,
  "num_experts_per_tok": 1,
  "num_hidden_layers": 1,
  "num_key_value_heads": 128,
  "pad_token_id": null,
  "pretraining_tp": 1,
  "q_lora_rank": 1536,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "rms_norm_eps": 1e-6,
  "rope_scaling": {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 40,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
    "type": "yarn"
  },
  "rope_theta": 10000.0,
  "routed_scaling_factor": 1.0,
  "scoring_func": "softmax",
  "seq_aux": true,
  "tie_word_embeddings": false,
  "topk_group": 1,
  "topk_method": "greedy",
  "torch_dtype": "bfloat16",
  "use_cache": true,
  "use_mla": true,
  "v_head_dim": 128,
  "vocab_size": 102400
}
JSON
export PYTHONNOUSERSITE=1
export FLASHINFER_WORKSPACE_BASE
export CUDA_HOME="$FRONTIER_CUDA_HOME"
export CUDA_PATH="$FRONTIER_CUDA_HOME"
export PYTORCH_NVCC="$FRONTIER_CUDA_HOME/bin/nvcc"
export PATH="$FRONTIER_CUDA_HOME/bin:$(dirname "$PYTHON_BIN"):$PATH"
export LD_LIBRARY_PATH="$FRONTIER_CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$FRONTIER_CUDA_HOME/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
export PYTHONPATH="$PROJECT_ROOT:$VLLM_ROOT:${PYTHONPATH:-}"
export FRONTIER_PATH="${FRONTIER_PATH:-$PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-1}"
export VLLM_V1_ENABLE_CHUNKED_PREFILL=0
export VLLM_V1_ALLOW_NO_CHUNKED_PREFILL=1
export VLLM_V1_ENABLE_PREFIX_CACHING=0
export VLLM_ATTENTION_BACKEND
export VLLM_CONFIG_ROOT="$VLLM_ROOT/.vllm_config"
export VLLM_FRONTIER_INSTRUMENTATION=1
export VLLM_FRONTIER_BATCH_LOG_PATH="$BATCH_LOG"
export VLLM_FRONTIER_TRACE_SKIP_WARMUP=1
export VLLM_FRONTIER_CUDA_EVENT_OP_LOG_PATH="$CUDA_EVENT_OP_LOG_PATH"
export VLLM_FRONTIER_CUDA_EVENT_OP_SCOPES="$CUDA_EVENT_OP_SCOPES"
export VLLM_FRONTIER_RUNTIME_META_ENABLED=1
export VLLM_FRONTIER_COMPILED_PACKAGE
export VLLM_FRONTIER_VLLM_FLASH_ATTN_PACKAGE
export VLLM_FRONTIER_OP_TIMING_MODE="${VLLM_FRONTIER_OP_TIMING_MODE:-record_function}"
export VLLM_FRONTIER_OP_AGG_MODE="${VLLM_FRONTIER_OP_AGG_MODE:-per_scope}"

if [[ ! -x "$PYTORCH_NVCC" ]]; then
  echo "ERROR: nvcc not found or not executable at PYTORCH_NVCC=${PYTORCH_NVCC}." >&2
  exit 5
fi

"$PYTHON_BIN" - "$FLASHINFER_PYTHON_EXPECTED_VERSION" <<'PY'
from importlib.metadata import version
import sys

expected_flashinfer_version = sys.argv[1]
flashinfer_version = version("flashinfer-python")
if flashinfer_version != expected_flashinfer_version:
    raise SystemExit(
        "ERROR: flashinfer-python version mismatch: "
        f"expected {expected_flashinfer_version}, got {flashinfer_version}"
    )
print(f"flashinfer-python={flashinfer_version}")
PY

cd "$VLLM_ROOT"
"$PYTHON_BIN" examples/offline_inference/measure_ttft.py \
  --model "$MODEL_RUNTIME_DIR" \
  --num-requests "$NUM_REQUESTS" \
  --prefill-tokens "$PREFILL_TOKENS" \
  --decode-tokens "$DECODE_TOKENS" \
  --seed 42 \
  --warmup-iters 0 \
  --tensor-parallel-size "$TP_SIZE" \
  --pipeline-parallel-size "$PP_SIZE" \
  --data-parallel-size "$DP_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --vocab-size 102400 \
  --load-format dummy \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-model-len-buffer "$MAX_MODEL_LEN_BUFFER" \
  --enable-chunked-prefill "$CHUNKED_PREFILL_FLAG" \
  --enable-profiler False \
  --enforce-eager "$ENFORCE_EAGER" \
  --results-csv "$RESULT_CSV" \
  --results-json "$RESULT_JSON" \
  > "$RUN_LOG" 2>&1

"$PYTHON_BIN" "$PROJECT_ROOT/tests/analysis/mla_deepseek_v2/validate_flashinfer_mla_live_probe.py" \
  --cuda-op-log "$CUDA_EVENT_OP_LOG_PATH" \
  --run-log "$RUN_LOG" \
  --expected-backend "$EXPECTED_BACKEND" \
  --expected-runtime-head-size 576 \
  --expected-runtime-kv-heads 1 \
  --expected-block-size "$EXPECTED_BLOCK_SIZE" \
  | tee "$VALIDATION_LOG"

if [[ "$RUN_FRONTIER_IMPORT_VALIDATION" == "1" ]]; then
  "$PYTHON_BIN" -m frontier.profiling.attention.main \
    --models deepseek-ai/DeepSeek-V2-Lite \
    --precision BF16 \
    --output_dir "$FRONTIER_IMPORT_OUTPUT_DIR" \
    --device h100 \
    --profile_method "$FRONTIER_IMPORT_PROFILE_METHOD" \
    --num_tensor_parallel_workers 1 \
    --max_model_len 163840 \
    --attention_backend FLASHINFER_MLA \
    --model_architecture_profile generic \
    --vllm_mla_cuda_op_log "$CUDA_EVENT_OP_LOG_PATH"

  "$PYTHON_BIN" "$PROJECT_ROOT/tests/analysis/mla_deepseek_v2/validate_flashinfer_mla_live_probe.py" \
    --cuda-op-log "$CUDA_EVENT_OP_LOG_PATH" \
    --run-log "$RUN_LOG" \
    --expected-backend "$EXPECTED_BACKEND" \
    --expected-runtime-head-size 576 \
    --expected-runtime-kv-heads 1 \
    --expected-block-size "$EXPECTED_BLOCK_SIZE" \
    --frontier-import-sidecar "$FRONTIER_IMPORT_SIDECAR" \
    --max-absolute-error-ms 0.0 \
    --max-relative-error-pct 0.0 \
    | tee -a "$VALIDATION_LOG"
fi

printf 'MLA DeepSeek-V2 FlashInfer live probe artifacts:\n'
printf '  output_dir=%s\n' "$OUTPUT_DIR"
printf '  cuda_ops=%s\n' "$CUDA_EVENT_OP_LOG_PATH"
printf '  run_log=%s\n' "$RUN_LOG"
printf '  validation_log=%s\n' "$VALIDATION_LOG"
if [[ "$RUN_FRONTIER_IMPORT_VALIDATION" == "1" ]]; then
  printf '  frontier_import_profile=%s\n' "$FRONTIER_IMPORT_PROFILE_CSV"
  printf '  frontier_import_sidecar=%s\n' "$FRONTIER_IMPORT_SIDECAR"
fi
