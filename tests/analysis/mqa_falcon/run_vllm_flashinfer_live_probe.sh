#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VLLM_ROOT="$PROJECT_ROOT/sota-infer-engine/vllm"

NUM_REQUESTS="${NUM_REQUESTS:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/frontier_mqa_falcon_flashinfer_live_probe}"
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
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
EXPECTED_BACKEND="${EXPECTED_BACKEND:-FLASHINFER_VLLM_V1}"
CUDA_EVENT_OP_SCOPES="${CUDA_EVENT_OP_SCOPES:-attn_kv_cache_save,attn_prefill,attn_decode}"

MODEL_RUNTIME_DIR="$OUTPUT_DIR/model"
CUDA_EVENT_OP_LOG_PATH="$OUTPUT_DIR/cuda_ops.jsonl"
BATCH_LOG="$OUTPUT_DIR/batch_log.jsonl"
RESULT_CSV="$OUTPUT_DIR/ttft.csv"
RESULT_JSON="$OUTPUT_DIR/ttft.json"
RUN_LOG="$OUTPUT_DIR/run.log"
VALIDATION_LOG="$OUTPUT_DIR/validation.log"

if [[ "$VLLM_ATTENTION_BACKEND" != "FLASHINFER" ]]; then
  echo "ERROR: Falcon MQA Stage 2 requires VLLM_ATTENTION_BACKEND=FLASHINFER. Got VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}." >&2
  exit 2
fi

if [[ "$TP_SIZE" != "1" ]]; then
  echo "ERROR: Native Falcon-7B MQA profiling requires TP_SIZE=1 because 71 Q heads are not divisible by common TP sizes. Got TP_SIZE=${TP_SIZE}." >&2
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

if [[ ! -f "$MODEL_RUNTIME_DIR/config.json" ]]; then
  cat > "$MODEL_RUNTIME_DIR/config.json" <<'JSON'
{
  "architectures": ["FalconForCausalLM"],
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "bias": false,
  "eos_token_id": 11,
  "hidden_dropout": 0.0,
  "hidden_size": 4544,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "falcon",
  "multi_query": true,
  "new_decoder_architecture": false,
  "num_attention_heads": 71,
  "num_hidden_layers": 1,
  "parallel_attn": true,
  "alibi": false,
  "torch_dtype": "bfloat16",
  "use_cache": true,
  "vocab_size": 65024
}
JSON
fi

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
  --vocab-size 65024 \
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

"$PYTHON_BIN" "$PROJECT_ROOT/tests/analysis/mqa_falcon/validate_flashinfer_live_probe.py" \
  --cuda-op-log "$CUDA_EVENT_OP_LOG_PATH" \
  --run-log "$RUN_LOG" \
  --expected-backend "$EXPECTED_BACKEND" \
  --expected-head-dim 64 \
  --expected-q-heads 71 \
  --expected-kv-heads 1 \
  --expected-tp "$TP_SIZE" \
  --expected-chunked-prefill-enabled "$CHUNKED_PREFILL_FLAG" \
  | tee "$VALIDATION_LOG"

printf 'MQA Falcon FlashInfer live probe artifacts:\n'
printf '  output_dir=%s\n' "$OUTPUT_DIR"
printf '  cuda_ops=%s\n' "$CUDA_EVENT_OP_LOG_PATH"
printf '  run_log=%s\n' "$RUN_LOG"
printf '  validation_log=%s\n' "$VALIDATION_LOG"
