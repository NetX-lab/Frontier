#!/usr/bin/env bash
# Capture a first-formal Kineto trace on the clean naive H200 path.
set -euo pipefail

RUN_ROOT="${1:?Provide a fresh output directory under the active task directory.}"
SOURCE="${ISSUE26_DIAGNOSTIC_VLLM_SOURCE:-/data/ycfeng/tmp/vLLM-BS}"
COMMIT="${ISSUE26_DIAGNOSTIC_VLLM_COMMIT:-46f7b179fd3bf42b9616dc4670cba419afdb2085}"
WORKERS="$(cd "$(dirname "$0")" && pwd)"

source "$WORKERS/issue26_h200_environment_probe.sh" "$RUN_ROOT/preflight"
source /data/ycfeng/tmp/issue26-h200-network/company-proxy.sh
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" ALL_PROXY="$all_proxy"
export NO_PROXY="$no_proxy,127.0.0.1,localhost,::1" no_proxy="$NO_PROXY"

git config --global --add safe.directory "$SOURCE"
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$SOURCE" status --porcelain)"
export PYTHONPATH="$SOURCE:$REPO_ROOT/tests/e2e"
export VLLM_MOE_UNIFORM_ROUTING=1
export VLLM_V1_ALLOW_NO_CHUNKED_PREFILL=1 VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_ALL2ALL_BACKEND=naive
export VLLM_FRONTIER_INSTRUMENTATION=0
unset VLLM_FRONTIER_BATCH_LOG_PATH VLLM_FRONTIER_CUDA_EVENT_OP_LOG_PATH
unset VLLM_FRONTIER_SCHED_LOG_PATH VLLM_FRONTIER_MOE_ROUTING_LOG_PATH
unset VLLM_FRONTIER_RUNTIME_META_ENABLED VLLM_FRONTIER_PREFILL_ENDPOINT_LOG_PATH
unset VLLM_FRONTIER_SCHED_DECISION_LOG_PATH VLLM_FRONTIER_DP_ROUTE_LOG_PATH
unset VLLM_FRONTIER_PROFILE_REQUEST_PREFIX VLLM_FRONTIER_PROFILE_BATCH_LIMIT
unset VLLM_FRONTIER_CUDA_EVENT_OP_SCOPES
export VLLM_TORCH_PROFILER_DIR="$RUN_ROOT/torch-profiler"
export VLLM_TORCH_PROFILER_RECORD_SHAPES=0
export VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY=0
export VLLM_TORCH_PROFILER_WITH_STACK=0
export VLLM_TORCH_PROFILER_WITH_FLOPS=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
export VLLM_CACHE_ROOT="$TMPDIR/$(basename "$RUN_ROOT")/vllm-cache"
export CUDA_CACHE_PATH="$TMPDIR/$(basename "$RUN_ROOT")/cuda-cache"
export TRITON_CACHE_DIR="$TMPDIR/$(basename "$RUN_ROOT")/triton-cache"
mkdir -p "$VLLM_TORCH_PROFILER_DIR"

printf '%s\n' "$(git -C "$SOURCE" rev-parse HEAD)" > "$RUN_ROOT/vllm_commit.txt"
printf '%s\n' "$(git -C "$SOURCE" status --porcelain)" > "$RUN_ROOT/vllm_status.txt"
cat > "$RUN_ROOT/profile_manifest.json" <<EOF
{
  "gpu": "H200",
  "cluster": "step_main",
  "num_gpu_blocks_override": 310809,
  "backend": "naive",
  "workload": "4096-prefill/1024-output",
  "warmup_replays": 10,
  "formal_requests": 100,
  "profiler": "torch.profiler (Kineto)",
  "operator_instrumentation": false,
  "capture_control": "vLLM /start_profile after warmup drain; /stop_profile on first formal first token",
  "limits": "Profiler trace is diagnostic and may perturb timing; no Frontier operator logger is enabled."
}
EOF

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID"
    wait "$SERVER_PID" || [[ "$?" -eq 143 ]]
  fi
}
trap cleanup EXIT

mkdir -p "$RUN_ROOT/runtime"
"$PY" -m vllm.entrypoints.cli.main serve /data/ycfeng/tmp/issue26-qwen3-dummy \
  --tensor-parallel-size 4 --data-parallel-size 2 --enable-expert-parallel \
  --dtype bfloat16 --load-format dummy --max-model-len 16384 \
  --max-num-batched-tokens 16384 --max-num-seqs 1024 \
  --gpu-memory-utilization 0.9 --block-size 16 --num-gpu-blocks-override 310809 \
  --seed 0 --no-enable-chunked-prefill --no-enable-prefix-caching \
  --enforce-eager --served-model-name Qwen3-30B-A3B-Instruct-2507 \
  --skip-tokenizer-init > "$RUN_ROOT/runtime/server.log" 2>&1 &
SERVER_PID=$!
"$PY" - "$SERVER_PID" <<'PY'
import os
import sys
import time
import urllib.error
import urllib.request

deadline = time.monotonic() + 900
while time.monotonic() < deadline:
    os.kill(int(sys.argv[1]), 0)
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as response:
            if response.status == 200:
                break
    except (urllib.error.URLError, TimeoutError):
        time.sleep(2)
else:
    raise TimeoutError("vLLM health endpoint did not become ready within 900 seconds")
PY

"$PY" "$WORKERS/issue26_naive_profiler_client.py" \
  --base-url http://127.0.0.1:8000 \
  --model Qwen3-30B-A3B-Instruct-2507 \
  --row pf4096_dc1024 --prefill-tokens 4096 --decode-tokens 1024 \
  --requests 100 --warmups "${ISSUE26_WARMUPS:-10}" --qps 2 --seed 20260908 \
  --output "$RUN_ROOT/runtime/client.jsonl" \
  > "$RUN_ROOT/runtime/client.log" 2>&1

test "$(wc -l < "$RUN_ROOT/runtime/client.jsonl")" -eq "$(((${ISSUE26_WARMUPS:-10}) + 1) * 100)"
find "$RUN_ROOT/torch-profiler" -type f -name '*.json.gz' -print > "$RUN_ROOT/trace_files.txt"
test -s "$RUN_ROOT/trace_files.txt"
TRACE_COUNT="$(wc -l < "$RUN_ROOT/trace_files.txt")"
test "$TRACE_COUNT" -ge 4
printf '{"trace_count":%s,"gpu_trace_minimum":4}\n' "$TRACE_COUNT" \
  > "$RUN_ROOT/trace_manifest.json"
echo NAIVE_KINETO_PROFILER_EXECUTION_COMPLETE
