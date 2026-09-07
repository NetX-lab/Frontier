#!/usr/bin/env bash
# Measure official server request TTFT for the frozen H200 case.
set -euo pipefail
source "$(dirname "$0")/issue26_h200_environment_probe.sh" "${1:?Provide a fresh output directory.}"
source /data/ycfeng/tmp/issue26-h200-network/company-proxy.sh
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" ALL_PROXY="$all_proxy"
export NO_PROXY="$no_proxy,127.0.0.1,localhost,::1" no_proxy="$no_proxy,127.0.0.1,localhost,::1"
export VLLM_V1_ALLOW_NO_CHUNKED_PREFILL=1 VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_FRONTIER_INSTRUMENTATION=0 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT="$TMPDIR/$(basename "$PROBE_ROOT")/vllm-cache"
export CUDA_CACHE_PATH="$TMPDIR/$(basename "$PROBE_ROOT")/cuda-cache"
export TRITON_CACHE_DIR="$TMPDIR/$(basename "$PROBE_ROOT")/triton-cache"
unset VLLM_FRONTIER_BATCH_LOG_PATH VLLM_FRONTIER_CUDA_EVENT_OP_LOG_PATH
unset VLLM_FRONTIER_SCHED_LOG_PATH VLLM_FRONTIER_MOE_ROUTING_LOG_PATH
unset VLLM_FRONTIER_RUNTIME_META_ENABLED VLLM_FRONTIER_PREFILL_ENDPOINT_LOG_PATH
unset VLLM_FRONTIER_SCHED_DECISION_LOG_PATH
export VLLM_ALL2ALL_BACKEND=naive
git config --global --add safe.directory /data/ycfeng/tmp/vLLM-BS
git -C /data/ycfeng/tmp/vLLM-BS rev-parse HEAD > "$PROBE_ROOT/vllm_commit.txt"
test "$(git -C /data/ycfeng/tmp/vLLM-BS rev-parse HEAD)" = 46f7b179fd3bf42b9616dc4670cba419afdb2085
test -z "$(git -C /data/ycfeng/tmp/vLLM-BS status --porcelain)"
git -C /data/ycfeng/tmp/vLLM-BS diff > "$PROBE_ROOT/vllm_uncommitted.patch"
SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID"
    wait "$SERVER_PID" || [[ "$?" -eq 143 ]]
  fi
}
trap cleanup EXIT
for MODE in clean; do
  RUN="$PROBE_ROOT/$MODE"
  mkdir "$RUN"
  export VLLM_FRONTIER_REQUEST_METRICS_LOG_PATH="$RUN/server.request_metrics.jsonl"
  "$PY" -m vllm.entrypoints.cli.main serve /data/ycfeng/tmp/issue26-qwen3-dummy \
    --tensor-parallel-size 4 --data-parallel-size 2 --enable-expert-parallel \
    --dtype bfloat16 --load-format dummy --max-model-len 16384 \
    --max-num-batched-tokens 16384 --max-num-seqs 1024 \
    --gpu-memory-utilization 0.9 --block-size 16 --num-gpu-blocks-override 310809 \
    --seed 0 --no-enable-chunked-prefill --no-enable-prefix-caching \
    --enforce-eager --served-model-name Qwen3-30B-A3B-Instruct-2507 \
    --skip-tokenizer-init > "$RUN/server.log" 2>&1 &
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
  "$PY" "$REPO_ROOT/tests/e2e/issue26_token_id_client.py" \
    --base-url http://127.0.0.1:8000 --model Qwen3-30B-A3B-Instruct-2507 \
    --row pf4096_dc1024 --prefill-tokens 4096 --decode-tokens 1024 \
    --requests 100 --warmups 3 --qps 2 --seed 20260908 \
    --output "$RUN/client.jsonl" > "$RUN/client.log" 2>&1
  cleanup
  SERVER_PID=""
done
echo OFFICIAL_TTFT_EXECUTION_COMPLETE
