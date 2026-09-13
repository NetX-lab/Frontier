#!/usr/bin/env bash
# Capture a reduced-overhead first-formal CUDA window with Nsight Systems.
# This variant intentionally omits NCCL API/GPU tracing; CUDA GPU activity still
# exposes NCCL kernel names for category classification while reducing CUPTI hooks.
set -euo pipefail

RUN_ROOT="${1:?Provide a fresh output directory under the active task directory.}"
SOURCE="${ISSUE26_DIAGNOSTIC_VLLM_SOURCE:-/data/ycfeng/tmp/vLLM-BS}"
COMMIT="${ISSUE26_DIAGNOSTIC_VLLM_COMMIT:-0f34fb271fd66d7dd84201ebdd4722781f829390}"
BACKEND="${ISSUE26_NSYS_BACKEND:-naive}"
WORKERS="$(cd "$(dirname "$0")" && pwd)"

case "$BACKEND" in
  naive|pplx) ;;
  *) echo "Unsupported ISSUE26_NSYS_BACKEND: $BACKEND" >&2; exit 1 ;;
esac

source "$WORKERS/issue26_h200_environment_probe.sh" "$RUN_ROOT/preflight"
source /data/ycfeng/tmp/issue26-h200-network/company-proxy.sh
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" ALL_PROXY="$all_proxy"
export NO_PROXY="${no_proxy:-${NO_PROXY:-}},127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"

git config --global --add safe.directory "$SOURCE"
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$SOURCE" status --porcelain)"
export PYTHONPATH="$REPO_ROOT/tests/e2e:$SOURCE"
if [[ -n "${ISSUE26_OPTIONAL_PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$ISSUE26_OPTIONAL_PYTHONPATH:$PYTHONPATH"
fi
if [[ -n "${ISSUE26_EXTRA_LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="$ISSUE26_EXTRA_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
fi
if [[ -n "${ISSUE26_LD_PRELOAD:-}" ]]; then
  export LD_PRELOAD="$ISSUE26_LD_PRELOAD"
fi
export ISSUE26_NSYS_CONTROL_DIR="$RUN_ROOT/nsys-control"
export VLLM_MOE_UNIFORM_ROUTING=1
export VLLM_V1_ALLOW_NO_CHUNKED_PREFILL=1 VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_ALL2ALL_BACKEND="$BACKEND"
export VLLM_FRONTIER_INSTRUMENTATION=0
export ISSUE26_NSYS_CAPTURE_STRICT=1
unset VLLM_FRONTIER_BATCH_LOG_PATH VLLM_FRONTIER_CUDA_EVENT_OP_LOG_PATH
unset VLLM_FRONTIER_SCHED_LOG_PATH VLLM_FRONTIER_MOE_ROUTING_LOG_PATH
unset VLLM_FRONTIER_RUNTIME_META_ENABLED VLLM_FRONTIER_PREFILL_ENDPOINT_LOG_PATH
unset VLLM_FRONTIER_SCHED_DECISION_LOG_PATH VLLM_FRONTIER_DP_ROUTE_LOG_PATH
unset VLLM_FRONTIER_PROFILE_REQUEST_PREFIX VLLM_FRONTIER_PROFILE_BATCH_LIMIT
unset VLLM_FRONTIER_CUDA_EVENT_OP_SCOPES
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
export VLLM_CACHE_ROOT="$TMPDIR/$(basename "$RUN_ROOT")/vllm-cache"
export CUDA_CACHE_PATH="$TMPDIR/$(basename "$RUN_ROOT")/cuda-cache"
export TRITON_CACHE_DIR="$TMPDIR/$(basename "$RUN_ROOT")/triton-cache"
mkdir -p "$RUN_ROOT/runtime" "$RUN_ROOT/nsys" "$ISSUE26_NSYS_CONTROL_DIR"

NSYS_TARBALL=/data/ycfeng/tmp/issue26-nsys-runtime-20260913-01.tgz
NSYS_SHA256=576c7a2b38d6db2d2ec69256b88346ae5adca6baf922665c8f7dc76ac335e1b4
NSYS_HOST_TARBALL=/data/ycfeng/tmp/issue26-nsys-host-runtime-20260913-01.tgz
NSYS_HOST_SHA256=b858b6448f5aef90a1ed32ee51cb2896ea19ef5716fee7c32b527b3fa7fcfe92
test -s "$NSYS_TARBALL"
test "$(sha256sum "$NSYS_TARBALL" | awk '{print $1}')" = "$NSYS_SHA256"
test -s "$NSYS_HOST_TARBALL"
test "$(sha256sum "$NSYS_HOST_TARBALL" | awk '{print $1}')" = "$NSYS_HOST_SHA256"
NSYS_ROOT="$TMPDIR/issue26-nsys-runtime-20260913-01"
mkdir -p "$NSYS_ROOT"
if [[ ! -x "$NSYS_ROOT/target-linux-x64/nsys" ]]; then
  tar -xzf "$NSYS_TARBALL" -C "$NSYS_ROOT"
fi
if [[ ! -x "$NSYS_ROOT/host-linux-x64/QdstrmImporter" ]]; then
  tar -xzf "$NSYS_HOST_TARBALL" -C "$NSYS_ROOT"
fi
NSYS="$NSYS_ROOT/target-linux-x64/nsys"
test -x "$NSYS"
test -x "$NSYS_ROOT/host-linux-x64/QdstrmImporter"
TARGET_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PATH="$NSYS_ROOT/target-linux-x64:$PATH"
export LD_LIBRARY_PATH="$NSYS_ROOT/host-linux-x64:$NSYS_ROOT/target-linux-x64:${LD_LIBRARY_PATH:-}"
"$NSYS" --version > "$RUN_ROOT/nsys/version.txt"

cat > "$RUN_ROOT/profile_manifest.json" <<EOF
{
  "gpu": "H200",
  "cluster": "step_main",
  "num_gpu_blocks_override": 310809,
  "backend": "$BACKEND",
  "workload": "4096-prefill/1024-output",
  "warmup_replays": 10,
  "formal_requests": 100,
  "profiler": "Nsight Systems CUDA/CUPTI (CUDA-only trace)",
  "operator_instrumentation": false,
  "capture_control": "cudaProfilerStart before first formal model forward; cudaProfilerStop immediately after that forward returns",
  "nsys_runtime_tarball": "$NSYS_TARBALL",
  "nsys_runtime_sha256": "$NSYS_SHA256",
  "nsys_host_runtime_tarball": "$NSYS_HOST_TARBALL",
  "nsys_host_runtime_sha256": "$NSYS_HOST_SHA256",
  "limits": "Low-perturbation diagnostic decomposition; clean batch span remains the acceptance metric."
}
EOF

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

"$NSYS" profile \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace=cuda \
  --sample=none \
  --cpuctxsw=none \
  --cuda-memory-usage=false \
  --cuda-event-trace=false \
  --flush-on-cudaprofilerstop=false \
  --cuda-trace-scope=process-tree \
  --trace-fork-before-exec=false \
  --show-output=false \
  --wait=all \
  --env-var="LD_LIBRARY_PATH=$TARGET_LD_LIBRARY_PATH" \
  --output="$RUN_ROOT/nsys/first_formal" \
  --force-overwrite=true \
  -- "$PY" -m vllm.entrypoints.cli.main serve /data/ycfeng/tmp/issue26-qwen3-dummy \
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

BASE_URL="http://127.0.0.1:8000"
"$PY" "$WORKERS/issue26_naive_profiler_client.py" \
  "--base-url=$BASE_URL" \
  --model Qwen3-30B-A3B-Instruct-2507 \
  --row pf4096_dc1024 --prefill-tokens 4096 --decode-tokens 1024 \
  --requests 100 --warmups "${ISSUE26_WARMUPS:-10}" --qps 2 --seed 20260908 \
  --nsys-control-dir "$ISSUE26_NSYS_CONTROL_DIR" \
  --output "$RUN_ROOT/runtime/client.jsonl" \
  > "$RUN_ROOT/runtime/client.log" 2>&1

EXPECTED_ROWS=$(((${ISSUE26_WARMUPS:-10} + 1) * 100))
test "$(wc -l < "$RUN_ROOT/runtime/client.jsonl")" -eq "$EXPECTED_ROWS"
test -s "$ISSUE26_NSYS_CONTROL_DIR/start"
test -s "$ISSUE26_NSYS_CONTROL_DIR/stop"

cleanup
SERVER_PID=""
find "$RUN_ROOT/nsys" -maxdepth 1 -type f -printf '%f\n' | sort > "$RUN_ROOT/nsys_files.txt"
REP="$(find "$RUN_ROOT/nsys" -maxdepth 1 -type f -name '*.nsys-rep' -print -quit)"
test -n "$REP"
printf '%s\n' "$REP" > "$RUN_ROOT/nsys_report_path.txt"

for REPORT in cuda_gpu_trace cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_api_sum; do
  "$NSYS" stats --report "$REPORT" --format csv --output "$RUN_ROOT/nsys/stats" \
    --force-overwrite=true "$REP" > "$RUN_ROOT/nsys/stats_${REPORT}.log" 2>&1 || true
done

echo NSYS_CUDA_ONLY_PROFILER_EXECUTION_COMPLETE
