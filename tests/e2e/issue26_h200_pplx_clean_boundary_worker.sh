#!/usr/bin/env bash
# Capture one first-formal PPLX CUDA boundary with only an independent event pair.
set -euo pipefail
RUN_ROOT="${1:?Provide a fresh output directory under the active task directory.}"
WORKERS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Keep the launch script in RUN_ROOT while placing probe artifacts in a fresh
# child directory.  The probe intentionally rejects an already-existing root.
PROBE_ROOT="$RUN_ROOT/preflight"
source "$WORKERS/issue26_h200_environment_probe.sh" "$PROBE_ROOT"
source /data/ycfeng/tmp/issue26-h200-network/company-proxy.sh
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" ALL_PROXY="$all_proxy"
export NO_PROXY="$no_proxy,127.0.0.1,localhost,::1" no_proxy="$no_proxy,127.0.0.1,localhost,::1"
export VLLM_V1_ALLOW_NO_CHUNKED_PREFILL=1 VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_FRONTIER_INSTRUMENTATION=0
export WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WARMUPS="${ISSUE26_WARMUPS:-10}"
if (( WARMUPS < 10 )); then
  echo "ISSUE26_WARMUPS must be at least 10" >&2
  exit 1
fi
export ISSUE26_WARMUPS="$WARMUPS"
export VLLM_ALL2ALL_BACKEND=pplx
export VLLM_MOE_UNIFORM_ROUTING=1
export VLLM_MOE_DP_CHUNK_SIZE="${ISSUE26_MOE_DP_CHUNK_SIZE:-4096}"
if (( VLLM_MOE_DP_CHUNK_SIZE < 4096 )); then
  echo "ISSUE26_MOE_DP_CHUNK_SIZE must be at least 4096" >&2
  exit 1
fi

SOURCE="${ISSUE26_DIAGNOSTIC_VLLM_SOURCE:-/data/ycfeng/tmp/issue26-vllm-pplx-boundary-20260913-wt}"
COMMIT="${ISSUE26_DIAGNOSTIC_VLLM_COMMIT:-448f2b65e7679ae7490114ad382b6ba79becb3c3}"

RUN="$RUN_ROOT/runtime"
mkdir "$RUN"
RUN_CACHE_ROOT="$TMPDIR/$(basename "$PROBE_ROOT")"
export VLLM_CACHE_ROOT="$RUN_CACHE_ROOT/vllm-cache"
export CUDA_CACHE_PATH="$RUN_CACHE_ROOT/cuda-cache"
export TRITON_CACHE_DIR="$RUN_CACHE_ROOT/triton-cache"

# Boundary-only mode deliberately disables every existing diagnostic logger.
unset VLLM_FRONTIER_BATCH_LOG_PATH VLLM_FRONTIER_CUDA_EVENT_OP_LOG_PATH
unset VLLM_FRONTIER_CUDA_EVENT_OP_SCOPES VLLM_FRONTIER_CUDA_EVENT_SCOPE_MODE
unset VLLM_FRONTIER_SCHED_LOG_PATH VLLM_FRONTIER_SCHED_DECISION_LOG_PATH
unset VLLM_FRONTIER_DP_ROUTE_LOG_PATH VLLM_FRONTIER_MOE_ROUTING_LOG_PATH
unset VLLM_FRONTIER_MOE_COMPLETION_LOG_PATH VLLM_FRONTIER_PP_BOUNDARY_LOG_PATH
unset VLLM_FRONTIER_RUNTIME_META_ENABLED VLLM_FRONTIER_PREFILL_ENDPOINT_LOG_PATH
export ISSUE26_CLEAN_BOUNDARY_LOG_PATH="$RUN/server.boundary.jsonl"
export ISSUE26_CLEAN_BOUNDARY_REQUEST_PREFIX="cmpl-pf4096_dc1024:0-0"
export ISSUE26_CLEAN_BOUNDARY_SOURCE_COMMIT="$COMMIT"
export PYTHONPATH="$SOURCE"
if [[ -n "${ISSUE26_OPTIONAL_PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$ISSUE26_OPTIONAL_PYTHONPATH:$PYTHONPATH"
fi
if [[ -n "${ISSUE26_EXTRA_LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="$ISSUE26_EXTRA_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
fi
if [[ -n "${ISSUE26_LD_PRELOAD:-}" ]]; then
  export LD_PRELOAD="$ISSUE26_LD_PRELOAD"
fi
export VLLM_FRONTIER_BATCH_BOUNDARY_SOURCE_COMMIT="$COMMIT"
git config --global --add safe.directory "$SOURCE"
git -C "$SOURCE" rev-parse HEAD > "$PROBE_ROOT/vllm_commit.txt"
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$SOURCE" status --porcelain)"
git -C "$SOURCE" diff > "$PROBE_ROOT/vllm_uncommitted.patch"

"$PY" - "$PROBE_ROOT/mode_manifest.json" <<'PYMETA'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "mode": "pplx_clean_boundary_only",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "warmup_rounds": int(os.environ["ISSUE26_WARMUPS"]),
    "formal_requests": 100,
    "all2all_backend": os.environ["VLLM_ALL2ALL_BACKEND"],
    "moe_dp_chunk_size": int(os.environ["VLLM_MOE_DP_CHUNK_SIZE"]),
    "source_commit": os.environ["VLLM_FRONTIER_BATCH_BOUNDARY_SOURCE_COMMIT"],
    "boundary_request_prefix": os.environ[
        "ISSUE26_CLEAN_BOUNDARY_REQUEST_PREFIX"],
    "instrumentation": {
        "batch_boundary": False,
        "independent_clean_boundary": True,
        "per_op": False,
        "scheduler": False,
        "routing": False,
        "full_diagnostic_outer_span": False,
    },
    "limits": (
        "One independent first-formal model-forward CUDA event envelope per selected DP lane; "
        "one post-end synchronization only to read elapsed_time; Frontier instrumentation is disabled."
    ),
}, indent=2) + "\n")
PYMETA

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID"
    wait "$SERVER_PID" || [[ "$?" -eq 143 ]]
  fi
}
trap cleanup EXIT

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
  --requests 100 --warmups "$WARMUPS" --qps 2 --seed 20260908 \
  --output "$RUN/client.jsonl" > "$RUN/client.log" 2>&1
cleanup
SERVER_PID=""
"$PY" "$REPO_ROOT/tests/e2e/issue26_pplx_boundary_analysis.py" \
  --run "$RUN" --output "$RUN/boundary_analysis.json" --warmups "$WARMUPS"
echo PPLX_BOUNDARY_EXECUTION_COMPLETE
