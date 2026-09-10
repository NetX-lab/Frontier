#!/usr/bin/env bash
# Collect operator and routing evidence separately for the frozen H200 case.
set -euo pipefail
DIAGNOSTIC_SELECTION="${2:-operators_and_routing}"
case "$DIAGNOSTIC_SELECTION" in
  operators_and_routing) MODES=(operators routing) ;;
  batch) MODES=(batch) ;;
  rca) MODES=(batch operators kernels) ;;
  communication) MODES=(operators) ;;
  compute_attention|compute_moe|compute_detail) MODES=(operators) ;;
  *) echo "Unsupported diagnostic selection: $DIAGNOSTIC_SELECTION" >&2; exit 1 ;;
esac
source "$(dirname "$0")/issue26_h200_environment_probe.sh" "${1:?Provide a fresh output directory.}"
source /data/ycfeng/tmp/issue26-h200-network/company-proxy.sh
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" ALL_PROXY="$all_proxy"
export NO_PROXY="$no_proxy,127.0.0.1,localhost,::1" no_proxy="$no_proxy,127.0.0.1,localhost,::1"
export VLLM_V1_ALLOW_NO_CHUNKED_PREFILL=1 VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_FRONTIER_INSTRUMENTATION=1 VLLM_FRONTIER_TRACE_SKIP_WARMUP=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT="$TMPDIR/$(basename "$PROBE_ROOT")/vllm-cache"
export CUDA_CACHE_PATH="$TMPDIR/$(basename "$PROBE_ROOT")/cuda-cache"
export TRITON_CACHE_DIR="$TMPDIR/$(basename "$PROBE_ROOT")/triton-cache"
unset VLLM_FRONTIER_BATCH_LOG_PATH VLLM_FRONTIER_CUDA_EVENT_OP_LOG_PATH
unset VLLM_FRONTIER_SCHED_LOG_PATH VLLM_FRONTIER_MOE_ROUTING_LOG_PATH
unset VLLM_FRONTIER_RUNTIME_META_ENABLED VLLM_FRONTIER_PREFILL_ENDPOINT_LOG_PATH
unset VLLM_FRONTIER_SCHED_DECISION_LOG_PATH VLLM_FRONTIER_DP_ROUTE_LOG_PATH
export VLLM_ALL2ALL_BACKEND=naive
unset VLLM_FRONTIER_PROFILE_REQUEST_PREFIX VLLM_FRONTIER_PROFILE_BATCH_LIMIT
unset VLLM_FRONTIER_CUDA_EVENT_OP_SCOPES
if [[ "$DIAGNOSTIC_SELECTION" == rca || "$DIAGNOSTIC_SELECTION" == communication || "$DIAGNOSTIC_SELECTION" == compute_* ]]; then
  export VLLM_MOE_UNIFORM_ROUTING=1
  export VLLM_FRONTIER_PROFILE_REQUEST_PREFIX=cmpl-pf4096_dc1024:
  export VLLM_FRONTIER_PROFILE_BATCH_LIMIT=3
fi
case "$DIAGNOSTIC_SELECTION" in
  compute_attention)
    export VLLM_FRONTIER_CUDA_EVENT_OP_SCOPES=attn_pre_proj,attn_rope,attn_kv_cache_save,attn_prefill,row_parallel_gemm ;;
  compute_moe)
    export VLLM_FRONTIER_CUDA_EVENT_OP_SCOPES=moe_gating,moe_shuffling,moe_grouped_gemm,moe_sum ;;
  compute_detail)
    export VLLM_FRONTIER_CUDA_EVENT_OP_SCOPES=input_layernorm,post_attention_layernorm,embedding_compute,final_layernorm,attn_output_init,moe_grouped_gemm_w1,moe_activation,moe_grouped_gemm_w2 ;;
esac
if [[ "$DIAGNOSTIC_SELECTION" == compute_* ]]; then
  export VLLM_FRONTIER_PROFILE_BATCH_LIMIT=1
fi
if [[ "$DIAGNOSTIC_SELECTION" == communication ]]; then
  export VLLM_FRONTIER_CUDA_EVENT_OP_SCOPES=expert_parallel_allreduce,attn_post_proj_tp_allreduce,tensor_parallel_allreduce
fi
SOURCE="${ISSUE26_DIAGNOSTIC_VLLM_SOURCE:-/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908}"
COMMIT="${ISSUE26_DIAGNOSTIC_VLLM_COMMIT:-8453dd342c6aa2721aaf4b410998aab2f38bc2ec}"
export PYTHONPATH="$SOURCE"
git config --global --add safe.directory "$SOURCE"
git -C "$SOURCE" rev-parse HEAD > "$PROBE_ROOT/vllm_commit.txt"
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$SOURCE" status --porcelain)"
git -C "$SOURCE" diff > "$PROBE_ROOT/vllm_uncommitted.patch"
SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID"
    wait "$SERVER_PID" || [[ "$?" -eq 143 ]]
  fi
}
trap cleanup EXIT
for MODE in "${MODES[@]}"; do
  RUN="$PROBE_ROOT/$MODE"
  mkdir "$RUN"
  unset VLLM_FRONTIER_REQUEST_METRICS_LOG_PATH
  export VLLM_FRONTIER_BATCH_LOG_PATH="$RUN/server.batch.jsonl"
  unset VLLM_FRONTIER_DP_ROUTE_LOG_PATH VLLM_FRONTIER_SCHED_LOG_PATH
  unset VLLM_FRONTIER_SCHED_DECISION_LOG_PATH
  if [[ "$MODE" == operators || "$MODE" == kernels ]]; then
    export VLLM_FRONTIER_CUDA_EVENT_OP_LOG_PATH="$RUN/server.ops.jsonl"
    export VLLM_FRONTIER_OP_TIMING_MODE=cuda_event VLLM_FRONTIER_OP_AGG_MODE=per_scope
    export VLLM_FRONTIER_CUDA_EVENT_SCOPE_MODE=default VLLM_FRONTIER_RUNTIME_META_ENABLED=0
    if [[ "$MODE" == kernels ]]; then
      export VLLM_FRONTIER_OP_TIMING_MODE=record_function
    fi
    unset VLLM_FRONTIER_MOE_ROUTING_LOG_PATH
  elif [[ "$MODE" == routing ]]; then
    unset VLLM_FRONTIER_CUDA_EVENT_OP_LOG_PATH VLLM_FRONTIER_RUNTIME_META_ENABLED
    export VLLM_FRONTIER_MOE_ROUTING_LOG_PATH="$RUN/server.routing.jsonl"
  else
    unset VLLM_FRONTIER_CUDA_EVENT_OP_LOG_PATH VLLM_FRONTIER_RUNTIME_META_ENABLED
    unset VLLM_FRONTIER_MOE_ROUTING_LOG_PATH
    export VLLM_FRONTIER_SCHED_LOG_PATH="$RUN/server.scheduler.log"
    export VLLM_FRONTIER_SCHED_DECISION_LOG_PATH="$RUN/server.decisions.jsonl"
    if [[ "$DIAGNOSTIC_SELECTION" == rca ]]; then
      export VLLM_FRONTIER_DP_ROUTE_LOG_PATH="$RUN/server.dp_route.jsonl"
    fi
  fi
  "$PY" - "$RUN/mode_manifest.json" "$MODE" <<'PYMETA'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "mode": sys.argv[2], "created_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version, "warmup_rounds": 3, "formal_requests": 100,
    "environment": {key: value for key, value in os.environ.items()
                    if key.startswith("VLLM_") or key == "PYTHONPATH"},
    "limits": "Instrumented diagnostics; not clean TTFT evidence.",
}, indent=2) + "\n")
PYMETA
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
echo "DIAGNOSTIC_EXECUTION_COMPLETE selection=$DIAGNOSTIC_SELECTION"
