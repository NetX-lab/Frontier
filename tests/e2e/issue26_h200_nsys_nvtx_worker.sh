#!/usr/bin/env bash
# Capture a bounded first-formal CUDA window with an NVTX range trigger.
#
# This worker keeps Frontier instrumentation disabled and does not modify the
# vLLM checkout. A process-local sitecustomize shim replaces the existing
# Issue26NsightCapture CUDA profiler API calls with NVTX push/pop calls at the
# same model-forward boundary. The shim is created below under RUN_ROOT and
# is loaded only for this profiling process tree.
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

NVTX_SHIM_ROOT="$RUN_ROOT/nvtx-shim"
NVTX_SHIM_STATUS="$NVTX_SHIM_ROOT/status.json"
mkdir -p "$NVTX_SHIM_ROOT"
cat > "$NVTX_SHIM_ROOT/sitecustomize.py" <<'PY'
"""Process-local NVTX capture shim for Issue 26 profiling.

The production vLLM checkout is intentionally unchanged. The existing
``Issue26NsightCapture`` class still performs exact first-formal batch
matching; this subclass changes only the capture trigger and leaves all
matching and marker semantics intact.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


status_path = Path(os.environ["ISSUE26_NVTX_SHIM_STATUS"])


def _write_status(payload: dict[str, str]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, sort_keys=True) + "\n",
                           encoding="utf-8")


try:
    import vllm.v1.issue26_nsys_capture as capture_module

    class Issue26NvtxCapture(capture_module.Issue26NsightCapture):
        """Use NVTX range markers at the existing model-forward boundary."""

        def maybe_start(self, scheduler_output):  # type: ignore[no-untyped-def]
            if self._disabled or self._started or self._stopped:
                return False
            if not self._marker_exists("start"):
                return False
            if not self._matches_first_formal_batch(scheduler_output):
                return False
            try:
                self._torch.cuda.nvtx.range_push("issue26_formal_forward")
            except Exception as exc:  # pragma: no cover - GPU-only path
                self._disabled = True
                if self._strict:
                    raise RuntimeError(
                        "Issue 26 NVTX range_push failed") from exc
                print(f"Issue 26 NVTX range_push disabled: {exc}",
                      flush=True)
                return False
            self._started = True
            return True

        def maybe_stop(self):  # type: ignore[no-untyped-def]
            if self._disabled or not self._started or self._stopped:
                return False
            try:
                self._torch.cuda.nvtx.range_pop()
            except Exception as exc:  # pragma: no cover - GPU-only path
                self._disabled = True
                if self._strict:
                    raise RuntimeError(
                        "Issue 26 NVTX range_pop failed") from exc
                print(f"Issue 26 NVTX range_pop disabled: {exc}",
                      flush=True)
                return False
            self._stopped = True
            return True

    capture_module.Issue26NsightCapture = Issue26NvtxCapture
    _write_status({
        "status": "PASS",
        "capture_class": "Issue26NvtxCapture",
        "range": "issue26_formal_forward",
        "source_module": "vllm.v1.issue26_nsys_capture",
    })
except Exception as exc:  # pragma: no cover - startup failure path
    _write_status({"status": "FAIL", "error": repr(exc)})
    raise
PY

export ISSUE26_NVTX_SHIM_STATUS="$NVTX_SHIM_STATUS"
export ISSUE26_NSYS_CONTROL_DIR="$RUN_ROOT/nsys-control"
export ISSUE26_NSYS_CAPTURE_STRICT=1
export ISSUE26_NSYS_CAPTURE_MODE=nvtx
export VLLM_MOE_UNIFORM_ROUTING=1
export VLLM_V1_ALLOW_NO_CHUNKED_PREFILL=1 VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_ALL2ALL_BACKEND="$BACKEND"
export VLLM_FRONTIER_INSTRUMENTATION=0
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

# sitecustomize must precede vLLM and the client helper on PYTHONPATH.
export PYTHONPATH="$NVTX_SHIM_ROOT:$REPO_ROOT/tests/e2e:$SOURCE"
if [[ -n "${ISSUE26_OPTIONAL_PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$ISSUE26_OPTIONAL_PYTHONPATH:$PYTHONPATH"
fi
if [[ -n "${ISSUE26_EXTRA_LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="$ISSUE26_EXTRA_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
fi
if [[ -n "${ISSUE26_LD_PRELOAD:-}" ]]; then
  export LD_PRELOAD="$ISSUE26_LD_PRELOAD"
fi

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
  "profiler": "Nsight Systems CUDA/NVTX (NVTX-only range trigger)",
  "operator_instrumentation": false,
  "capture_control": "NVTX range_push before first formal model forward; range_pop immediately after that forward returns",
  "nvtx_capture_range": "issue26_formal_forward",
  "nvtx_shim": "run-root sitecustomize; vLLM checkout unchanged",
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
  --capture-range=nvtx \
  --nvtx-capture=issue26_formal_forward \
  --capture-range-end=stop \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --cuda-memory-usage=false \
  --cuda-event-trace=false \
  --cuda-trace-scope=process-tree \
  --trace-fork-before-exec=false \
  --show-output=false \
  --wait=all \
  --env-var="LD_LIBRARY_PATH=$TARGET_LD_LIBRARY_PATH,PYTHONPATH=$PYTHONPATH" \
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

"$PY" - "$NVTX_SHIM_STATUS" <<'PY'
import json
import sys
from pathlib import Path

status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if status.get("status") != "PASS":
    raise RuntimeError(f"NVTX shim failed to load: {status}")
if status.get("range") != "issue26_formal_forward":
    raise RuntimeError(f"unexpected NVTX range: {status}")
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

for REPORT in nvtx_sum cuda_gpu_trace cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_api_sum; do
  "$NSYS" stats --report "$REPORT" --filter-nvtx=issue26_formal_forward \
    --format csv --output "$RUN_ROOT/nsys/stats" --force-overwrite=true "$REP" \
    > "$RUN_ROOT/nsys/stats_${REPORT}.log" 2>&1 || true
done

echo NSYS_NVTX_PROFILER_EXECUTION_COMPLETE
