#!/usr/bin/env bash
# Inspect the exact H200 worker and image before collecting calibration data.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CASE_ROOT="$REPO_ROOT/task_memory/task_2026-09-07_issue26_ttft_h200"
PROBE_ROOT="${1:?Provide a fresh output directory under the active task directory.}"
case "$PROBE_ROOT" in
  "$CASE_ROOT"/*) ;;
  *) echo "Probe output must belong to $CASE_ROOT" >&2; exit 1 ;;
esac
if [[ -e "$PROBE_ROOT" ]]; then
  echo "Probe output already exists: $PROBE_ROOT" >&2
  exit 1
fi
mkdir -p "$PROBE_ROOT" /data/ycfeng/tmp/issue26-h200-runtime
export TMPDIR=/data/ycfeng/tmp/issue26-h200-runtime
export PYTHONDONTWRITEBYTECODE=1
exec > >(tee "$PROBE_ROOT/environment.log") 2>&1

date -u +%Y-%m-%dT%H:%M:%SZ
hostname
git -C "$REPO_ROOT" rev-parse HEAD
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv
nvidia-smi topo -m
nvidia-smi nvlink -s

PY=/local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python
COMPILED=/local/ycfeng/anaconda3/envs/vidur_te/lib/python3.10/site-packages/vllm
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$(dirname "$PY"):$PATH"
RUNTIME_ROOT="$(dirname "$(dirname "$PY")")"
export LIBRARY_PATH="$CUDA_HOME/lib64:$RUNTIME_ROOT/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:$CUDA_HOME/lib64:$RUNTIME_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD=/usr/local/nvidia/lib64/libcuda.so.1
export PYTHONPATH=/data/ycfeng/tmp/vLLM-BS
export VLLM_FRONTIER_COMPILED_PACKAGE="$COMPILED"
test -x "$PY"
test -f "$COMPILED/_C.abi3.so"
"$CUDA_HOME/bin/nvcc" --version
"$PY" - <<'PY'
import sys
import flashinfer
import torch
import vllm

print("python_executable", sys.executable)
print("python_version", sys.version)
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("flashinfer", flashinfer.__version__)
print("vllm_source", vllm.__file__)
assert torch.cuda.device_count() == 8, "Expected one eight-GPU serving pod"
for index in range(8):
    name = torch.cuda.get_device_name(index)
    assert "H200" in name, (index, name)
    with torch.cuda.device(index):
        tensor = torch.ones((32, 32), device="cuda")
        value = (tensor @ tensor)[0, 0].item()
        assert value == 32.0, (index, value)
    print("gpu_compute_pass", index, name, value)
print("H200_RUNTIME_PROBE_PASS")
PY
