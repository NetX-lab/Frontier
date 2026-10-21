#!/usr/bin/env bash
# Keep the locked image's Python, torch, vLLM and NCCL unchanged.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# Reuse the qualified H200 launch recipe; R580+ must not load R570 compatibility.
major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | cut -d. -f1)
if [ "$major" = 570 ]; then
    driver_directory=/usr/local/cuda-13.0/compat
    export VLLM_ENABLE_CUDA_COMPATIBILITY=1
elif [ "$major" -ge 580 ]; then
    driver_directory=/usr/local/nvidia/lib64
    export VLLM_ENABLE_CUDA_COMPATIBILITY=0
else
    echo "Unsupported observed driver branch $major for the qualified launch recipe" >&2
    exit 1
fi
test -e "$driver_directory/libcuda.so.1"
export LD_LIBRARY_PATH="$driver_directory${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd "$REPO_ROOT"
exec python3 tests/integration/step5_comm/worker.py --config "$1"
