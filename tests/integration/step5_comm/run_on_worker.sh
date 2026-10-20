#!/usr/bin/env bash
# Keep the locked image's Python, torch, vLLM and NCCL unchanged.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export VLLM_ENABLE_CUDA_COMPATIBILITY=1
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd "$REPO_ROOT"
exec python3 tests/integration/step5_comm/worker.py --config "$1"
