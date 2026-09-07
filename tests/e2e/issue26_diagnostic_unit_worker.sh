#!/usr/bin/env bash
# Validate diagnostic logger identity in the pinned H200 runtime.
set -euo pipefail
OUT="${1:?Provide a fresh output directory.}"
test ! -e "$OUT"
mkdir -p "$OUT"
exec > >(tee "$OUT/test.log") 2>&1
export TMPDIR=/data/ycfeng/tmp/issue26-d007-unit
mkdir -p "$TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
/usr/local/nvidia/bin/nvidia-smi --query-gpu=name,uuid --format=csv
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
export PYTHONPATH=/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908
"$PY" -c 'import sys, vllm; print(sys.version); print(vllm.__file__)'
"$PY" -m pytest "$PYTHONPATH/tests/core/test_frontier_op_logger.py" \
  --confcutdir="$PYTHONPATH/tests/core" --basetemp="$TMPDIR/pytest-$(basename "$OUT")" \
  -q -p no:cacheprovider
echo D007_LOGGER_TEST_PASS
