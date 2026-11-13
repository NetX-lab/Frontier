#!/usr/bin/env bash
# Run one Frontier Python command on an H800 worker in the Step-5-Preview image.
#
# Usage: run_on_worker.sh <python arguments...>
# Example: run_on_worker.sh -m frontier.profiling.moe.main --models stepfun-ai/Step-5-Preview ...
#
# The image's CUDA 13 user-space needs the forward-compatibility driver on the
# H800 hosts; VLLM_ENABLE_CUDA_COMPATIBILITY must be set before Python starts.
# Python packages the profilers need but the image lacks are installed from the
# internal mirror into a worker-local directory, without dependencies, so the
# image's torch, vLLM, Triton and numpy stay untouched. scipy and scikit-learn
# are deliberately absent: the profilers do not import them, and a --no-deps
# scikit-learn breaks transformers' optional sklearn import inside vLLM.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXTRA_SITE="${TMPDIR:-/tmp}/frontier_profiling_site"

driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
echo "WORKER_HOST_DRIVER $driver_version"
if (( ${driver_version%%.*} < 580 )); then
  export VLLM_ENABLE_CUDA_COMPATIBILITY=1
  export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
  export VLLM_ENABLE_CUDA_COMPATIBILITY=0
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
if [ -n "${FRONTIER_STEP5_PROFILE_MANIFEST-}" ]; then
  neutral="${TMPDIR:?bound profiling requires TMPDIR}/native_source_preflight"
  mkdir -p "$neutral"
  (
    cd "$neutral"
    python3 -I -B - "$REPO_ROOT" <<'PY'
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[1])
from frontier.profiling.common.runtime_identity import file_sha256, verify_step5_profile_runtime
manifest = Path(os.environ["FRONTIER_STEP5_PROFILE_MANIFEST"])
facts = verify_step5_profile_runtime(manifest)
if file_sha256(Path(os.environ["TMPDIR"]) / "model/config.json") != facts["shared"]["model_config_sha256"]:
    raise ValueError("worker effective model config copy differs from sealed source")
print("NATIVE_SOURCE_PREFLIGHT", json.dumps(facts, sort_keys=True))
PY
  )
fi
export PYTHONPATH="$REPO_ROOT:$EXTRA_SITE${PYTHONPATH:+:$PYTHONPATH}"

# Import name -> distribution name.
missing="$(python3 - <<'PY'
import importlib
packages = {
    "pandas": "pandas",
    "dateutil": "python-dateutil",
    "pytz": "pytz",
    "tzdata": "tzdata",
    "yaml": "pyyaml",
    "tqdm": "tqdm",
}
missing = []
for module, distribution in packages.items():
    try:
        importlib.import_module(module)
    except ImportError:
        missing.append(distribution)
print(" ".join(missing))
PY
)"
if [ -n "$missing" ]; then
  echo "WORKER_INSTALL $missing"
  mkdir -p "$EXTRA_SITE"
  # shellcheck disable=SC2086
  python3 -m pip install --no-cache-dir --quiet --root-user-action=ignore --no-deps \
    --target "$EXTRA_SITE" \
    --index-url http://mirrors.i.basemind.com/pypi/simple/ \
    --extra-index-url http://pypi.i.basemind.com/brain/dev/+simple \
    --trusted-host mirrors.i.basemind.com --trusted-host pypi.i.basemind.com \
    $missing
fi

cd "$REPO_ROOT"
python3 - <<'PY'
from pathlib import Path
import torch

assert torch.cuda.is_available()
print("WORKER_DEVICE", torch.cuda.get_device_name(0), torch.__version__)
print("WORKER_LIBCUDA", sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                               if "/libcuda.so" in line}))
PY
exec python3 "$@"
