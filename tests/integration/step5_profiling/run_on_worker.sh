#!/usr/bin/env bash
# Run one Frontier Python command on an H800 worker in the Step-5-Preview image.
#
# Usage: run_on_worker.sh <python arguments...>
# Example: run_on_worker.sh -m frontier.profiling.moe.main --models stepfun-ai/Step-5-Preview ...
#
# The image's CUDA 13 user-space needs the forward-compatibility driver on the
# H800 hosts; VLLM_ENABLE_CUDA_COMPATIBILITY must be set before Python starts.
# Python packages Frontier needs but the image lacks are installed from the
# internal mirror into a worker-local directory, without dependencies, so the
# image's torch, vLLM, Triton and numpy stay untouched.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXTRA_SITE="${TMPDIR:-/tmp}/frontier_profiling_site"

export VLLM_ENABLE_CUDA_COMPATIBILITY=1
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
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
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "joblib": "joblib",
    "threadpoolctl": "threadpoolctl",
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
python3 -c 'import torch; assert torch.cuda.is_available(); print("WORKER_DEVICE", torch.cuda.get_device_name(0), torch.__version__)'
exec python3 "$@"
