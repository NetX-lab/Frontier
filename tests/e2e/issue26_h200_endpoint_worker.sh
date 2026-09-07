#!/usr/bin/env bash
# Check the approved clean endpoint recorder in the pinned H200 image.
set -euo pipefail
source "$(dirname "$0")/issue26_h200_environment_probe.sh" "${1:?Provide a fresh output directory.}"
"$PY" "$REPO_ROOT/tests/performance/issue26_prefill_endpoint_probe.py" "$PROBE_ROOT/clock-check"
