#!/bin/bash
# =============================================================================
# Profiling Example - Metadata Smoke
# =============================================================================
# Validates that an existing profiling directory follows:
#   data/profiling/compute/<device>/<model>/...
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-rtx_pro_6000}"
MODEL="${MODEL:-qwen2_dense_test}"
DATA_DIR_BASE="${DATA_DIR_BASE:-$REPO_ROOT/data/profiling}"
PROFILE_DIR="${PROFILE_DIR:-}"

require_cli_value() {
  local option="$1"
  local value="${2-}"
  if [ -z "$value" ] || [[ "$value" == --* ]]; then
    echo "ERROR: $option requires a value" >&2
    exit 2
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --python-bin)
      require_cli_value "$1" "${2-}"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --device)
      require_cli_value "$1" "${2-}"
      DEVICE="$2"
      shift 2
      ;;
    --model)
      require_cli_value "$1" "${2-}"
      MODEL="$2"
      shift 2
      ;;
    --output-root|--data-dir-base)
      require_cli_value "$1" "${2-}"
      DATA_DIR_BASE="$2"
      shift 2
      ;;
    --data_path|--data-path|--profile-dir)
      require_cli_value "$1" "${2-}"
      PROFILE_DIR="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [ "$#" -gt 0 ]; then
  echo "ERROR: unexpected positional arguments: $*" >&2
  exit 2
fi

PROFILE_DIR="${PROFILE_DIR:-$DATA_DIR_BASE/compute/$DEVICE/$MODEL}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: PYTHON_BIN is not executable or not on PATH: $PYTHON_BIN" >&2
  exit 2
fi

if [ ! -d "$PROFILE_DIR" ]; then
  echo "ERROR: profiling directory does not exist: $PROFILE_DIR" >&2
  echo "Expected taxonomy: data/profiling/compute/<device>/<model>/..." >&2
  exit 2
fi

cat <<EOF
============================================
  Profiling Metadata Smoke
============================================
Taxonomy: data/profiling/compute/<device>/<model>/...
Profile dir: $PROFILE_DIR
============================================
EOF

cd "$REPO_ROOT"
"$PYTHON_BIN" -m frontier.utils.check_profiling_precision --data_path "$PROFILE_DIR"
