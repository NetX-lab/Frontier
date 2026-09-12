#!/usr/bin/env bash
# Verify that the approved H200 worker image exposes a usable Nsight Systems CLI.
set -euo pipefail

OUT_ROOT="${1:?Provide a fresh capability output directory.}"
mkdir -p "$OUT_ROOT"
exec > >(tee "$OUT_ROOT/capability.log") 2>&1

date -u +%Y-%m-%dT%H:%M:%SZ
hostname
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L
else
  /usr/local/nvidia/bin/nvidia-smi -L
fi

NSYS="$(command -v nsys || true)"
if [[ -z "$NSYS" ]]; then
  for candidate in \
    /usr/local/bin/nsys \
    /opt/nvidia/nsight-systems/*/target-linux-x64/nsys \
    /opt/nvidia/nsight-systems/*/bin/nsys; do
    if [[ -x "$candidate" ]]; then
      NSYS="$candidate"
      break
    fi
  done
fi
if [[ -z "$NSYS" ]]; then
  echo "NSYS_UNAVAILABLE" >&2
  exit 2
fi
printf 'nsys_path=%s\n' "$NSYS"
"$NSYS" --version
"$NSYS" profile --help >/dev/null
printf '{"status":"PASS","nsys_path":"%s"}\n' "$NSYS" > "$OUT_ROOT/capability.json"
echo NSYS_CAPABILITY_PASS
