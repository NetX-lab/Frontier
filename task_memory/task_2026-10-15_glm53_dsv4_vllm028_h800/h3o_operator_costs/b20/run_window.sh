#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${B20_REPO:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}
B20_SCRATCH=${B20_SCRATCH:-/data/ycfeng/tmp/codex-implement/4bcc70f2-4440-43f7-9489-12ff832c399a/b20}
IDENTITY_ROOT=${1:-$B20_SCRATCH/candidate-final-cold}
WINDOW_ROOT=${2:-$B20_SCRATCH/window}
PYTHON_BIN=${PYTHON_BIN:-/data/ycfeng/envs/frontier-py310/bin/python}
"$PYTHON_BIN" - "$IDENTITY_ROOT" "$WINDOW_ROOT" <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
import sys
source, target = map(Path, sys.argv[1:])
receipt = json.loads((source / "receipt.json").read_text())
assert receipt["status"] == "FINISHED", receipt["status"]
target.mkdir(parents=True, exist_ok=True)
for name in ("cache", "cc_cache"):
    if (source / name).exists():
        shutil.copytree(source / name, target / name)
files = [{"name": str(p.relative_to(target / "cache")),
          "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
         for p in sorted((target / "cache").rglob("*")) if p.is_file()]
(target / "warm_cache_copy.json").write_text(json.dumps(dict(source=str(source), files=files), indent=2) + "\n")
PY
exec bash "$SCRIPT_DIR/run.sh" "$REPO" "$WINDOW_ROOT" --window 300
