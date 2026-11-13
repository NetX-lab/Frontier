"""Capture the authorized evidence archive command and its actual outcome."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PRIMARY = Path("/data/ycfeng/Frontier/.worktrees/step5-preview")
DEST = PRIMARY / "task_memory/task_2026-10-22_step5_position_routing/ledger_memory/evidence/frozen_regressions_2026-10-25"
SCRIPT = ROOT / "archive_frozen_regressions.py"
argv = ["timeout", "60s", "sudo", "-n", "systemd-run", "--scope", "--quiet", "-p", "MemoryMax=30G", "runuser", "-u", "brainpp", "--", "env", "PYTHONDONTWRITEBYTECODE=1", "TMPDIR=/data/ycfeng/tmp", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1", "/data/ycfeng/envs/frontier-py310/bin/python", "-B", str(SCRIPT)]
start = datetime.now(timezone.utc).isoformat()
begin = time.perf_counter()
with (ROOT / "archive.log").open("wb") as log:
    result = subprocess.run(argv, cwd=PRIMARY, stdout=log, stderr=subprocess.STDOUT)
receipt = {"argv": argv, "cwd": str(PRIMARY), "start_utc": start, "end_utc": datetime.now(timezone.utc).isoformat(), "duration_s": time.perf_counter() - begin, "returncode": result.returncode, "script_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(), "configured_memory_max_bytes": 32212254720, "effective_cgroup_peak": "not measured"}
(ROOT / "archive_execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
if result.returncode == 0:
    for name in ("archive.log", "archive_execution.json"):
        source = ROOT / name
        target = DEST / name
        assert not target.exists()
        target.write_bytes(source.read_bytes())
        assert target.read_bytes() == source.read_bytes()
print(json.dumps(receipt, indent=2))
raise SystemExit(result.returncode)
