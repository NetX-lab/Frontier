"""Capture the final canonical comparison with the unchanged acceptance predicates."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
ROOT = Path(__file__).resolve().parent
argv = ["timeout", "60s", "sudo", "-n", "systemd-run", "--scope", "--quiet", "-p", "MemoryMax=30G", "runuser", "-u", "brainpp", "--", "env", "PYTHONDONTWRITEBYTECODE=1", "TMPDIR=/data/ycfeng/tmp", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1", "/data/ycfeng/envs/frontier-py310/bin/python", "-B", str(ROOT / "compare_final_unit.py")]
start = datetime.now(timezone.utc).isoformat()
begin = time.perf_counter()
with (ROOT / "unit_compare.log").open("xb") as stream:
    execution = subprocess.run(argv, cwd="/data/ycfeng/Frontier/.worktrees/step5-preview", stdout=stream, stderr=subprocess.STDOUT)
receipt = {"argv": argv, "cwd": "/data/ycfeng/Frontier/.worktrees/step5-preview", "start_utc": start, "end_utc": datetime.now(timezone.utc).isoformat(), "duration_s": time.perf_counter()-begin, "returncode": execution.returncode, "script_sha256": hashlib.sha256((ROOT / "compare_final_unit.py").read_bytes()).hexdigest(), "metadata_correction": "Baseline receipt names the identical package-version object python_versions; equality predicate remains exact.", "first_failure": {"returncode": 1, "log": str(ROOT / "unit_compare_first_failure.log"), "timing": "not independently captured"}}
(ROOT / "unit_compare_execution.json").write_text(json.dumps(receipt, indent=2)+"\n")
print(json.dumps(receipt, indent=2))
raise SystemExit(execution.returncode)
