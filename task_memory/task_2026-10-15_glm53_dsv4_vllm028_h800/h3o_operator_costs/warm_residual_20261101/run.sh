#!/bin/bash
set -euo pipefail
ulimit -v 31457280
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp
PY=/data/ycfeng/envs/frontier-py310/bin/python
TASK=task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800
OUT=$TASK/h3o_operator_costs/warm_residual_20261101
date -u +%FT%TZ > "$OUT/start_utc.txt"
"$PY" --version > "$OUT/python_version.txt"
git rev-parse HEAD > "$OUT/frontier_head.txt"
git status --short > "$OUT/git_status.txt"
"$PY" "$TASK/h3o_operator_costs/a10/remainder/remainder.py" /data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-lifecycle-20261025-01/artifacts "$OUT/native32_remainder.json" > "$OUT/native32_remainder.txt"
"$PY" "$OUT/analyze.py" "$OUT" > "$OUT/summary_stdout.json"
"$PY" "$OUT/sweep.py" "$OUT" > "$OUT/sweep_stdout.json"
"$PY" "$OUT/validate.py" "$OUT" > "$OUT/validation.txt"
date -u +%FT%TZ > "$OUT/end_utc.txt"
