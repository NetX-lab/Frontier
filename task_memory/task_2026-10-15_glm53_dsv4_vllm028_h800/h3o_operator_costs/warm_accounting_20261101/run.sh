#!/usr/bin/env bash
set -euo pipefail
ulimit -v 31457280
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp
OUT=task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/warm_accounting_20261101
PY=/data/ycfeng/envs/frontier-py310/bin/python
date -u +%FT%TZ > "$OUT/run_start_utc.txt"
timeout 60s "$PY" "$OUT/inspect_current.py" > "$OUT/current_stdout.txt" 2> "$OUT/current_stderr.txt"
timeout 60s "$PY" "$OUT/payload.py" > "$OUT/payload_stdout.json" 2> "$OUT/payload_stderr.txt"
timeout 60s "$PY" "$OUT/segments.py" p2_64 > "$OUT/p2_64_segments_stdout.txt" 2> "$OUT/p2_64_segments_stderr.txt"
timeout 60s "$PY" "$OUT/segments.py" p2_32 > "$OUT/p2_32_segments_stdout.txt" 2> "$OUT/p2_32_segments_stderr.txt"
timeout 60s "$PY" "$OUT/gap_verify.py" > "$OUT/gap_stdout.json" 2> "$OUT/gap_stderr.txt"
timeout 60s "$PY" "$OUT/validate.py" > "$OUT/validation_stdout.txt" 2> "$OUT/validation_stderr.txt"
date -u +%FT%TZ > "$OUT/run_end_utc.txt"
