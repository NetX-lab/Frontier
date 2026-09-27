#!/bin/bash
# Submit one DP-placement ground-truth run, keep the launcher alive, mirror the
# worker log locally and write the command receipt into the run directory.
set -u
set +x
SP="$(cd "$(dirname "$0")" && pwd)"
: "${RUN_TAG:?}" "${ENGINE_INPUT:?}" "${TRACE_INPUT:?}" "${NUM_GPUS:?}" "${NUM_CPUS:?}" "${MEM_GB:?}"
CASE=/data/ycfeng/Frontier/.worktrees/issue26-correctness-pr/task_memory/task_2026-09-21_issue26_correctness_pr/calibration/dp_pp_case_001
LOCAL=/data/ycfeng/tmp/issue26-correctness-pr/calibration/dp_pp_case_001/rjob/$RUN_TAG
mkdir -p "$LOCAL"
SUBMIT_LOG="$LOCAL/submit.log"
WORKER_LOG="$LOCAL/worker.log"
: > "$SUBMIT_LOG"
START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)

source "$SP/../w6_env.sh" >/dev/null 2>&1
# The environment script turns on errexit and pipefail; this wrapper handles
# its own exit codes, and an empty grep while the job name is pending is normal.
set +e +o pipefail
export RUN_TAG ENGINE_INPUT TRACE_INPUT NUM_GPUS NUM_CPUS MEM_GB

"$STEPMIND_PYTHON" -u "$SP/submit_dp_pp_run.py" >> "$SUBMIT_LOG" 2>&1 &
SUBMIT_PID=$!
JOB=""
for _ in $(seq 1 90); do
  JOB=$(grep -ao 'exp-[0-9]*-[0-9]*-[0-9]*' "$SUBMIT_LOG" | head -1)
  [ -n "$JOB" ] && break
  kill -0 "$SUBMIT_PID" 2>/dev/null || break
  sleep 2
done
echo "JOB=${JOB:-unknown}"
POLL_PID=""
if [ -n "$JOB" ]; then
  STEPMIND_JOB="$JOB" WORKER_LOG_PATH="$WORKER_LOG" POLL_SECONDS=14400 \
    "$STEPMIND_PYTHON" -u "$SP/poll_replica_logs.py" > "$LOCAL/poll.log" 2>&1 &
  POLL_PID=$!
fi
wait "$SUBMIT_PID"
SUBMIT_STATUS=$?
[ -n "$POLL_PID" ] && wait "$POLL_PID"
END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)

python3 - "$CASE/runs/groundtruth_clean/$RUN_TAG/launcher_receipt.json" <<PY
import json, sys
json.dump({
  "command": "RUN_TAG=$RUN_TAG ENGINE_INPUT=$ENGINE_INPUT TRACE_INPUT=$TRACE_INPUT NUM_GPUS=$NUM_GPUS NUM_CPUS=$NUM_CPUS MEM_GB=$MEM_GB bash $SP/run_dp_pp_job.sh",
  "working_directory": "/data/ycfeng/Frontier (spawn_tasks source and code_mount_point)",
  "environment_declarations": ["STEPMIND_BACKEND=rjob", "BRAINPP_ACCESS_KEY/BRAINPP_SECRET_KEY from the personal credential files (values not recorded)", "company http proxy from deploy.i.shaipower.com/httpproxy", "PYTHONPATH=/data/ycfeng/steptron", "STEPMIND_PYTHON=/data/ycfeng/tmp/stepmind-env/bin/python"],
  "launcher_scripts": ["$SP/run_dp_pp_job.sh", "$SP/submit_dp_pp_run.py", "$SP/poll_replica_logs.py"],
  "rjob_name": "${JOB:-unknown}",
  "start_utc": "$START_UTC",
  "end_utc": "$END_UTC",
  "launcher_exit_code": $SUBMIT_STATUS,
  "artifacts": {"evidence": "$CASE/runs/groundtruth_clean/$RUN_TAG/", "archive": "/mnt/codesign-exp/ycfeng/frontier/dp_pp_calibration/$RUN_TAG/", "submit_log": "$SUBMIT_LOG", "worker_log": "$WORKER_LOG"},
}, open(sys.argv[1], "w"), indent=1)
PY
echo "SUBMIT_STATUS=$SUBMIT_STATUS"
tail -40 "$WORKER_LOG" 2>/dev/null || echo "no worker log captured"
exit "$SUBMIT_STATUS"
