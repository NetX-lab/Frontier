#!/usr/bin/env bash
# Worker entry point for the DP-placement vLLM ground truth
# (vllm/vllm-openai:v0.10.2). Builds the instrumented overlay, serves the case
# with `vllm serve`, replays its trace over HTTP, and publishes the evidence to
# the cloud-volume archive and to the calibration case directory on the
# mounted workspace.
#
# Required environment:
#   RUN_TAG        identifier of this run
#   MODE           ground-truth mode of vllm_replay.py: clean or instrumented
#   REPLAY_TIMEOUT_S  wall-time limit of the replay, below the job's cap so
#                  that a stopped replay is still published
#   FRONTIER_TREE  Frontier worktree on the mounted workspace
#   GROUNDTRUTH    vLLM-BS checkout on the mounted workspace
#   CASE_DIR       calibration case directory on the mounted workspace; reads
#                  inputs/, writes runs/groundtruth_<MODE>/<RUN_TAG>/
#   ENGINE_CONFIG  engine settings file under CASE_DIR/inputs
#   TRACE_DIR      trace directory (trace.csv, request_ids.json) under CASE_DIR/inputs
#   ARCHIVE_DIR    cloud-volume directory for this run
set -euo pipefail
set +x
: "${RUN_TAG:?}" "${MODE:?}" "${REPLAY_TIMEOUT_S:?}" "${FRONTIER_TREE:?}" "${GROUNDTRUTH:?}" \
  "${CASE_DIR:?}" "${ENGINE_CONFIG:?}" "${TRACE_DIR:?}" "${ARCHIVE_DIR:?}"
EVIDENCE_DIR="$CASE_DIR/runs/groundtruth_$MODE/$RUN_TAG"
# A run never writes over an earlier run's evidence.
for target in "$EVIDENCE_DIR/run" "$ARCHIVE_DIR"; do
  if [ -e "$target" ]; then
    echo "WORKER_STATUS=2 output already exists: $target"
    exit 2
  fi
done

for d in /usr/local/nvidia/lib64 /usr/local/nvidia/lib /usr/lib/x86_64-linux-gnu; do
  if [ -e "$d/libcuda.so.1" ]; then
    export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    break
  fi
done

PY=python3
SCRIPT_DIR="$FRONTIER_TREE/tests/comparison/dp_placement_pp"
WORK=/tmp/dp_placement_pp/$RUN_TAG
mkdir -p "$WORK/run"
export VLLM_CACHE_ROOT="$WORK/vllm_cache" HF_HOME="$WORK/hf" HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1

publish() {
  local target="$1"
  mkdir -p "$target"
  for item in run overlay_report.json worker_env.json vllm_import.txt; do
    if [ -e "$WORK/$item" ]; then cp -r "$WORK/$item" "$target/"; fi
  done
  echo "status=$status" > "$target/COMPLETE"
}
# The worker writes the mounted workspace as root; hand the evidence back to
# the owner of the case directory.
publish_evidence() {
  publish "$EVIDENCE_DIR"
  chown -R "$(stat -c %u:%g "$CASE_DIR")" "$EVIDENCE_DIR"
}
status=0
"$PY" - <<'PY' | tee "$WORK/worker_env.json"
import json, platform, torch
print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}))
PY

SITE_VLLM=$("$PY" -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))')
"$PY" "$FRONTIER_TREE/tests/comparison/stage_admission_pp/vllm_burst_driver.py" overlay \
  --site-vllm "$SITE_VLLM" --checkout "$GROUNDTRUTH" --destination "$WORK/overlay" \
  --expected-changes "$CASE_DIR/inputs/fork_changed_files.txt" \
  --patch "$CASE_DIR/inputs/groundtruth_overlay.patch" \
  --report "$WORK/overlay_report.json" || status=3
if [ "$status" -ne 0 ]; then
  publish "$ARCHIVE_DIR"; publish_evidence
  echo "WORKER_STATUS=$status overlay rejected"
  exit "$status"
fi
export PYTHONPATH="$WORK/overlay"
"$PY" -c 'import vllm, vllm.v1.frontier_trace as t; print("VLLM_IMPORT", vllm.__version__, vllm.__file__, t.__file__)' \
  | tee "$WORK/vllm_import.txt"

if timeout "$REPLAY_TIMEOUT_S" "$PY" "$SCRIPT_DIR/vllm_replay.py" \
     --engine-config "$ENGINE_CONFIG" --trace-dir "$TRACE_DIR" \
     --output-dir "$WORK/run" --mode "$MODE" > "$WORK/run/replay.log" 2>&1; then
  echo "REPLAY_PASS"
else
  status=$?
  echo "REPLAY_FAIL exit=$status"
  tail -n 80 "$WORK/run/server.log" 2>/dev/null || true
fi

publish "$ARCHIVE_DIR"
publish_evidence

grep -h "REPLAY_DONE" "$WORK/run/replay.log" || tail -n 30 "$WORK/run/replay.log" || true
wc -l "$WORK"/run/dp_placement/*.jsonl "$WORK"/run/*.jsonl 2>/dev/null || true
echo "WORKER_STATUS=$status RUN_TAG=$RUN_TAG"
exit "$status"
