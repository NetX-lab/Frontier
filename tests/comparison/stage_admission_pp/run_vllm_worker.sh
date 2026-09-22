#!/usr/bin/env bash
# Worker entry point for the stage-admission vLLM ground truth (4 GPUs,
# vllm/vllm-openai:v0.10.2). Builds the instrumented overlay, runs the MoE and
# dense bursts, and publishes the traces to the cloud-volume archive and to the
# calibration case directory on the mounted workspace.
#
# Required environment:
#   RUN_TAG        identifier of this run
#   FRONTIER_TREE  Frontier worktree on the mounted workspace
#   GROUNDTRUTH    vLLM-BS checkout on the mounted workspace
#   CASE_DIR       calibration case directory on the mounted workspace; reads
#                  inputs/, writes runs/vllm-instrumented/<RUN_TAG>/
#   ARCHIVE_DIR    cloud-volume directory for this run
set -euo pipefail
set +x
: "${RUN_TAG:?}" "${FRONTIER_TREE:?}" "${GROUNDTRUTH:?}" "${CASE_DIR:?}" "${ARCHIVE_DIR:?}"
EVIDENCE_DIR="$CASE_DIR/runs/vllm-instrumented/$RUN_TAG"

for d in /usr/local/nvidia/lib64 /usr/local/nvidia/lib /usr/lib/x86_64-linux-gnu; do
  if [ -e "$d/libcuda.so.1" ]; then
    export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    break
  fi
done

PY=python3
SCRIPT_DIR="$FRONTIER_TREE/tests/comparison/stage_admission_pp"
WORK=/tmp/stage_admission_pp/$RUN_TAG
mkdir -p "$WORK/runs"
export VLLM_CACHE_ROOT="$WORK/vllm_cache" HF_HOME="$WORK/hf" HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1

publish() {
  local target="$1"
  mkdir -p "$target"
  for item in runs overlay_report.json worker_env.json vllm_import.txt; do
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
"$PY" "$SCRIPT_DIR/vllm_burst_driver.py" overlay \
  --site-vllm "$SITE_VLLM" --checkout "$GROUNDTRUTH" --destination "$WORK/overlay" \
  --expected-changes "$CASE_DIR/inputs/fork_changed_files.txt" \
  --report "$WORK/overlay_report.json" || status=3
if [ "$status" -ne 0 ]; then
  publish "$ARCHIVE_DIR"; publish_evidence
  echo "WORKER_STATUS=$status overlay rejected"
  exit "$status"
fi
export PYTHONPATH="$WORK/overlay"
"$PY" -c 'import vllm, vllm.v1.frontier_trace as t; print("VLLM_IMPORT", vllm.__version__, vllm.__file__, t.__file__)' \
  | tee "$WORK/vllm_import.txt"

run_scenario() {
  local name="$1"; shift
  local out="$WORK/runs/$name"
  mkdir -p "$out"
  # Engine cores write their placement records at interpreter exit, which a
  # forked multiprocessing child skips; spawned children run it.
  if VLLM_FRONTIER_INSTRUMENTATION=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
     VLLM_FRONTIER_PP_BOUNDARY_LOG_PATH="$out/pp_boundary.jsonl" \
     VLLM_FRONTIER_DP_PLACEMENT_LOG_DIR="$out/dp_placement" \
     timeout 1500 "$PY" "$SCRIPT_DIR/vllm_burst_driver.py" run --output-dir "$out" "$@" \
       > "$out/driver.log" 2>&1; then
    echo "SCENARIO_PASS $name"
  else
    echo "SCENARIO_FAIL $name exit=$?"
    tail -n 60 "$out/driver.log"
    status=1
  fi
}
run_scenario moe --model-config "$FRONTIER_TREE/data/config/models/Qwen3-30B-A3B-tiny.json" --enable-expert-parallel
run_scenario dense --model-config "$FRONTIER_TREE/data/config/models/Llama-3.2-1B-Instruct.json"

publish "$ARCHIVE_DIR"
publish_evidence

for name in moe dense; do
  grep -h "DRIVER_DONE" "$WORK/runs/$name/driver.log" | sed "s/^/$name /" || true
  wc -l "$WORK/runs/$name/pp_boundary.jsonl" 2>/dev/null || true
done
echo "WORKER_STATUS=$status RUN_TAG=$RUN_TAG"
exit "$status"
