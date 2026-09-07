#!/usr/bin/env bash
# Run Frontier using only the new H200 profiles and observed queue arrivals.
set -euo pipefail
source "$(dirname "$0")/issue26_h200_environment_probe.sh" "${1:?Provide a fresh output directory.}"
source /data/ycfeng/tmp/issue26-h200-network/company-proxy.sh
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" ALL_PROXY="$all_proxy"
export NO_PROXY="$no_proxy,127.0.0.1,localhost,::1" no_proxy="$no_proxy,127.0.0.1,localhost,::1"
export PYTHONPATH="$REPO_ROOT"
export WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
export PROBE_ROOT REPO_ROOT
export PATH="/usr/local/nvidia/bin:$PATH"
export VLLM_FRONTIER_INSTRUMENTATION=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_CACHE_PATH="$TMPDIR/$(basename "$PROBE_ROOT")/cuda-cache"
export TRITON_CACHE_DIR="$TMPDIR/$(basename "$PROBE_ROOT")/triton-cache"
SIM_PY=""
for ENV_NAME in vllm-bs-0.10.2 vidur_te; do
  CANDIDATE="/local/ycfeng/anaconda3/envs/$ENV_NAME/bin/python"
  CANDIDATE_LIB="/local/ycfeng/anaconda3/envs/$ENV_NAME/lib"
  if LD_LIBRARY_PATH="$CANDIDATE_LIB:/usr/local/nvidia/lib64:$CUDA_HOME/lib64" \
    "$CANDIDATE" "$REPO_ROOT/tests/e2e/issue26_simulator_runtime_check.py" \
    "$PROBE_ROOT/runtime-$ENV_NAME.json" > "$PROBE_ROOT/runtime-$ENV_NAME.log" 2>&1; then
    if [[ -z "$SIM_PY" ]]; then SIM_PY="$CANDIDATE"; fi
  fi
done
if [[ -z "$SIM_PY" ]]; then
  echo "No validated simulator Python; inspect both runtime audit reports." >&2
  exit 1
fi
printf '%s\n' "$SIM_PY" > "$PROBE_ROOT/simulator_python.txt"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$SIM_PY")")/lib:/usr/local/nvidia/lib64:$CUDA_HOME/lib64"
"$SIM_PY" - <<'PYCODE'
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(os.environ["REPO_ROOT"])
out = Path(os.environ["PROBE_ROOT"])
case = root / "task_memory/task_2026-09-07_issue26_ttft_h200"
settings = json.loads((case / "config/frontier_run_01.json").read_text())
moe_key = "random_forrest_execution_time_predictor_config_moe_input_file"
merged = case / "supplements/moe-context-01/moe.csv"
import pandas as pd
from frontier.moe_gating_runtime import filter_moe_gating_rows_by_runtime_context
moe_rows = pd.read_csv(merged)
for context in ("standalone_legacy", "prefill_hot"):
    assert len(filter_moe_gating_rows_by_runtime_context(moe_rows,
        requested_context=context, source_name=str(merged))) == 387
settings[moe_key] = str(merged)
for key, value in settings.items():
    if key.endswith(("input_file", "trace_file")):
        if not Path(value).is_file():
            raise FileNotFoundError(f"Fresh input unavailable: {key}={value}")
settings["metrics_config_run_id"] = out.parent.name
settings["metrics_config_output_dir"] = str(out / "metrics")
settings["metrics_config_cache_dir"] = str(out / "predictor-cache")
settings["collective_sim_cc_backend_config_cache_dir"] = str(out / "collective-cache")
settings["collective_sim_cc_backend_config_runner_out_dir"] = str(
    Path(os.environ["TMPDIR"]) / out.parent.name / "htsim")
command = [sys.executable, "-m", "frontier.main"]
for key, value in settings.items():
    if isinstance(value, bool):
        command.append(("--" if value else "--no-") + key)
    else:
        command.extend(["--" + key, str(value)])
(out / "command.json").write_text(json.dumps(command, indent=2) + "\n")
(out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
with (out / "frontier.log").open("w") as stream:
    result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
(out / "exit_code.txt").write_text(str(result.returncode) + "\n")
result.check_returncode()
PYCODE
echo FRESH_FRONTIER_EXECUTION_COMPLETE
