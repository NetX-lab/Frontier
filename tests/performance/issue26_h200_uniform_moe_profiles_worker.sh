#!/usr/bin/env bash
# Collect fresh uniform-routing MoE profiles for both required gating contexts.
set -euo pipefail
source "$(dirname "$0")/../e2e/issue26_h200_environment_probe.sh" "${1:?Provide a fresh output directory.}"
source /data/ycfeng/tmp/issue26-h200-network/company-proxy.sh
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" ALL_PROXY="$all_proxy"
export NO_PROXY="$no_proxy,127.0.0.1,localhost,::1" no_proxy="$no_proxy,127.0.0.1,localhost,::1"
export PATH="/usr/local/nvidia/bin:$PATH"
export PYTHONPATH="$REPO_ROOT:/data/ycfeng/tmp/vLLM-BS"
export WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_FRONTIER_INSTRUMENTATION=0
export TRITON_CACHE_DIR="$TMPDIR/$(basename "$(dirname "$PROBE_ROOT")")/triton-cache"
export CUDA_CACHE_PATH="$TMPDIR/$(basename "$(dirname "$PROBE_ROOT")")/cuda-cache"
export PROBE_ROOT REPO_ROOT
test "$(git -c safe.directory=/data/ycfeng/tmp/vLLM-BS -C /data/ycfeng/tmp/vLLM-BS rev-parse HEAD)" = 46f7b179fd3bf42b9616dc4670cba419afdb2085
TOKENS=(1 2 4 8 16 24 32 48 64 80 96 100 128)
for BASE in 4096 8192 12288; do
  for OFFSET in 0 16 32 48 64 80 96 112 128; do TOKENS+=("$((BASE + OFFSET))"); done
done
TOKENS+=(16384 24576 32768)
for CONTEXT in prefill_hot standalone_legacy; do
  set -x
  "$PY" -m frontier.profiling.moe.main --disable_ray --models qwen3-a3b-30b-moe \
    --device h200 --num_gpus 8 --precision BF16 --profile_method cuda_event --yes \
    --output_dir "$PROBE_ROOT/$CONTEXT" --num_tensor_parallel_workers 1 \
    --expert_parallel_sizes 8 --routing_runtime_path uniform_topk \
    --gating_runtime_context "$CONTEXT" --enable_load_imbalance \
    --load_distributions uniform skewed extremely_skewed --num_samples_per_distribution 3 \
    --max_tokens 32768 --num_tokens_list "${TOKENS[@]}" > "$PROBE_ROOT/$CONTEXT.log" 2>&1
  set +x
done
"$PY" - <<'PYCODE'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd

out = Path(os.environ["PROBE_ROOT"])
root = Path(os.environ["REPO_ROOT"])
case = root / "task_memory/task_2026-09-07_issue26_ttft_h200"
merged = case / "supplements" / out.parent.name / "moe.csv"
tool = Path("/data/ycfeng/frontier-calibration-old-20260831/current-default/frontier-calibration/op-supplement/scripts/profiling_dataset_tools.py")
assert len(tool.read_bytes()) == 99483
assert hashlib.sha256(tool.read_bytes()).hexdigest() == "8e23cf1e0c6bd274c4230d2d8310ebab953f032042b4a436565a7a8756416c8f"
sources = [out / context / "compute/h200/qwen3-a3b-30b-moe/moe.csv"
           for context in ("prefill_hot", "standalone_legacy")]
for source in sources:
    rows = pd.read_csv(source)
    assert len(rows) == 387 and rows.num_tokens.nunique() == 43
    assert set(rows.routing_runtime_path) == {"uniform_topk"}
    assert set(rows.gating_runtime_context) == {source.parents[3].name}
command = [sys.executable, str(tool), "merge", "--module", "moe",
    "--base-csv", str(sources[0]), "--supplement-csv", str(sources[1]),
    "--output-csv", str(merged), "--case-dir", str(case),
    "--key-columns", "num_tensor_parallel_workers", "expert_parallel_size", "num_tokens",
    "load_distribution", "seed", "gating_runtime_context", "routing_runtime_path",
    "profiling_precision", "measurement_type", "--coverage-basis", "Fresh uniform_topk in both predictor contexts",
    "--profiling-command", f"bash {root}/tests/performance/issue26_h200_uniform_moe_profiles_worker.sh {out}",
    "--test-name", "pf4096_dc1024", "--source-type", "measured",
    "--previous-analysis-artifact", str(case / "analysis/uniform-moe-profiles-01/preflight.json"),
    "--next-action", "rerun_analysis"]
(out / "merge-command.json").write_text(json.dumps(command, indent=2) + "\n")
subprocess.run(command, check=True)
assert len(pd.read_csv(merged)) == 774
PYCODE
echo UNIFORM_MOE_PROFILING_EXECUTION_COMPLETE
