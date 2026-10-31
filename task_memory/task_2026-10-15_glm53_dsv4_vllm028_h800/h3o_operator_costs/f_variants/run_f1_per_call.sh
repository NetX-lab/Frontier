#!/usr/bin/env bash
# Usage: [REPO=<worktree>] run_f1_per_call.sh <name> [field=value ...]
# REPO selects the checkout whose Frontier code prices the rows (default: this
# worktree). astra_sim_analytical is always selected.
set -euo pipefail
REPO=${REPO:-/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028}
O=/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/f_variants/f1
name=$1; shift
mkdir -p "$O"
cd "$REPO"
export PYTHONPATH=$REPO WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
ulimit -v 31457280
timeout 600 /data/ycfeng/envs/frontier-py310/bin/python \
  task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/f_variants/f1_astra_per_call.py \
  "$O/$name" cc_backend_config_type=astra_sim_analytical "$@" > "$O/$name.log" 2>&1
/usr/bin/python3 - "$O/$name/summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
for k, v in s["ratio_by_curve"].items():
    print(f"{k:24s} n={v['n']:3d} min={v['min']:.3f} med={v['median']:.3f} max={v['max']:.3f}")
for k, v in s["graph_forward_us"].items():
    print(f"{k:18s} measured={v['measured']:9.1f} predicted={v['predicted']:9.1f} ratio={v['ratio']:.3f}")
PY
