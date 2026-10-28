#!/bin/bash
# Post-run analysis of P1h package 03 (lead, CPU only).
set -u
W=/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028
V=/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify
D=/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/bridge
PKG=$W/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/p1h_bridge/package_03
P32=/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-profile-20261025-01/artifacts/torch_traces
P64=/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-64-profile-20261025-01/artifacts/torch_traces
PY=/data/ycfeng/envs/frontier-py310/bin/python
export PYTHONPATH=$W:$W/tests/unit:$W/tests/comparison/calibration:$W/tests/analysis
export PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
ulimit -v 31457280
cd $W
STEPS="${STEPS:-config groups planners bridge composition gaps identity}"
RAW="$D/FULL_AND_PIECEWISE/raw/dsv4_raw_e0.jsonl $D/FULL_AND_PIECEWISE/raw/dsv4_raw_e1.jsonl"
for step in $STEPS; do
  echo "== $step"
  case $step in
    config)
      $PY -c "import json,sys; d=json.load(open('$PKG/bridge_config.json')); d['output_root']='$D'; json.dump(d, open('$V/bridge_config.json','w'), indent=2)"
      echo "rc $?";;
    groups)
      timeout 600 $PY $V/realized_groups.py $V/bridge_config.json $RAW > $V/groups.log 2>&1; echo "rc $?"; tail -3 $V/groups.log;;
    planners)
      timeout 600 $PY $V/planner_layers.py $RAW > $V/planners.log 2>&1; echo "rc $?"; tail -20 $V/planners.log;;
    bridge)
      timeout 1200 $PY tests/comparison/calibration/dsv4_graph_bridge_check.py --config $V/bridge_config.json --report > $V/bridge.log 2>&1; echo "rc $?"; tail -3 $V/bridge.log;;
    composition)
      timeout 1800 $PY tests/comparison/calibration/dsv4_composition_check.py --config $V/bridge_config.json > $V/composition.log 2>&1; echo "rc $?"; tail -3 $V/composition.log;;
    gaps)
      timeout 1800 $PY -m frontier.profiling.kernel_gap.dsv4_replay_gaps --raw_files $RAW --output_dir $V/gap_output --device h200 --model DeepSeek-V4-Flash --diagnostics_path $V/gap_diagnostics.json > $V/gaps.log 2>&1; echo "rc $?"; tail -5 $V/gaps.log;;
    identity)
      for pair in "p32 $P32" "p64 $P64"; do
        set -- $pair
        timeout 1800 $PY tests/comparison/calibration/dsv4_kernel_identity_check.py --config $V/bridge_config.json --native_trace_dirs $2 --margin_cases e0-decode-b24-p6144 e0-dummy-full16 --output $V/identity_$1.json > $V/identity_$1.log 2>&1
        echo "identity $1 rc $?"; tail -2 $V/identity_$1.log
      done;;
  esac
done
