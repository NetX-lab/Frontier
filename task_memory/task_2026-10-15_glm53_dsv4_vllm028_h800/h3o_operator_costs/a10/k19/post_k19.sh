#!/bin/bash
# Post-process one finished A10 K=19 replay: client gate, A8 census and the
# Frontier TTFT remainder. Usage: post_k19.sh <run name, e.g. p2_64_warm_k19b24>
set -euo pipefail
name=$1
case=${name:0:5}
S=/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4
REPO=/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028
A10=$REPO/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10
PY=/data/ycfeng/envs/frontier-py310/bin/python
OUT=$S/a10_gate/${name}_post
mkdir -p $OUT
export PYTHONPATH=$REPO:$A10:$REPO/tests/analysis PYTHONDONTWRITEBYTECODE=1
ulimit -v 31457280
cd $A10
timeout 900 $PY a10_gate.py $case $S/a10/$name $S/a10_gate/${name}_run > $OUT/gate_stdout.txt 2>&1
timeout 900 $PY a8_collectives.py $S/a10/$name > $OUT/a8_stdout.txt 2>&1
metrics=$(dirname $(ls $S/a10/$name/metrics/*/*/*/metrics_ground_truth.jsonl))
cd $A10/remainder
timeout 900 $PY frontier_remainder.py $metrics $OUT/frontier_remainder.json > $OUT/frontier_remainder.txt 2>&1
echo "post done $name"
