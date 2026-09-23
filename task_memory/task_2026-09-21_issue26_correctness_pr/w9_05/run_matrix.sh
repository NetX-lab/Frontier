#!/bin/bash
# W9-05 B6: stage-admission matrix G3b, G9, G10 on git worktrees (provenance needs git).
set -u
V=/data/ycfeng/tmp/issue26-correctness-pr/w9_05
PY=/data/ycfeng/envs/frontier-py310/bin/python
B=/data/ycfeng/Frontier/.worktrees/w9-05-before
A=/data/ycfeng/Frontier/.worktrees/w9-05-after
export WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp
for side in before after; do
  tree=$B; [ "$side" = after ] && tree=$A
  cd "$tree" && PYTHONPATH="$tree" "$PY" -m tests.e2e.stage_admission_matrix run --set "w905-$side" \
    --group G3b --group G9 --group G10 --jobs 16 > "$V/matrix-$side.log" 2>&1
  echo "run exit $?" >> "$V/matrix-$side.log"
done
mkdir -p $V/matrix
cd "$A" && PYTHONPATH="$A" "$PY" -m tests.e2e.stage_admission_matrix compare --before w905-before --after w905-after \
  --output "$V/matrix/compare_w905.json" > "$V/matrix-compare.log" 2>&1
echo "compare exit $?" >> "$V/matrix-compare.log"
