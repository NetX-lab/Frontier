#!/bin/bash
# W9-04 fix: A4 fidelity matrix, A5 stage-admission G3b/G9/G10, A6 suites.
set -u
W=/data/ycfeng/tmp/issue26-correctness-pr/w9_04
PY=/data/ycfeng/envs/frontier-py310/bin/python
B=/data/ycfeng/Frontier/.worktrees/w9-04-before
A=/data/ycfeng/Frontier/.worktrees/w9-04-after
export WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp
(
  cd "$A"
  for side in before after; do
    tree=$B; [ "$side" = after ] && tree=$A
    PYTHONPATH="$A" "$PY" "$A/tests/e2e/refactor_fidelity/run_matrix.py" run \
      --repo-root "$tree" --label "$side" --output-root "$W/fidelity" \
      --python-bin "$PY" --jobs 12 --clean-cache > "$W/fidelity-$side.log" 2>&1 &
  done
  wait
  PYTHONPATH="$A" "$PY" "$A/tests/e2e/refactor_fidelity/run_matrix.py" compare \
    --output-root "$W/fidelity" --baseline-label before --candidate-label after > "$W/fidelity-compare.log" 2>&1
  echo "compare exit $?" >> "$W/fidelity-compare.log"
) &
(
  for side in before after; do
    tree=$B; [ "$side" = after ] && tree=$A
    cd "$tree" && PYTHONPATH="$tree" "$PY" -m tests.e2e.stage_admission_matrix run --set "w904-$side" \
      --group G3b --group G9 --group G10 --jobs 16 > "$W/matrix-$side.log" 2>&1
    echo "run exit $?" >> "$W/matrix-$side.log"
  done
  cd "$A" && PYTHONPATH="$A" "$PY" -m tests.e2e.stage_admission_matrix compare --before w904-before --after w904-after \
    --output "$W/matrix/compare_w904.json" > "$W/matrix-compare.log" 2>&1
  echo "compare exit $?" >> "$W/matrix-compare.log"
) &
bash /data/ycfeng/Frontier/.worktrees/issue26-correctness-pr/task_memory/task_2026-09-21_issue26_correctness_pr/w9_01_stage_admission_ordering/composition_run_suites.sh "$A" "$W/suites" &
wait
echo done
