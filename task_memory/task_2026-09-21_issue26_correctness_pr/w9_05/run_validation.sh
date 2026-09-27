#!/bin/bash
# W9-05 fix: B2-B8 on the 2ffb062 export (before) and the fixed export (after).
set -u
V=/data/ycfeng/tmp/issue26-correctness-pr/w9_05
TASK=/data/ycfeng/Frontier/.worktrees/issue26-correctness-pr/task_memory/task_2026-09-21_issue26_correctness_pr
PY=/data/ycfeng/envs/frontier-py310/bin/python
B=$V/trees/before
A=$V/trees/after
export WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp
cd $V
# B2/B3: C2 PP=1 policy matrix.
( PYTHONPATH=$B "$PY" c2_pp1_policy_matrix.py run $B before $V/c2 > $V/c2_before.log 2>&1
  PYTHONPATH=$A "$PY" c2_pp1_policy_matrix.py run $A after $V/c2 > $V/c2_after.log 2>&1
  "$PY" c2_pp1_policy_matrix.py compare $V/c2 $B $A > $V/c2_compare.log 2>&1
  echo "c2 compare exit $?" >> $V/c2_compare.log ) &
# B2 breadth: KV-pressure sweep on both trees.
for side in before after; do
  tree=$B; [ "$side" = after ] && tree=$A
  "$PY" $TASK/w9_05/kv_pressure_sweep.py $tree $V/kv_sweep/$side > $V/kv_sweep_$side.log 2>&1 &
done
# B4: deadlock sweep, three cluster schedulers, fixed tree.
for s in rr:RoundRobinClusterSchedulerConfig lor:LORClusterSchedulerConfig random:RandomClusterSchedulerConfig; do
  "$PY" deadlock_sweep.py $A ${s#*:} $V/sweep/fix_${s%%:*} > $V/sweep_fix_${s%%:*}.log 2>&1 &
done
wait
# B5: fidelity matrix.
(
  cd "$A"
  for side in before after; do
    tree=$B; [ "$side" = after ] && tree=$A
    PYTHONPATH="$A" "$PY" "$A/tests/e2e/refactor_fidelity/run_matrix.py" run \
      --repo-root "$tree" --label "$side" --output-root "$V/fidelity" \
      --python-bin "$PY" --jobs 12 --clean-cache > "$V/fidelity-$side.log" 2>&1 &
  done
  wait
  PYTHONPATH="$A" "$PY" "$A/tests/e2e/refactor_fidelity/run_matrix.py" compare \
    --output-root "$V/fidelity" --baseline-label before --candidate-label after > "$V/fidelity-compare.log" 2>&1
  echo "compare exit $?" >> "$V/fidelity-compare.log"
) &
# B6: stage-admission matrix groups G3b, G9, G10, on the git worktrees (provenance needs git).
bash $TASK/w9_05/run_matrix.sh &
# B7: suites on the fixed git worktree (git-provenance tests need a checkout),
# compared with the W9-04 JUnit at 2ffb062.
( mkdir -p $V/suites_git
  bash $TASK/w9_01_stage_admission_ordering/composition_run_suites.sh /data/ycfeng/Frontier/.worktrees/w9-05-after "$V/suites_git"
  for suite in unit integration; do
    "$PY" $TASK/w9_01_stage_admission_ordering/composition_compare_junit.py \
      /data/ycfeng/tmp/issue26-correctness-pr/w9_04/suites/$suite.xml $V/suites_git/$suite.xml $V/suites_git/${suite}_compare.json \
      > $V/suites_git/${suite}_compare.log 2>&1
  done ) &
# B8: 16 architecture examples.
( for side in before after; do
    tree=$B; [ "$side" = after ] && tree=$A
    bash $V/run_examples.sh $tree $V/examples/$side > $V/examples_$side.log 2>&1
  done
  "$PY" $TASK/w9_05/compare_examples_symlink.py > $V/examples_compare_symlink.log 2>&1 ) &
wait
echo done
