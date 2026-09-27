#!/bin/bash
# W9-04 fix: A2 sweep (three cluster schedulers) and A3 C2 (bacdbb4 vs 2ffb062).
W=/data/ycfeng/tmp/issue26-correctness-pr/w9_04
PY=/data/ycfeng/envs/frontier-py310/bin/python
export WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp
cd $W
for s in rr:RoundRobinClusterSchedulerConfig lor:LORClusterSchedulerConfig random:RandomClusterSchedulerConfig; do
  "$PY" deadlock_sweep.py $W/trees/after ${s#*:} $W/sweep/fix_${s%%:*} > $W/sweep_fix_${s%%:*}.log 2>&1 &
done
( PYTHONPATH=$W/trees/before "$PY" c2_pp1_policy_matrix.py run $W/trees/before before $W/c2 > $W/c2_before.log 2>&1
  PYTHONPATH=$W/trees/after "$PY" c2_pp1_policy_matrix.py run $W/trees/after after $W/c2 > $W/c2_after.log 2>&1
  "$PY" c2_pp1_policy_matrix.py compare $W/c2 $W/trees/before $W/trees/after > $W/c2_compare.log 2>&1
  echo "c2 compare exit $?" >> $W/c2_compare.log ) &
wait
echo done
