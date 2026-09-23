#!/usr/bin/env bash
# Step 9 P5: refactor fidelity matrix, before (d1a2a06) and after (bacdbb4), one harness revision.
set -u
H=/data/ycfeng/Frontier/.worktrees/p5-fidelity-after
OUT=/data/ycfeng/tmp/issue26-correctness-pr/step9_p5/fidelity
PY=/data/ycfeng/envs/frontier-py310/bin/python
export WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp PYTHONPATH="$H"
cd "$H"
for side in before after; do
  "$PY" "$H/tests/e2e/refactor_fidelity/run_matrix.py" run \
    --repo-root "/data/ycfeng/Frontier/.worktrees/p5-fidelity-$side" --label "$side" \
    --output-root "$OUT" --python-bin "$PY" --jobs 12 --clean-cache > "$OUT-$side.log" 2>&1 &
done
wait
"$PY" "$H/tests/e2e/refactor_fidelity/run_matrix.py" compare \
  --output-root "$OUT" --baseline-label before --candidate-label after > "$OUT-compare.log" 2>&1
echo "compare exit $?" >> "$OUT-compare.log"
