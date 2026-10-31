#!/bin/bash
# Export each commit with git archive and run the two failing tests there.
REPO=/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028
OUT=/data/ycfeng/tmp/b28_lead/bisect
mkdir -p $OUT
for c in "$@"; do
  d=$OUT/$c
  if [ ! -d $d ]; then
    mkdir -p $d && (cd $REPO && git archive $c frontier tests task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10 | tar -x -C $d)
    ln -s $REPO/data $d/data
  fi
  (cd $d && ulimit -v 31457280 && PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OMP_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp/b28_lead/tmp FRONTIER_TMP_ROOT=/data/ycfeng/tmp/b28_lead/tmp PYTHONPATH=$d:$d/tests/unit:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages timeout 900 /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_v4_measured_collective_consumer.py::test_v4_wave_passes_widths_without_mutating_source_batches tests/unit/test_v4_replay_census.py::test_observer_snapshots_final_descriptor_and_records_end_once -q -p no:cacheprovider > $OUT/$c.log 2>&1)
  echo "$c rc=$? $(tail -1 $OUT/$c.log)"
done
