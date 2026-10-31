#!/bin/bash
# F1a diagnostic A10 replay: astra_sim_analytical with H200 specification links
# (Switch, 3,600 / 400 Gbit/s, default 1 / 1 us latency) instead of the measured
# collective table, from the detached worktree f1a-astra-diagnostic at 9f5f8b61
# (main 3b7f44a5 with B29 delays + B30a d8493fd2). Ingress and handoff delays stay at
# the driver defaults, so the run differs from p2_*_warm_k19b29 only in the CC backend.
# usage: run_f1a.sh <case> <name>   (run root $S/f_variants/runs/<name>, warm cache copied from the B28 warm run)
REPO=/data/ycfeng/Frontier/.worktrees/f1a-astra-diagnostic
A10=$REPO/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10
S=/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4
P=astra_sim_analytical_cc_backend_config
case=$1; name=$2
R=$S/f_variants/runs/$name
mkdir -p $S/tmp $R
[ -d $R/cache ] || cp -a $S/a10/${case}_warm_k19b28/cache $R/cache
cd $REPO
ulimit -v 62914560
env PYTHONPATH=$REPO:$REPO/tests/analysis PYTHONDONTWRITEBYTECODE=1 TMPDIR=$S/tmp FRONTIER_TMP_ROOT=$S/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout 86400 /data/ycfeng/envs/frontier-py310/bin/python -u $A10/a10_driver.py $case $R \
    --field cc_backend_config_type=astra_sim_analytical \
    --field ${P}_intra_server_topology=Switch --field ${P}_intra_server_bandwidth_gbps=3600 \
    --field ${P}_inter_server_topology=Switch --field ${P}_inter_server_bandwidth_gbps=400 \
    > $R.log 2>&1
echo "EXIT $?" >> $R.log
