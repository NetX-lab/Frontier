## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Verified full simulator dependency recovery in the pinned H200 image. |

# Simulator runtime recovery

Execution: H200 step_main RJob yc26-h200-fresh-frontier-20260908-02 on gpu-h200-0128; pinned image b97a2fdc624953679562a4835ea66e085ec22563f2086fecfe6710a9e682dbbc. Exact worker: /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/e2e/issue26_h200_frontier_worker.sh. Exact entry command: `bash tests/e2e/issue26_h200_frontier_worker.sh /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/runs/h200-fresh-frontier-02/runtime`. Runtime audits execute `tests/e2e/issue26_simulator_runtime_check.py OUTPUT_JSON` separately with each image interpreter and that environment's native library path.

Criteria: import all nine pyproject base dependencies; import frontier.simulator.Simulator; execute a deterministic RandomForestRegressor fit/predict; record package versions and module paths; select only a passing interpreter; consume current-task profiles with fresh predictor/collective caches.

Observed profiling environment vllm-bs-0.10.2 FAIL: ddsketch and plotly missing. This reproduces and expands the prior generation01 plotly import failure. Observed simulator environment vidur_te PASS: Python3.10.16, NumPy2.1.3, pandas2.2.2, plotly6.0.0, ddsketch3.0.1, fasteners0.19, sklearn1.6.1, scipy1.15.2, PyYAML6.0.2, tqdm4.67.1. All module paths belong to vidur_te; no package-tree overlay or download was used.

Actual simulator selected /local/ycfeng/anaconda3/envs/vidur_te/bin/python, read exactly100 fresh queue arrivals, initialized CollectiveSimCCBackend for h200/h200_dgx/world8 and began independent predictor training at18:44:23UTC. Full evidence: runs/h200-fresh-frontier-02/runtime/runtime-*.json, simulator_python.txt, settings.json, command.json and frontier.log.

Recovery verification PASS; committed bdd261abafa235cda48c852fe2a05b8eced4ff12. This establishes runtime import/native-library compatibility and successful entry into real training, not completed numerical simulation or TTFT parity. The exact-case result remains pending.
