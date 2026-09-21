## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Validated all100 fresh Frontier requests and computed the unadjusted official-server comparison. |

# Fresh H200 Frontier Baseline03

## Execution

Worktree: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`.
GPU job: `yc26-h200-fresh-frontier-20260908-03`, H200/step_main, same pinned image and exact4096/1024 case. Frontier69764e50,collective-sime564935. Existing worker: `tests/e2e/issue26_h200_frontier_worker.sh`; actual command/environment: `runs/h200-fresh-frontier-03/runtime/command.json`, `settings.json`, `simulator_python.txt`, and `runtime-vidur_te.json`. Simulation runtime Python3.10.16 in image conda `vidur_te`. All predictor/collective caches were new for this generation; compute CSV inputs belong to the current fresh H200 task.

Analyzer: `/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python`, Python3.13.13, conda `dev-vidur-v03-hopper-e2e`. Full script: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/e2e/issue26_frontier_baseline_analysis.py`.

Exact executed command from the worktree:

```bash
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_frontier_baseline_analysis.py --metrics task_memory/task_2026-09-07_issue26_ttft_h200/runs/h200-fresh-frontier-03/runtime/metrics/qwen3_a3b_30b_moe/online_serving/h200_fresh_frontier_03 --mapping task_memory/task_2026-09-07_issue26_ttft_h200/analysis/official-clean-01/request_id_map.csv --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/frontier-baseline-03
```

## Criteria and evidence

PASS:100unique formal requests, IDs0..99,100arrivals+100completions, exact4096/1024tokens, queue-arrival replay mapping, ordered endpoints, seconds/milliseconds consistency, agreement between JSON/CSV/system mean, and100/100completed requests. Analyzer also previously rejected incomplete generation02 before producing any mean.

| Metric | Value |
| --- | ---: |
| Frontier queue-to-prefill mean | 98.783926557ms |
| Official vLLM server TTFT mean | 115.982880592ms |
| Signed difference, Frontier minus vLLM | -17.198954035ms |
| Absolute error of means | 17.198954035ms |
| Relative error of means | 14.828872974% |

Raw unadjusted difference exceeds10%. D006 formal gate remains NOT_EVALUATED_UNADJUSTED_BASELINE because the additional critical-path work has not been independently measured/accounted and routing/operator semantics remain unresolved. The17.199ms gap is not established CPU overhead and must not be fitted as a constant correction.

## Artifacts and limits

`analysis/frontier-baseline-03/request_comparison.csv` retains all100per-request comparisons. `analysis/frontier-baseline-03/summary.json` retains aggregate evidence. This closes the first complete fresh baseline execution and comparison, not the full calibration/RCA/repair/rerun objective. D009 remains pending; no routing logging extension is applied. Source-proven MoE profiling/communication contract differences remain analysis candidates, not quantified TTFT causes.
