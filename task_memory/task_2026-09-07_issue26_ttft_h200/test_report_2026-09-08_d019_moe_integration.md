## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded actual corrected-target training, first-stage query and heldout checks for dataset-builder commit0199432c. |

# D019 corrected expert dataset integration

Environment: CPU master, conda `dev-vidur-v03-hopper-e2e`, Python3.13.13. Workdir `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`. Set `PYTHONPATH=$PWD`, `PYTHONDONTWRITEBYTECODE=1`, `TMPDIR=/data/ycfeng/tmp`.

## Commands

```bash
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/performance/issue26_moe_profile_dataset.py \
  --base task_memory/task_2026-09-07_issue26_ttft_h200/supplements/moe-uniform-01/moe.csv \
  --corrected task_memory/task_2026-09-07_issue26_ttft_h200/analysis/d019-moe-normalized/moe.csv \
  --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/d019-moe-integration/moe_first_forward_v2.csv \
  --tokens 4096

FRONTIER_LOG_LEVEL=WARNING /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python \
  tests/e2e/issue26_predictor_query_audit.py --allow-fit \
  --config task_memory/task_2026-09-07_issue26_ttft_h200/analysis/d019-moe-integration/config_v2.json \
  --output /data/ycfeng/tmp/issue26-d019-moe-integration-query-02
```

Output directories already contain retained evidence; use fresh paths for another reproduction. The initial cache `/data/ycfeng/tmp/issue26-d019-moe-integration-01/predictor-cache` did not exist before query01. Retry02 continues only the unchanged models freshly trained during that same generation.

## Criteria and observed results

1. Obsolete expert timings absent: PASS, all774 original GG targets are NaN; only9 measured corrected M4096 rows train GG;4095 holdout and4097 reference excluded. Other operation populations remain unchanged, including both gating contexts.
2. Real loader/trainer admission: query01 FAIL on mixed `CUDA_EVENT`/`cuda_event` labels; canonicalized with existing MeasurementType parser, preserving raw input/log. Verified all non-measurement columns identical. Retry02 PASS; actual logs show `Dropping 774/783 rows ... moe_grouped_gemm`, `Dropping 9/783 ... moe_shuffling`, `Dropping 9/396 ... moe_gating_linear__prefill_hot` and routing counterpart.
3. Actual fresh training: PASS, query02 records6 successful RF fits, independent of flags alone.
4. First-stage numerical expectation: PASS, new GG exact=.4619448847240872ms from9 corrected rows, other10 compute P unchanged. Actual72.32753521728748ms equals old65.79007690374297ms plus48*(newGG−oldGG) to less than1e-14s. This is not a clean E2E metric.
5. Untouched4095 operator holdout: actual `_get_on_demand_prediction` queries against new `moe_grouped_gemm_bac049d1.pkl`, no fitting. EP0 P .461944885 vs S .445365330ms, abs .016579555ms/+3.723%; EP1 .461944885 vs .457087994ms, abs .004856891ms/+1.063%; EP7 .461809295 vs .473600000ms, abs .011790705ms/−2.490%. EP0/1 duplicate the training feature; EP7 actually uses RF at an untrained key. This is one nearby feature check, not full-case generalization.

Evidence: `analysis/d019-moe-integration/{first_forward_validation.json,holdout_query.json,moe_first_forward_v2.receipt.json,report.md}`; raw query02 receipt/log under `/data/ycfeng/tmp/issue26-d019-moe-integration-query-02`. Committed only `tests/performance/issue26_moe_profile_dataset.py` as0199432c. Full-case corrected coverage remains pending; a22-training/10-holdout shape proposal is recorded separately and not executed.
