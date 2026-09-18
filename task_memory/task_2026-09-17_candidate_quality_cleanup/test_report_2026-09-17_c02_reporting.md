## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Verified reporting-demand gating and payload allocation counts. |

# C02 — focused PASS; final integration recorded separately

Environment: worktree `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`; Python 3.12.3 at `/data/ycfeng/tmp/quality-review-env/bin/python`, uv/no conda. Prefix: `PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.

```bash
/data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_prefill_reporting_demand.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/pr33-c02-red
/data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_prefill_reporting_demand.py tests/unit/test_pdaf_prefill_model_time.py tests/unit/test_metrics_full_stage_scope.py tests/unit/test_metrics_stage_execution_time.py tests/unit/test_prefill_ep_wave_materialization.py tests/unit/test_decode_ep_wave_materialization.py tests/unit/test_stage_reporting_contract.py tests/unit/test_execution_time_metrics_ownership.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/pr33-c02-green-v2
```

Inspected consumers: MetricsStore.on_replica_stage_schedule emits op traces independently of write_metrics; operation metrics and full/summary stage ledgers consume execution-time payloads when write_metrics is enabled. Utilization consumes BatchStage timing, so its callback remains invoked even without a payload. The read-only demand query derives exclusively from existing config flags and the existing ledger predicate. EP demand delegates to the same query, including summary-only consumption.

Criteria: all 64 combinations of write/trace/operation/ledger/summary/utilization flags select the right demand; completion events and BatchStage wall/model timing are identical with reporting enabled or disabled. The disabled final-sync path reduces predictor calls from **2 to 1**, complete-stage constructions from **1 to 0**, and metrics correction calls from **1 to 0**, while retaining one metrics callback. Existing physical-layer/lane/trace/ledger/utilization contracts must pass.

Evidence: red **65 FAIL**, 7.78 s (missing demand API and observed unconditional work). First green **128 PASS / 1 FAIL**, 7.34 s: the new test indexed cluster_type instead of payload; corrected args[5] to args[4] and materialized pytest parameter combinations to remove the iterator deprecation. Final **129 PASS**, 6.99 s. No production fallback or suppressed assertion.

Logs: `/data/ycfeng/tmp/pr33-c02-{red,green,green-v2}.log`. Counts characterize a deterministic two-layer completion fixture, not E2E speedup. Final reporting-enabled simulation artifacts and isolated timing are checked in the final review report.
