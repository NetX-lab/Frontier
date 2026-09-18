## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Recorded dead trace-reader cleanup and schema-preservation decision. |

# P2 trace reader cleanup

Invariant: mixed physical layers use StageExecutionTime; no production code writes the historical private trace override/dense annotations. Scalar timing callers remain valid. Searches across `frontier` and `tests/unit` found only neutral annotation-isolation fixtures after removing dead readers. The pre-change suite is the 139-test S1 result.

Environment: dedicated `/data/ycfeng/tmp/quality-review-env/bin/python`, Python 3.12.3, no conda, CPU. Exact post-change command from feature worktree:

```bash
env PATH=/data/ycfeng/tmp/quality-review-env/bin:$PATH PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true MPLCONFIGDIR=/data/ycfeng/tmp/quality-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_metrics_stage_execution_time.py tests/unit/test_execution_time_metrics_ownership.py tests/unit/test_execution_time_op_times.py tests/unit/test_metrics_full_stage_scope.py tests/unit/test_stage_reporting_contract.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/quality-p2-readers
```

PASS: **69 passed in 6.57 s**; `/data/ycfeng/tmp/quality-p2-readers.log`. Assertions retain mixed-layer scope, tensor/family identity, owner-once accounting, source isolation and positive residual traces. Phase-level non-dummy artifact comparison remains required.

The proposed FFN metric loop was deliberately not applied: `FFN_FAMILY` contains `mlp_act`, but `OperationMetrics.MLP_ACTIVATION.value` is `mlp_activation`. No shared mapping exists. Keeping the three current output projections preserves the schema with less machinery than a new alias adapter. An unused stage family lookup was removed safely because the actual attention iterator performs required family resolution.
