## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Verified bounded first-stage prediction attribution, recorded the serialization-precision failure and corrected rerun. |

# D019 predictor-query audit

Result: **PASS**. Eleven nonzero compute models were observed at the real first-stage prediction call sites. Nine returned measured exact rows; shuffling and grouped GEMM each invoked RF once and then reused383 runtime-cache results. The first-stage endpoint was65.79007690374297ms. This is an existing-P attribution check, not fresh model training, GPU timing or corrected E2E evidence.

## Execution

Working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`.

Environment: conda `dev-vidur-v03-hopper-e2e`, Python3.13.13, executable `/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python`. CPU only. Existing current-task predictor cache `/data/ycfeng/tmp/issue26-cpu-frontier-p5n9slvi/predictor-cache` was reused; `RandomForestRegressor.fit` was replaced within the diagnostic process by a fail-fast function so an unintended estimator fit would fail the audit. The production code was not modified.

```bash
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp FRONTIER_LOG_LEVEL=ERROR \
  /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python \
  /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/e2e/issue26_predictor_query_audit.py \
  --config /data/ycfeng/tmp/issue26-first-batch-op-frontier-01/runtime/settings.json \
  --output /data/ycfeng/tmp/issue26-predictor-query-d019-02 \
  > /data/ycfeng/tmp/issue26-predictor-query-d019-02.log 2>&1
```

The same command with `-01` output/log paths produced the initial execution; its raw files remain in `/data/ycfeng/tmp`.

## Criteria and observed evidence

The check detects unobserved or misclassified predictor branches, unexpected fitting, failure to stop at the first stage, wrong feature-row attribution, and discrepancies between actual returned P and the prior first-stage log.

1. Reach the first `BatchStageEndEvent` and stop before its handler; observe request0/batch0 and the recorded first-forward boundary. PASS: time0.06579007690374297s, exactly the prior unrounded endpoint.
2. Observe eleven unique nonzero compute models after excluding wrapper delegation/cache repeats. PASS:9 measured exact and2 RF models,2592 total query returns,14 unique query records.
3. For every exact model, the matching filtered CSV-row mean equals its returned value within1e-12ms. PASS after preserving15-digit JSON precision.
4. For the two RF models, observe `RandomForestRegressor` and actual estimator-return line with zero exact matching filtered rows. PASS; all383 later cache hits per model returned the same value.
5. Multiply each per-layer returned value once by48 and compare to the rounded first-stage log. PASS: maximum absolute difference0.000021333558ms (`moe_gating_linear`), below48×0.0000005ms of accumulated per-layer decimal rounding. Maximum relative difference0.00322532%. Detailed predicted, logged, absolute and relative values are in `analysis/d019-predictor-query/validation.json`.

Observed stdout:

```json
{"status":"PASS_BOUNDED_PREDICTOR_QUERY_AUDIT","unique_queries":14,"calls":2592,"endpoint":{"time_s":0.06579007690374297,"batch_id":0,"request_ids":[0]}}
```

Reproducible receipt validation:

```bash
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python - <<'PY'
import json
import math
from pathlib import Path
task = Path('task_memory/task_2026-09-07_issue26_ttft_h200')
p = json.loads((task / 'analysis/d019-predictor-query/query_receipt.json').read_text())
old = json.loads((task / 'analysis/first-batch-op-rca/frontier_summary.json').read_text())
selected = [q for q in p['queries'] if q['branch'] in ('measured_exact', 'estimator_predict')]
assert len(selected) == 11
assert sum(q['branch'] == 'measured_exact' for q in selected) == 9
assert sum(q['branch'] == 'estimator_predict' for q in selected) == 2
assert abs(p['stop_boundary']['time_s'] * 1000 - old['endpoint_ms']) < 1e-8
for q in selected:
    op = q['model'].removesuffix('__prefill_hot')
    assert abs(q['prediction_ms'] * 48 - old['op_totals_ms'][op]) < 0.000025
    rows = q['matching_filtered_rows']
    if q['branch'] == 'measured_exact':
        assert rows
        mean = sum(row[q['target_column']] for row in rows) / len(rows)
        assert math.isclose(mean, q['prediction_ms'], abs_tol=1e-12)
    else:
        assert not rows and q['estimator'] == 'RandomForestRegressor'
print('PASS: 11 model paths, exact-row means, first-stage endpoint, and rounded P values')
PY
```

## Failure and correction

The initial first-stage execution passed, but post-processing raised `AssertionError` when checking a matching-row mean against the full-precision returned P. The diagnostic used pandas `DataFrame.to_json`'s default10-digit decimal serialization, losing source-row precision. The reusable diagnostic now sets `double_precision=15`; the same bounded CPU case was rerun and all numerical receipt checks passed. Predictor behavior and profiling data were unchanged.

## Practical limits

Observed exact/RF branch selection does not quantify P−S or S−V. No CUDA-span repair or clean-TTFT acceptance is claimed. The old grouped-GEMM P describes its old profiling implementation; new activation/reduction-complete measurements must be labeled with their changed implementation. The detailed conclusions and CSV/feature mapping are maintained in `analysis/d019-predictor-query/report.md`.
