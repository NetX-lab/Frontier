## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Verified default fit rejection, explicit fit opt-in and fresh corrected-GG first-stage integration; retained failed metadata admission and successful same-generation continuation. |

# D019 bounded predictor fitting and corrected-GG integration

Result: **PASS** for the scoped harness change and actual first-stage data integration. This does **not** close the CUDA-span correction gate: known operator/communication gaps can cancel in the total.

The harness now accepts `--allow-fit`; omission retains the original read-only estimator policy. The receipt records whether fitting is allowed and the number of successful RF fits observed in the current process. Permission to fit alone is not reported as proof of a fresh fit.

## Execution

Working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`.

Environment: conda `dev-vidur-v03-hopper-e2e`, Python3.13.13; executable `/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python`. All execution in this report is on the CPU master. No GPU case or simulation input length was changed.

Script: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/e2e/issue26_predictor_query_audit.py`.

Before the initial command, `/data/ycfeng/tmp/issue26-d019-moe-integration-01/predictor-cache` was confirmed absent (`test ! -e ...`, output `FRESH_PREDICTOR_CACHE_ABSENT`).

```bash
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp FRONTIER_LOG_LEVEL=WARNING \
  /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python \
  tests/e2e/issue26_predictor_query_audit.py --allow-fit \
  --config task_memory/task_2026-09-07_issue26_ttft_h200/analysis/d019-moe-integration/config.json \
  --output /data/ycfeng/tmp/issue26-d019-moe-integration-query-01 \
  > /data/ycfeng/tmp/issue26-d019-moe-integration-query-01.log 2>&1
```

This initial attempt trained fresh unchanged linear/attention models, then failed before MoE training because its input mixed `CUDA_EVENT` and `cuda_event` metadata. The dataset owner normalized that field using the existing `MeasurementType` parser and produced a separate v2 CSV/config, preserving the failed original. No timing value or production loader was changed by that correction.

```bash
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp FRONTIER_LOG_LEVEL=WARNING \
  /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python \
  tests/e2e/issue26_predictor_query_audit.py --allow-fit \
  --config task_memory/task_2026-09-07_issue26_ttft_h200/analysis/d019-moe-integration/config_v2.json \
  --output /data/ycfeng/tmp/issue26-d019-moe-integration-query-02 \
  > /data/ycfeng/tmp/issue26-d019-moe-integration-query-02.log 2>&1
```

The continuation reused the unchanged models freshly trained by the first attempt in this same cache generation. It completed six actual MoE RF fits in the main process. This is not a claim that subprocess CV fit calls are included in that counter.

The focused fit-policy check used the following direct CPU command:

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp \
  /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python - <<'PY'
import importlib.util
from sklearn.ensemble import RandomForestRegressor
spec = importlib.util.spec_from_file_location('query_audit', 'tests/e2e/issue26_predictor_query_audit.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
original = RandomForestRegressor.fit
try:
    policy = module.configure_fit_policy(False)
    try:
        RandomForestRegressor(n_estimators=1, random_state=0).fit([[0], [1]], [0, 1])
    except RuntimeError as exc:
        assert 'fitting is forbidden' in str(exc)
    else:
        raise AssertionError('Default policy allowed fitting')
    assert policy == {'allowed': False, 'successful_calls_in_process': 0}
    RandomForestRegressor.fit = original
    policy = module.configure_fit_policy(True)
    estimator = RandomForestRegressor(n_estimators=1, random_state=0).fit([[0], [1]], [0, 1])
    assert len(estimator.predict([[0], [1]])) == 2
    assert policy == {'allowed': True, 'successful_calls_in_process': 1}
    print('PASS: default rejection and actual opt-in RF fit')
finally:
    RandomForestRegressor.fit = original
PY
```

## Acceptance criteria and evidence

- Default fitting policy rejects a real `RandomForestRegressor.fit`; explicit opt-in completes a tiny two-row RF fit and records one successful call. **PASS**.
- Sparse per-target profiling rows must follow the existing loader filter. **PASS**: actual logs show `Dropping 774/783 rows ... before training moe_grouped_gemm`, `Dropping 9/783 rows ... moe_shuffling`, and `Dropping 9/396 rows` for each prefill-hot gating model.
- Actual first-forward GG query must use only the nine corrected4096 rows. **PASS**: measured-exact branch, physical CSV lines776–784, Pnew **0.4619448847240872ms/layer**. The old GG P was0.32574783652524153ms/layer.
- The other ten independent compute P values must remain unchanged. **PASS**: every returned value is numerically identical to the old query receipt, including RF shuffling.
- Stop at the same first-stage boundary and reconcile its change with only GG. **PASS**: actual **72.32753521728748ms**, expected **72.32753521728756ms**, absolute difference8.5265e-14ms, relative difference1.1789e-13%. Old boundary65.79007690374297ms; observed increase6.5374583135445ms.
- Preserve the4096 simulation input. **PASS**: original single-request trace, request0/batch0, same TP4/DP2/EP8, collective_sim/nvlink_analytic and CPU-overhead-off settings. M4095 remains a separate predictor-feature holdout owned by the dataset agent.

Successful output:

```json
{"status":"PASS_BOUNDED_PREDICTOR_QUERY_AUDIT","unique_queries":13,"calls":2592,"endpoint":{"time_s":0.07232753521728748,"batch_id":0,"request_ids":[0]}}
```

## Artifacts and limits

Under `analysis/d019-moe-integration/`:

- `predictor_query_receipt.json`: actual feature keys, returned branches/values, nine corrected source rows, config and observed fit count.
- `predictor_query_validation.json`: all eleven old/new P values, unchanged checks, exact-row ownership and endpoint reconciliation.
- `predictor_query_01_failure.log`: full original metadata-admission failure.
- `predictor_query_02.log`: successful continuation and per-target NaN filtering evidence.

Fresh GG estimator: `/data/ycfeng/tmp/issue26-d019-moe-integration-01/predictor-cache/moe_grouped_gemm_bac049d1.pkl`. The dataset agent receives this exact estimator for the separate M4095 feature holdout; this report does not claim that holdout result.

The nine training samples share the first-forward balanced feature key. Exact lookup matching these samples verifies correct data integration; it does not demonstrate independent generalization, runtime-context parity or full operator correction. The new72.3275ms total must not be accepted merely because an overall percentage falls below10%; the measured communication excess and remaining QKV/RoPE discrepancies require their own corrections and fresh evidence.
