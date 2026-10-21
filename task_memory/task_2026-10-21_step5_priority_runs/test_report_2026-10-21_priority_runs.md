# D21-RUNS CPU verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded P0 code verification and retained-startup evidence. |
| 2026-10-21 | Added formal dependency and legacy/R2 regression results. |

## P0 source checks

Command (repository root):

```bash
env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp \
 FRONTIER_SIMULATOR_PYTHON=/data/ycfeng/envs/frontier-py310/bin/python \
 /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest \
 tests/unit/test_step5_canonical_native.py tests/unit/test_step5_priority_runs.py \
 tests/unit/test_step5_client_metrics.py tests/unit/test_step5_native_pd.py \
 -q -p no:cacheprovider
```

Expected: both DP2/DP4 actual CPU HTTP paths retain16 warmups and8 concurrent diagnostic completions, strict warmed idle; four registry resource payloads and original metric/provenance checks pass. Actual: **99 passed in66.92s — PASS**. These are CPU fixtures, not native GPU performance or qualification evidence. Initial command without `FRONTIER_SIMULATOR_PYTHON` produced85 passes/14 setup errors because asyncopd Python3.12 lacks Plotly. The existing simulator interpreter override selects the installed Frontier Python3.10 environment; no package or repository workaround was added.

## Actual retained startup disposition

Read-only input: `/data/ycfeng/tmp/step5_d18_native_20261021/qualification02/stage/evidence/replay/canonical_summary.json`. The original `REFERENCE_UNSTABLE` verdict is unchanged. It records116 first-start observations with missing running/waiting/KV gauges, then verified all-DP4 zero gauges after16 warmups and24 total diagnostic completions. Exact relevant native metric lines plus original summary SHA256 are retained in `tests/fixtures/step5_pd/startup_metrics_qualification02.json`.

Locked native source `/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm/v1/metrics/loggers.py:521-539,614-621` uses `multiprocess_mode="mostrecent"`; `:1096-1123` sets those gauges only when `scheduler_stats is not None`. `v1/engine/async_llm.py:886-895` records stats from engine outputs. Thus absent first-start gauges are not idle and must not become zero. New runs preserve one first-start snapshot as explicitly diagnostic; no admission derives from it. Every post-warmup, post-formal and inter-repeat idle check remains strict, including required fresh cumulative completions. The first formal/concurrent phase still requires a verified16-warmup idle boundary. The image-source client reader recognizes this explicit Step-only schema, while preserving old fully verified seven-boundary summaries and the GLM path.

Regression replays real snapshot pages over CPU HTTP: missing first-start gauges stay unverified, later real16/24 completion pages pass strict idle, and the missing startup page still fails when used as a required warmed boundary. No GPU retry is justified solely by this diagnostic; actual remaining native readiness is parent-owned.

## Formal dependency and regression checks

Same environment/command as above, with these exact test paths:

```text
tests/unit/test_step5_priority_runs.py
tests/unit/test_step5_client_metrics.py
tests/unit/test_canonical_client_metrics.py
tests/unit/test_step5_canonical_native.py
tests/unit/test_step5_native_pd.py
tests/unit/test_step5_native_pd_r2_stage.py
tests/unit/test_e2e_gate.py
```

Additional pytest option: `--tb=short`. Actual: **204 passed in86.23s — PASS**. Expected/observed: both hardware families use actual existing clean-run and client reading schemas; missing own readiness, missing32 accuracy, scalar/boolean assertion, wrong native hardware/GPU inventory/runtime/model, incomplete population, wrong frozen workload, modified Frontier truth and wrong resolved Frontier hardware/scale all block formal64. All four errors in all three repeats are independently checked; exact30% accepts a release prerequisite while final10% remains FAIL;30.01% fails release. Formal32 requires own native readiness but not32 accuracy. Basic qualification bypasses both formal-only dependencies. All positive schema fixtures are explicitly synthetic arithmetic oracles, never native evidence.

During initial equality-test construction, inverse floating-point endpoint arithmetic produced an error infinitesimally above.30;40 tests passed and that equality oracle failed. The test was corrected to binary-exact2.5/3.25-second endpoints, so the actual existing comparator returns exactly.30. No tolerance, scaling factor or comparator change was introduced.

`git diff --check` passed. First sealed P0 package verified independently with the host `python3 -B .../canonical_stage.py verify --stage-dir ...`; all28 original tool pins and config SHA256 reproduced. Original native/runtime evidence remains unchanged.
