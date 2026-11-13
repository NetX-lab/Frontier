# Frozen e41 full-unit parent04 and explicit environment comparison

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Preserve raw8015/279/32/281 and close exact-ID regression gate with101PASS supplement. |

TaskID D29-E41-FINAL-REGRESSIONS. Frozen source
e41abc073500d61f224eda4fca1123e74e40544d; submodule
3dac53c544e546574cb6e8078cd14053d0916113. Cwd
`/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-05`.
Clean before/after; source files read-only at original absolute paths.

Environment: frontier-py310 Python3.10.6, pytest9.1.1; numpy2.2.6,
pandas2.3.3, scipy1.15.3, sklearn1.7.2, plotly7.1.0, fasteners0.20,
ddsketch3.0.1. Torch and aiohttp absent. PYTHONPATH exactly repo:repo/tests/unit;
PYTHONDONTWRITEBYTECODE=1; TMPDIR/FRONTIER_TMP_ROOT=/data/ycfeng/tmp;
OMP/OPENBLAS/MKL threads1. No changed gates, exclusion or dependency installation.
Bubblewrap preserves source paths/read-only entries; generated cache/outputs stay
in unique scratch. Expanded mount command and full environment in execution.json.

```bash
python3 /data/ycfeng/tmp/d29-e41-final-regressions-20261023-6dah8l8s/run_suite.py unit
python3 /data/ycfeng/tmp/d29-e41-final-regressions-20261023-6dah8l8s/compare_unit.py
```

Executed pytest payload:
```bash
timeout 1800s /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit -q -p no:cacheprovider --continue-on-collection-errors --basetemp=/data/ycfeng/tmp/d29-e41-final-regressions-20261023-6dah8l8s/unit/basetemp --junitxml=/data/ycfeng/tmp/d29-e41-final-regressions-20261023-6dah8l8s/unit/results.xml
```

## Raw results and required expectation

Expectation: retain locked267 failure IDs/31 error IDs and no new code failures
after actual environment comparison; no baseline disappearance or PASS-to-SKIP.

Observed:8607 unique JUnit cases, **8015PASS/279FAIL/32ERROR/281SKIP**;
raw exit1, not timeout; JUnit1265.252s, outer1269.361s. Raw full-unit is **FAIL**,
not all-green. The required cross-environment regression gate is **PASS** below.
Suites are separate and never summed.

## Exact baseline and ID comparison

Actual raw baseline `/data/ycfeng/tmp/d30_unit_baseline01.xml`:
8093 cases,7617PASS/267FAIL/31ERROR/178SKIP, SHA256
`4c1629eaf42bde40b033f43e480f77aa5c81cba8ca4dca00398f43cc9eeda25d`.
Latest D30 `/data/ycfeng/tmp/d30_unit_candidate01.xml` plus65-ID
`/data/ycfeng/tmp/d30_canonical_count_green01.xml` overlay:
8124 cases,7647PASS/267FAIL/31ERROR/179SKIP. Both actual failure/error sets equal
`../repairs/evidence/d29_regression_baseline01.json`; no assertion by counts alone.

| Comparison | Observed |
| --- | --- |
| Locked failure IDs retained | 267/267;0 removed |
| Locked error IDs retained | 31/31;0 removed |
| Added failures | 12, each observed missing aiohttp |
| Added errors | 1, observed missing Torch during native-position collection |
| Baseline IDs missing (raw/latest) | 0/0 |
| Baseline PASS-to-SKIP (raw/latest) | 0/0 |
| Baseline PASS-to-FAIL/ERROR (raw/latest) | 0/0 |
| New test IDs (raw/latest) | 514/483; complete lists retained |
| Prior final03 ID additions/removals/status differences | 0/0/0 |
| metadata196/catalogue53 membership | Exact same IDs;all196/53 pass in full-unit |

Added12 exact failure IDs are all canonical G3 tests:
`test_actual_http_capability_covers_derived_and_legacy` for False/True crossed with
0.10.2,0.28.0,step5,step5_position_cyclic (8);
`test_actual_r2_driver_propagates_selected_engine_to_all_clients` for False/True
crossed with step5,step5_position_cyclic (4). Each traceback explicitly reports
`ModuleNotFoundError: No module named 'aiohttp'`.
Added error ID is `::tests.unit.test_step5_native_position_cyclic`; traceback
explicitly reports `No module named 'torch'`. Full exact IDs and tracebacks in
`evidence/unit_final04_comparison.json` and raw XML/log; none suppressed.

## Explicit same-source environment reconciliation

The independently executed same-source supplement uses asyncopd-py312,
Python3.12.13, pytest9.0.2, aiohttp3.13.3 and Torch2.9.1; exact repo-only imports
and explicit published native source root. It observes101PASS/0FAIL/ERROR/SKIP,
exit0. All12 identical HTTP IDs pass. All16 cases in the otherwise uncollectable
native-position module pass. Every added outcome is joined by exact ID or exact
module collection identity plus inspected exception, not simply by test name.
No new implementation failure remains after this explicit comparison.

Result: `NO_NEW_CODE_FAILURES_WITH_EXPLICIT_ENVIRONMENT_SUPPLEMENT`.
The raw Python3.10 suite remains exit1. Other pre-existing failures/errors/skips
are preserved, not fixed, excluded or claimed qualified. Broader native/GPU,
G3/E2E numerical acceptance and root main06 are separate.

## Evidence and final source checks

Raw: scratch `/unit/{execution.json,results.xml,command1.log}` and
`unit_final04_comparison.json`; retained identical copies under `evidence/`.
Each XML/locked ledger path, byte count and SHA256 is in the comparison receipt.
Final source check rehashes2611 tracked files against a during-run inventory:
0 differences; e41/sub3dac clean before/after. Inventory timing is explicit,
not claimed as a pre-launch full-file audit. Read-only mounts and per-suite
pre-launch clean HEAD checks provide the source-before guarantee.
