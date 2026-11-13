# Same-source Python3.12 environment supplement101

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record final-source101PASS; preserve explicit environment boundary. |

TaskID: D29-E41-FINAL-REGRESSIONS. Source
e41abc073500d61f224eda4fca1123e74e40544d; submodule3dac53c544e546574cb6e8078cd14053d0916113.
Both clean before/after. Cwd: `/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-05`.

Python3.12.13 `/data/ycfeng/envs/asyncopd-py312/bin/python`, pytest9.0.2,
Torch2.9.1, aiohttp3.13.3. Existing environment only; no installation or ambient
PYTHONPATH addition. PYTHONPATH is exactly repo:repo/tests/unit.
PYTHONDONTWRITEBYTECODE=1; TMPDIR/FRONTIER_TMP_ROOT=/data/ycfeng/tmp;
OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS=1;
STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/d29_published_sources01/source/vllm;
VLLM_TARGET_DEVICE=cpu. Every source entry is mounted read-only at the original path.

Exact launcher:
```bash
python3 /data/ycfeng/tmp/d29-e41-final-regressions-20261023-6dah8l8s/run_suite.py supplement101
```

Executed payload (expanded mount command retained in evidence):
```bash
timeout 180s /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_canonical_g3.py tests/unit/test_step5_native_position_cyclic.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-e41-final-regressions-20261023-6dah8l8s/supplement101/basetemp --junitxml=/data/ycfeng/tmp/d29-e41-final-regressions-20261023-6dah8l8s/supplement101/results.xml
```

Expected:101PASS, no failures/errors/skips. Observed: **101PASS/0FAIL/0ERROR/0SKIP**,
raw exit0, JUnit8.598s, outer9.785s. PASS for this supplement only, not a new full
unit suite or native GPU/runtime qualification. Its exact12 HTTP IDs and16
native-position cases must be joined with the later full-unit result before
accepting environment reconciliation.

Raw artifacts: scratch `/supplement101/{execution.json,results.xml,command1.log}`;
retained copies under `evidence/supplement101/`. Results are not summed with any
other suite. Broader asyncopd dependencies (e.g. sklearn/plotly) are absent;
this environment is used only for the required two modules.
