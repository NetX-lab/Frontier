# Fresh79 conditional governance52 environment supplement

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record52PASS on immutable source79 using explicitly approved existing py312 CPU extras. |

TaskID D29-PERFORMANCE-FINAL-REGRESSIONS. Source79f1a3fd35d42b751cd1520c764ddac717b923d9,
tree8c6db078e852d3b885a6f14bdf5e8007d9ecddb7, submodule
3dac53c544e546574cb6e8078cd14053d0916113;source/submodule clean before/after.
Cwd `/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-06` with source
entries read-only at original absolute paths and isolated generated cache.

## Necessity and exact execution

Fresh full-unit established47PASS/5 retained missing-TorchFAIL in
`tests/unit/test_profiling_governance_minimal_red.py`;these five affected IDs do
not overlap required supplement101. All other affected component parent PASS IDs
already pass in full-unit. Actual traceback inspection justified one additional
original module run,not a new broad union or duplicate unrelated checks.
Diagnosis and initial comparison preserved in new scratch:
`governance_failure_diagnosis.json`, `unit_final05_before_governance.json`.

```bash
python3 /data/ycfeng/tmp/d29_performance_regressions_final05/run_suite.py governance52
```

Executed payload:
```bash
timeout 120s /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_profiling_governance_minimal_red.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_performance_regressions_final05/governance52/basetemp --junitxml=/data/ycfeng/tmp/d29_performance_regressions_final05/governance52/results.xml
```

Expanded argv/environment/packages in `evidence/governance52/execution.json`.
Explicit existing environment: Python3.12.13,pytest9.0.2,Torch2.9.1,
aiohttp3.13.3,numpy2.2.6,pandas3.0.1,scipy1.17.1;sklearn1.7.2,
plotly7.1.0,fasteners0.20,ddsketch3.0.1 come from existing approved
`/data/ycfeng/tmp/d29_py312_cpu_extras01/site`.
PYTHONPATH exactly repo:repo/tests/unit:that named extras directory;
VLLM_TARGET_DEVICE=cpu,PYTHONDONTWRITEBYTECODE=1,
TMPDIR/FRONTIER_TMP_ROOT=/data/ycfeng/tmp,OMP/OPENBLAS/MKL threads1.
Module origins were explicitly inspected before execution and recorded in
`evidence/governance_environment_preflight.json`. No dependency installation,
ambient root,source edit or gate weakening.

## Expected versus observed

Expected same52 parent IDs with all five unresolved affected cases passing.
Observed **52PASS/0FAIL/0ERROR/0SKIP**,raw exit0,JUnit18.688s,outer22.552s:
**PASS**. All52 IDs exactly equal retained parent52;every unresolved exact ID now
passes. No cases excluded or renamed. Swig deprecation warnings do not constitute
failures and remain in the raw log.

Fresh XML SHA256 `4e27ecd06d3fbbcb391c8b1c38fac203de9593cca4a40ca8d1694ccceaf36461`.
Raw XML/log:
`/data/ycfeng/tmp/d29_performance_regressions_final05/governance52/results.xml`,
`results.log`. Fresh exact-ID comparison after this run passes;this does not
rewrite the raw full-unit result or resolve its other historical failures/skips.
No aggregate total is computed across52/101/197/53/full-unit. Primary main08 and
native/GPU/G3/E2E qualification remain separate.
