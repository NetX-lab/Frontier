# D26 parent source verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Record parent121 checks and independent source review. |

Worktree `/data/ycfeng/Frontier/.worktrees/step5-prefill-target`, codee8d9c6c,
d56f2dd,68ab45c. Python3.10.6; NumPy2.2.6, pandas2.3.3, sklearn1.7.2.
The following CPU check ran on final source; concurrent c86fca6 changes only
the author's existing report/progress with read-only P-wave findings.

```bash
timeout 240s env PYTHONPATH="$PWD:$PWD/tests/unit" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_prefill_target.py tests/integration/test_step5_measured_main.py tests/unit/test_step5_measured_pd.py tests/unit/test_nixl_read_cost.py tests/unit/test_step5_structural_pd.py tests/unit/test_glm53_structural_pd.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d26_parent01 --junitxml=/data/ycfeng/tmp/d26_parent01.xml
```

Expected all121 final affected cases pass; observed121 PASS in182.30s, exit0.
Log `/data/ycfeng/tmp/d26_parent01.log`; corresponding XML contains no
failure/error/skip. The real scheduler/event tests cover P target, client bound,
READ readiness/source lease release, both modes and complete output accounting;
fixture tests remain synthetic evidence, not native timing qualification.

Independent reviewer `ses_e9e8b2f76ffeL18tOAXFSAdgoY`, TaskID
D26-PREFILL-TARGET: No findings across13 supplied changed files. Checked
registered-family eligibility, shared Request semantics, semantic model bound,
dynamic exact prediction and v1 rejection/new v2 input identity. Reviewed
Request2049-line exception/split rationale. It did not execute tests.

Canonical target: oneP6128, oneD16 then1023 q1 forwards, semantic6144/1024.
Source logical pages383 and initial physical blocks1532 are allocation facts,
not wire-page counts. D27 changes live-wire payload only. New combined fixture,
merged strict77, full2400s CLI and native/GPU acceptance remain pending.
