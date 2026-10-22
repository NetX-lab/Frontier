# D28 parent source verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Record scoped parent tests and independent source review. |

Worktree `/data/ycfeng/Frontier/.worktrees/step5-d24-stage`, code6e73330,
record HEAD65878aa. Python3.12.13, Torch2.9.1+cu128, CPU execution only.

```bash
timeout 60s env TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/step5_profiling_20261020/image_src /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_native_pressure_observation.py tests/unit/test_step5_native_pd_r2_stage.py tests/unit/test_step5_canonical_g3.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d28_parent01 --junitxml=/data/ycfeng/tmp/d28_parent01.xml
```

Expected all affected checks pass. Observed81 PASS in3.99s, exit0;
`/data/ycfeng/tmp/d28_parent01.log` and XML. Includes actual pinned native
allocate_slots body with injected pregrant demand/free, unchanged native None,
and D-only flag/golden stage checks. This is source/fault-injection evidence,
not a running native engine or GPU allocation denial.

Reviewer `ses_e9e8b2f76ffeL18tOAXFSAdgoY`, TaskID
D28-D24-PRESSURE-STAGE: No findings. Reviewed full four-file diff, records,
package/import/seal receipts and current requirements. Existing callback
isolation remains; original native function called once and native errors kept.
The reviewer did not rerun tests. Author711 PASS is separate and overlapping.

Capture03 is sealed pressure preparation only. Balanced policy, merged/final
source binding, live creator/mount/GPU/argv/pool and actual denial/READ/state/
ownership/terminal outcomes remain unverified. No GPU/platform operation.
