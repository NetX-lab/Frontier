# D27 source and payload verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Record parent RED/GREEN,191 affected passes and independent review. |

Environment: primary step5-preview worktree at ebd5baf plus the three-file D27
diff; `/data/ycfeng/envs/frontier-py310/bin/python`3.10.6, CPU only. Existing
public payload resolver and real Step capacity are used; no native timing run.

## Commands and results

All commands use `timeout 60s env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1
/data/ycfeng/envs/frontier-py310/bin/python -m pytest` from step5-preview.

1. Before source fix: `tests/unit/test_step5_structural_pd.py
tests/unit/test_step5_native_memory.py -k 'component_bytes or window_clip or
native_prefix or sparse_side_storage' -q
--junitxml=/data/ycfeng/tmp/d27_swa_red01.xml`.
Expected aligned payload overcount; actual4 failed,14 passed,14 deselected in4.15s,
exit1. Exact failures: aligned4096/528 and native source6112/6128.
Log `/data/ycfeng/tmp/d27_swa_red01.log`.
2. Same command after fix, XML/log `d27_swa_green01`:18 passed,
14 deselected in4.06s, exit0.
3. `tests/unit/test_step5_structural_pd.py tests/unit/test_step5_native_memory.py
tests/unit/test_step5_cache_description.py tests/unit/test_step5_cache_scheduler.py
-q --junitxml=/data/ycfeng/tmp/d27_swa_affected01.xml`:
191 passed in49.59s, exit0. Log `/data/ycfeng/tmp/d27_swa_affected01.log`.

## Native payload comparison

| Source tokens | Native bytes/rank | Before | Absolute error before | Relative error before | After / error | Descriptors after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6112 |139592704|140440576|847872|0.6074%|139592704 /0|19780|
| 6128 |139887104|140734976|847872|0.6061%|139887104 /0|19826|

Both after relative errors are0. These compare byte geometry, not latency.
Conservative group caps remain33/33/33/full-history; physical capacity and
reserved blocks are unchanged. Short and unaligned structural cases preserve
their recorded modeled behavior; they are not native qualification evidence.

## Review and remaining work

Reviewer session `ses_e9e8b2f76ffeL18tOAXFSAdgoY`, TaskID
D27_STEP_SWA_LIVE_PAGES: No findings in Spec/Standards. Inspected exact
`/data/ycfeng/tmp/d27_swa_review.diff`, requirements/design, source, parent proof
and logs; no independent test execution. Review confirms same group/capacity
block size and live interval intersection; no other-family/allocation change.

Merged legacy77 and combined D26 lifecycle/new-fixture verification remain
pending. This source result does not qualify native state, timing or full CLI.
