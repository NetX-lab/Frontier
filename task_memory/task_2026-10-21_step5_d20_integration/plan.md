# D20 integration plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Define the parent-owned missing transfer input and integrated execution. |

Dependency: `{compute/bundle, PD review, explicit transfer input} -> merged small measured execution -> affected checks/fidelity -> source acceptance`; qualified native/profile inputs separately precede formal comparison.

1. Integrate committed PD9625984 and compute foundation1ca09be in isolated integration worktree; preserve ongoing compute WIP.
2. Add an exact NIXL profile loader and config input; reuse existing coverage/time hooks. Validate model/cache geometry, TP/devices, prefix/tail, payload and readiness duration. No interpolation or synthetic default.
3. Merge finished compute/bundle/delegates and execute the actual measured main path with clearly labelled CPU contract inputs, then verify output counts, context/tail and metrics. Real input absence stays a formal blocker.
4. Run combined regressions, one77-case fidelity comparison, review, consolidate code/records and update mainline checkpoint.

GPU jobs already submitted continue independently. No additional diagnostic campaign is part of this implementation.
