# D20 actual main plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Define the actual-main run and parent-owned fix dependency. |

Dependency: fixture/config inspection -> real offline/online CLI -> parent production fix if required -> successful reruns -> artifact assertions -> completion commit.

1. Inspect task handoff, current main help, fixture and D19 argument guide. Keep current P16/D16 and DP placement switches explicit.
2. Add one reusable subprocess runner with synthetic READ input and no simulator mutation. Save exact commands, config, logs and return status.
3. Execute offline and online. On source failure, hand the exact source chain and evidence to parent; do not bypass the configured graph policy or touch source.
4. After parent correction, verify one complete6144/1024 request in each mode, first D16/cached6128, one complete3.5ms READ with exact bytes, expected topology and final metrics. Commit completed test/records; report formal/native limitations.

Current blocker: `full_decode_only` global policy is passed into the Step profile graph-mode map without normalization. Both modes fail before event execution. The actual-main test is not accepted until step4 passes.
