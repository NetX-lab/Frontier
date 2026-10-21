# D20 actual main plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Define the actual-main run and parent-owned fix dependency. |
| 2026-10-21 | Apply parent-approved policy correction before complete actual CLI validation. |

Dependency: fixture/config inspection -> real offline/online CLI -> approved Step graph-policy correction -> successful reruns -> artifact assertions -> completion commit.

1. Inspect task handoff, current main help, fixture and D19 argument guide. Keep current P16/D16 and DP placement switches explicit.
2. Add one reusable subprocess runner with synthetic READ input and no simulator mutation. Save exact commands, config, logs and return status.
3. Implement and test the explicitly approved Step policy correction using current scheduler semantics. Execute offline/online with the unchanged public policy. Return different-ownership source failures to parent, without workaround.
4. Verify one complete6144/1024 request in each mode, first D16/cached6128, one complete3.5ms READ with exact bytes, expected topology, release and final metrics. Commit completed test/records; report formal/native limitations.

Original blocker: `full_decode_only` global policy is passed into the Step profile graph-mode map without normalization. Initial runs failed before event execution. This agent now owns the approved correction. The actual-main test is not accepted until step4 passes.

Current checkpoint: step3's approved correction is verified by100 affected cases and one actual-CLI initialization regression. Step4 remains blocked by the separate missing measured request-free DP dummy-forward coverage. Parent must assign that producer/runtime capability before full reruns; this graph-policy slice adds no authority to invent a substitute cost.
