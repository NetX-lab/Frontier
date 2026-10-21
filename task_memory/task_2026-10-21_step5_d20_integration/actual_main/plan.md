# D20 actual main plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Define the actual-main run and parent-owned fix dependency. |
| 2026-10-21 | Apply parent-approved policy correction before complete actual CLI validation. |
| 2026-10-21 | Add the newly assigned request-free capability and reached fixture coverage before complete reruns. |

Dependency: fixture/config inspection -> real offline/online CLI -> approved Step graph-policy correction -> assigned request-free binding -> reached synthetic fixture coverage -> successful reruns -> artifact assertions -> completion commit. Focused tests run independently of the actual-main subprocess after the current code slice is ready.

1. Inspect task handoff, current main help, fixture and D19 argument guide. Keep current P16/D16 and DP placement switches explicit.
2. Add one reusable subprocess runner with synthetic READ input and no simulator mutation. Save exact commands, config, logs and return status.
3. Implement and test the explicitly approved Step policy correction using current scheduler semantics. Execute offline/online with the unchanged public policy. Return different-ownership source failures to parent, without workaround.
4. Implement and test the separately assigned request-free idle-DP measured identity and canonical workload binding. Reuse existing source scopes/typed cache; preserve positive projection/norm/FFN/MoE costs and typed missing-coverage errors. Parent owns rows.py; report the required has_attention_metadata column seam.
5. Provide explicit synthetic coverage for reached physical6145-token EP work, uniform expert-load cells and every decode context required by1024 outputs. Do not modify production bounds/exact-coverage checks. Verify one complete6144/1024 request in each mode, first D16/cached6128, one complete3.5ms READ with exact bytes, expected topology, release and final metrics.
6. Commit verified scoped production/tests/records; report formal/native limitations and exact different-ownership failures.

Original blocker: `full_decode_only` global policy is passed into the Step profile graph-mode map without normalization. Initial runs failed before event execution. This agent now owns the approved correction. The actual-main test is not accepted until step4 passes.

Current checkpoint: step3 is committed7e41b80. Parent has explicitly assigned step4, whose canonical binding and typed coverage tests pass41 cases. Step5 is executing against complete explicit synthetic fixtures; actual-main acceptance remains pending until outputs and release are inspected. The broader structural suite's obsolete pre-measured-mode assertion is recorded separately, not changed or hidden.
