# Post-prepared pricing diagnosis plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Complete measured attribution, supported lifecycle checks and evidence archive. |

Dependency graph: source binding -> real baseline -> strict pricing profile -> existing-mechanism inspection -> structural-only control + lifecycle checks -> records.

| Step | Status | Evidence |
| --- | --- | --- |
| Create isolated eece / 3dac worktree and relocate all config output/cache paths before parse | completed | `pricing-final01/rebound_wave.py`; source receipt |
| Run real layer60 / sixteen-EP pricing, both calibration overrides removed | completed | cold/warm timings, 208-event wave, captured comparison |
| Attribute warm parsing strictly within pricing | completed | 1520 pricing + 448 trace parses; verification excluded |
| Inspect registry identity preparation and marker/model replacement | completed | `_measurement_identity` existing cache; nine differential records |
| Assess one fresh-value structural control, no numerical reuse | completed | final sequential baseline/control; control is not faster |
| Archive exact commands, profiles, hashes, contracts and limitations | completed | report, provenance JSON and summary |

No implementation follows from this plan. No new identity cache is recommended on the measured evidence. Registry-view typed-contract resolution is a separately observed cost, not an established cache-safe causal repair.
