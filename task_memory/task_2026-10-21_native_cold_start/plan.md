# Native cold-start plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Created scoped plan and acceptance checks. |

Dependency graph: captured-response RED -> {first-boundary proof, manifest-bound readiness} -> replay GREEN -> successor-checker verification -> focused regressions -> archive/commit.

1. Reproduce first-boundary failure with the actual retained native response. Verify actual V4 outcomes and source-backed publication behavior.
2. Add a first-boundary-only proof without synthesizing gauges. Readiness binds the actual Popen process, endpoint and prelaunch manifest. Reject missing, wrong-lane, nonzero, existing-client and stale completion evidence.
3. Preserve later strict observations and a non-optional completion target after a failed initial observation. Keep failed warmups unverified while all phases continue.
4. Copy frozen checker into this task only if needed; change only native first-boundary admission and test retained evidence. Never change old artifact admission or the root checker.
5. Run existing focused native/client tests; retain RED/GREEN and immutable-input checks, write report, commit.

Acceptance: captured first-boundary proof passes only with all three approved conditions; six later boundaries need explicit all-lane zero gauges and phase-fresh counters. Old V4 remains unverified. Canonical inputs and root checker remain byte-identical. No native qualification claim.
