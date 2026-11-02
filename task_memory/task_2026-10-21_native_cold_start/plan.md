# Native cold-start plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Created scoped plan and acceptance checks. |
| 2026-10-21 | Added P1 full-checker readiness correction and complete predicate verification. |

Dependency graph: captured-response RED -> {first-boundary proof, manifest-bound readiness} -> replay GREEN -> successor-checker verification -> focused regressions -> archive/commit.

1. Reproduce first-boundary failure with the actual retained native response. Verify actual V4 outcomes and source-backed publication behavior.
2. Add a first-boundary-only proof without synthesizing gauges. Readiness binds the actual Popen process, endpoint and prelaunch manifest. Reject missing, wrong-lane, nonzero, existing-client and stale completion evidence.
3. Preserve later strict observations and a non-optional completion target after a failed initial observation. Keep failed warmups unverified while all phases continue.
4. Copy frozen checker into this task only if needed; change only native first-boundary admission and test retained evidence. Never change old artifact admission or the root checker.
5. Run existing focused native/client tests; retain RED/GREEN and immutable-input checks, write report, commit.
6. Primary P1 review -> full-checker `native_workers` RED -> reuse pinned metadata binding for that predicate, preserve strict legacy path -> all eight predicate expectations and every readiness-field negative -> rerun existing seams -> exact cumulative/narrow diffs, receipt and commit. Freeze/package remains unauthorized.

Acceptance: captured first-boundary proof passes only with all three approved conditions; six later boundaries need explicit all-lane zero gauges and phase-fresh counters. Old V4 remains unverified. Canonical inputs and root checker remain byte-identical. No native qualification claim.

P1 acceptance: the CPU full-checker fixture passes checkout, inputs, pinned readers, requests/idle, native workers, lifecycle and terminal receipts; `rank_runtime_graph` and overall status remain FAIL for missing native qualification. Missing or wrong values for all eight readiness fields reject the new proof. Legacy exact `{role,url}` verdict is unchanged, including extra-field rejection. No code outside successor/test changes in this correction.
