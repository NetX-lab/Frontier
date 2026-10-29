# D23-MRV2-FORWARD Plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Minimal core implementation and CPU acceptance plan. |
| 2026-10-22 | Add source-backed P2 correction with RED-before-code acceptance. |
| 2026-10-22 | Add minimal fixed-capacity follow-up with supported-state verification. |
| 2026-10-22 | Primary verified F1-F3; source integration and strict77 remain parent-owned. |

1. Implement existing-owner hold/resume, unified graph policy and truthful
   batch/workload binding. Acceptance: no first prediction before agreement;
   P/D useful and dummy physical widths/modes bound correctly.
2. Focused real-event lifecycle, duplicate/wakeup, aggregate width and shared
   fit/cache/count/lookup regressions. Acceptance: complete group release,
   zero dummy request credit, no missing-kind borrowing.
3. Run affected suites and baseline RED evidence; commit scoped clean source
   and hand off exact input requirements for parent integration.

Dependency: 1 -> 2 -> 3. No agent parallelism per delegation constraint;
independent CPU suites may run concurrently. Parent runs final strict77 matrix
after merging. Native qualification and accuracy are outside this deliverable.

## P2 correction

1. Verify native request-capacity mapping; extend the real engine idle-lane
   wakeup/stage first-call regression and capture exact RED failures.
2. Pass existing V1 scheduler capacity to agreement; resolve capacity-supported
   FULL versus unrestricted PIECEWISE candidates before DP min/max re-dispatch.
   Check both roles, asymmetric useful lanes, FULL<=cap, capture miss and legal
   prefill wider than the request cap. Preserve descriptors and ownership.
3. Run the affected stage and measured suites, record exact commands/results,
   self-check scope, commit the correction and hand off to parent.

Dependency: P2.1 -> P2.2 -> P2.3. Independent CPU suites can run concurrently.
Do not rerun producer/native/full CLI/strict77 or touch their fixtures.

## Fixed capacity follow-up

1. completed: Inspect native fixed capacity and mutable admission override reachability;
   verify existing Step thinking rejection without constructing unsupported state.
2. completed: Change only agreement's capacity source to existing scheduler config. Bind
   focused harnesses to actual V1 config and retain supported arrival-order and
   cap3 tests. No new behavior branch or policy/cost change.
3. completed: Run affected stage/engine and direct public predictor tests, inspect artifacts,
   update reachability/disposition records and commit for parent integration.

Dependency: F1 -> F2 -> F3. No new native/producer/full CLI/strict77 work.

F1-F3 closed in `ef476fe` and integrated by primary at `32fb934`; final independent
source review has no findings. Parent-owned strict77 passes77/77; full CLI remains active.
