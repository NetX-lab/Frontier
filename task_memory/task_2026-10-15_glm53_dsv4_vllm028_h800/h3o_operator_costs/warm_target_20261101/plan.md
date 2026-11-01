# WARM-READING plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Created the CPU-only D-WARM-TARGET evidence plan. |

Task ID: WARM-READING. Authority: task requirements.md lines 149-165 and the delegated request.

1. Identify clean-reference handshake events using server records, request-owner inventory, and clock anchors. Reproduce the lifecycle/wave formal opener sets using the identical method. Stop on ambiguous formal binding; never select requests by latency.
2. Add tests/analysis/e2e_warm_reading.py using unchanged gate arithmetic. Verify original and empty-exclusion values exactly.
3. Produce six warm readings and the complete all-request/warm metric CSV, preserving source files.
4. Inspect outputs, retain command receipts and hashes, document limitations, and return evidence to the primary agent. No GPU operations, new replays, commits, or edits outside the assigned script/new evidence directory.

Dependency: 1 -> 2 -> 3 -> 4. Native/source inspection and row-path discovery are independent.

Acceptance: unique formal event/request/pair binding; exact lifecycle/wave agreement; six original and empty-set reproduction checks PASS; 192 metric CSV rows (6 generations/cases x 4 groups x 4 metrics x 2 readings); source hashes unchanged; per-repeat paired exclusion and pooled numerator/denominator sums on both sides. No confidence intervals or warmup modeling.
