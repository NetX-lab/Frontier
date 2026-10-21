# Native cold-start requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded delegated W128-NATIVE-COLD-START scope and approval. |

## Request and decisions

[Original Request] Task W128-NATIVE-COLD-START, worktree `native-cold-start`, base `e55c83c`. Implement D-NATIVE-COLD-START: only the first pre-warmup boundary may use newly dedicated services ready, zero client dispatches, and explicit native completion counters equal to zero on every DP lane. Subsequent six boundaries retain strict gauges, completion freshness and KV-release checks. Missing never means zero. Old V4 runs remain unverified.

[Original Request] Raw question prefix supplied by parent: `已确认 vLLM0.28首次调度前不发布gauges...`; approved answer: `批准冷启动证明（推荐）`. Parent retains the complete original question.

[Decision] Preserve canonical QPS1 workload at `68f891b`, 16 warmups and three repetitions, formulas, thresholds, auxiliary soft failures and native exceptions. Do not repair the repeat02 warmup HTTP502 (431/432 completions) in this slice.

[Decision] Frozen root `check_groundtruth_run.py` SHA256 `9116ab7c83da2b4e7cafb871c272af56c8322954eccd56abf2eeac1b55906b55` is read-only. If required, create a task-local successor from its exact bytes and provide a narrow diff and tests; parent reviews/freezes/repoints new generations. `native_full_structure.py` is another agent's read-only property.

[Decision] No GPU/platform/auth/SSH/cloud/external network, package installs, run launches, merge/push/delegation. At most five production/test files and 500 changed lines; Frontier core is untouched. Focused native/client regressions suffice.

[Approved test seams] Existing `idle_metrics` / `observe_idle`, actual `replay_repetitions` with captured HTTP responses at its transport seam, existing worker launch/readiness evidence, and task-local successor `native_idle_evidence`. CPU-only tests do not qualify a native GPU run.
