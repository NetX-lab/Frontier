# B30b sidecar verification plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-31 | Recorded the scoped verification route and acceptance checks. |

## Scope and dependency

`1 -> {2, 3, 4} -> 5`; tests, plan generation/preservation and source review can run independently before edits. Any verified repair reruns only affected checks plus the required final suite.

| Step | Work | Completion evidence |
| --- | --- | --- |
| 1 | Read authority and existing artifacts; confirm writer absence; snapshot original diff and protected files. | Original snapshot, exact frontier file/line counts, recorded decisions. |
| 2 | Execute B30b, affected collective profiling and G5h launch tests. | Exact commands/environment, PASS/FAIL and regression evidence. |
| 3 | Generate both fresh stages, independently count cases and budget arithmetic, and recheck historical plan preservation. | Fresh JSON/text plans, phase/job budget reconciliation, unchanged successful G5h/K5 listings and historical failure set. |
| 4 | Review profiling admission, KV geometry, launch-depth detection, rank placement, soft runtime checks and CPU analysis. Repair only demonstrated in-scope defects. | Separate standards/spec findings, any failing reproductions and verified repairs, scope count. |
| 5 | Re-read local requirements, finish test report and English summary, run progress self-check and create scoped commit. | Commit and changed-path list; protected main files/READMEs unchanged; all deliverables available to integration owner. |

## Limits

No GPU/platform/cloud actions, simulator/backend/schema changes, main task-record edits, README edits or deletion/move operations. Main owner integrates the resulting branch and performs hardware acceptance and submission.
