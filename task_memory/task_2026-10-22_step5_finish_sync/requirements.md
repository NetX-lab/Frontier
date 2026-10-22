# D30-STEP-FINISH-SYNC requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Reference parent approval and record isolated child scope. |

## [Original Request] Delegated D30 scope

Authoritative parent documents, read before work:
`/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_finish_sync/{requirements,analysis,plan,progress}.md`.
Verbatim YC decision: parent `task_2026-10-15_step5_preview_h800/requirements.md:216-234`.
YC selected **批准同步对齐 (Recommended)**: implement Step-scoped wave finish
synchronization at step 1 and every 16 steps, with wave reset; preserve other
models, request/transfer ownership, legitimate dummy participation, the 2400s
CLI cap, D24 resources and CPU-overhead-last ordering.

Worktree `/data/ycfeng/Frontier/.worktrees/step5-finish-sync`, branch
`feat/step5-finish-sync`, base `fa5bae2980366da1d075125d6c26248a737fc419`.
Own source/test correction and narrow counter observer only; at most five
source/test files and 500 changed lines. No parent edits, native routing or
scheduling changes, profile/workload/distribution changes, child agents,
Docker/platform/auth/cloud/SSH, full CLI, wall profiling or CPU-overhead suites.
Native package/image/pins and GPU operations remain parent-owned. Required
focused RED→GREEN: step 1, step 16, reset, active peer, request-free dummy,
legacy 32 and unaffected DP1. Parent owns full strict77 and live observation.
Commit verified code/tests and task records separately.

## [Decision] Existing ticket and acceptance limits

This is a bounded increment under existing
`../task_2026-10-15_step5_preview_h800/tickets/19-graph-host-scheduling.md`.
It does not close ticket19 or native qualification. Scalar observation failures
must be isolated; original native arguments, call count, return and exception
must remain unchanged. No new GPU fence, collective or tensor readback.
Missing native first-idle step1 sync can cause 31 extra iterations; the retained
P snapshot at step11 does not determine the native first idle boundary.
