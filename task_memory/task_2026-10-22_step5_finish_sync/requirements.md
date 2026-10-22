# D30-STEP-FINISH-SYNC requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Merge b8be287 approval; retire the five-file limit and D29 dependency while retaining child evidence. |
| 2026-10-22 | Reference parent approval and record isolated child scope. |
| 2026-10-22 | YC approves D30-SCOPE-7-750 independently of D29. |
| 2026-10-22 | Record approved Step workflow correction and observation scope. |

## [Original Request / Decision] D30

Authoritative verbatim question, answer and option text:
`../task_2026-10-15_step5_preview_h800/requirements.md`, D30 Step finish sync.
YC selected **批准同步对齐 (Recommended)**.

Match current Step native finish synchronization on wave step1 and each16
steps, with counter reset at wave completion. Retain other models' existing
behavior and legitimate dummy participation. Use the existing native-family/
runtime contract and scheduler owners; no global32-to16 replacement or
unnecessary alternative configuration path. The approved scope includes
necessary configuration/scheduler changes, focused boundaries and native
counter observation. Native scheduling itself is unchanged.

Preserve request/READ/readiness/source-release semantics, legacy artifacts,
source-bound input identity, full CLI2400s limit and D24 resource bounds.
No native CPU overhead/gap analysis or measurement suite. Observation errors
are isolated and must not affect native calls, returns or exceptions.

## [Original Request / Decision] D30-SCOPE-7-750

The exact parent question and direct answer are recorded in the main task's
`requirements.md`, D30 scope expansion. YC answered:

> 批准，继续

The approved scope is seven source/test files and at most750 total insertions
plus deletions, including candidate `cf1a53f` and its follow-up, against the
integrated non-D30 parent source. Parent `d581625`'s existing READ correction is
preserved and excluded from D30 ownership.

1. `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py`
2. `tests/integration/step5_pd/r2_observation.py`
3. `tests/unit/test_step5_finish_sync.py`
4. `tests/unit/test_step5_native_pd_r2_observation.py`
5. `tests/integration/step5_pd/r2_verdict.py`
6. `tests/unit/test_step5_pd_placement.py`
7. `tests/unit/test_step5_measured_pd.py`

Repair the six documented fixture instances using actual all-lane/dummy/release
events, without bypassing finish consensus. Preserve all ownership, READ,
readiness, output and retry assertions. Keep diagnostic metadata out of
request/transfer evidence; metadata-only evidence remains UNVERIFIED. Complete
affected checks, parent verification, independent review and strict77 before
accepting integration. Native image/package/submission remain parent-owned.
D29 proceeds in parallel under its own unresolved numerical scope.

## [Decision] Isolated execution and existing ticket

Worktree `/data/ycfeng/Frontier/.worktrees/step5-finish-sync`, branch
`feat/step5-finish-sync`, original base `fa5bae2`, integrated baseline `b8be287`.
No parent edits, native routing/scheduling, profile/workload/distribution changes,
child agents, Docker/platform/auth/cloud/SSH/push, full CLI or CPU-overhead suites.
Commit verified source/tests separately from task records. Existing ticket19
`../task_2026-10-15_step5_preview_h800/tickets/19-graph-host-scheduling.md`
and live native qualification remain open. No new GPU fence, collective or
tensor readback. Missing first-idle step1 sync can cause31 extra iterations;
the retained P snapshot at step11 does not establish the first idle boundary.
