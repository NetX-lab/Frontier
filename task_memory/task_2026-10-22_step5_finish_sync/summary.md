# D30-STEP-FINISH-SYNC candidate handoff

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Archive bounded source/CPU candidate and explicit integration blockers. |

## Outcome

The isolated candidate aligns registered Step DP waves to first-step/every16
finish synchronization and records native host counters through the existing
R2 install/pin/soft-error path. Legacy32 and DP1 remain unchanged in focused
checks. This is a source/CPU candidate, not a completed D30/native milestone.

Code commit: `cf1a53f63eef0492c369517f38505d7d7283c4df`.
Branch `feat/step5-finish-sync`, worktree
`/data/ycfeng/Frontier/.worktrees/step5-finish-sync`, base
`fa5bae2980366da1d075125d6c26248a737fc419`. No parent files were edited.

## Exact deliverables

- `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py:62,164-169,588-610`:
  registered-family policy, one-time binding and existing consensus boundary.
- `tests/integration/step5_pd/r2_observation.py:45,1173-1210`: native core pin/install
  inventory plus scalar finish-check and observed zero-counter seams.
- `tests/unit/test_step5_finish_sync.py:27-261`: first-step/16/reset/active-peer/
  dummy/legacy32/DP1 checks, soft-error fault injection and SHA-pinned native
  method CPU execution without vLLM import.
- `tests/unit/test_step5_native_pd_r2_observation.py:930-936`: existing source
  inventory now includes native core (13 files, previously12).
- Task records in this directory: `requirements.md`, `design.md`, `review.md`,
  `plan.md`, `progress.md`, `issues.md`, `checkpoint.md`, this `summary.md`, and
  `test_report_2026-10-22_finish_sync.md`.

Source/test diff:4 files,319 insertions/8 deletions,327 changed lines. Changed
modules remain below2000 lines; whitespace checks pass. No arbitrary model
string classifier, new CLI flag/scheduler, GPU fence/collective/readback,
native scheduling/routing edit or workload/profile/distribution edit.

## Observed validation

- Scheduler RED9FAIL/7PASS → focused GREEN27PASS65.58s.
- Observer RED5FAIL/1PASS → final included fault-isolation checks pass.
- Existing R2 observer/stage/forward-census128PASS1.98s in Python3.12.13.
- Existing legacy32/DP1 runtime3PASS16.10s in Python3.10.6; request/token/lane
  conservation assertions pass.
- Native config/core SHA256 match the parent pins. AST-extracted original
  native finish method calls only its existing CPU-substituted collective at
  1/16/32; this does not prove live native collective timing.

Detailed commands, logs/XML paths and environment are in the test report.

## Open work and scope decision

Extra P/D adjacent suite fails6 cases (68PASS); read-only baseline scheduler
fails1 (73PASS). Five newly failing tests drive only their selected lane and
never its peers, so they stall correctly at the newly represented first-step
finish barrier. The remaining READ-byte expectation already fails on baseline
and the parent owns D29 changes to that file.

Minimal complete test correction adds two existing PD test modules:6 total
source/test files, estimated<450 changed lines. This exceeds the five-file
bound, so further source/test edits are stopped. Parent must authorize that
specific expansion or integrate with its D29 fixture corrections. No regression
is suppressed or waived, and this candidate is not declared merge-ready.

Parent phase review, common-branch integration, strict77, fresh native staging,
loaded core/config source pins, all-DP scalar counter completeness, active-peer
continuation, first idle boundary and reset are pending. Native image/package/
root GPU operations remain parent-owned; no platform operation was performed.
