# W128-FULL requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded the delegated Q11 producer scope and fixed-workload authority. |

## [Original Request] W128-FULL

Implement the approved GLM DSA FULL dummy capture/replay profiling producer in
the isolated `glm-full-dummy-h200` worktree, based on `9ed979e`. The current
formal workload is 128 requests of 6144/1024 tokens at fixed QPS 1, 64 H200,
P32/D32, TP8/DP4/EP32 per role. Native capture sizes remain parameters until the
native configuration is frozen. Read-only authority is the parent task's
`requirements.md` (Q11 and D-FIXED-WORKLOAD), `plan.md`, ticket 07,
`glm_full_path_20261020/notes.md`, and its 2026-10-20 test report.

## [Decision] Q11 / W128-FULL

The primary delegates the owner's approved continuation: "User approved
continue FULL producer independently of H200 collection." Only DSA profiling
files and focused tests belong to this slice. Preserve typed coverage errors;
ordinary serving q1/context0 rows do not qualify. Do not change consumers,
shared interfaces, simulator, scheduler, metrics, or communication code. No
GPU/StepMind operations, pushes, merges, or shared-task-record edits.

Pinned native source: `/data/ycfeng/Frontier/.real-engine/vLLM-BS`, commit
`2cf0a6915ce544dc493a0990f2ea38d81601128a`; inspect with `git show` because its
working tree is another revision. The repair-approval reference was read before
implementation. This is an approved producer capability, not permission to
weaken a consumer gate or to declare measured coverage.

## Acceptance

1. Native dummy metadata, invalid KV slots, parameterized capture widths,
   both index variants, and actual whole-model FULL replay in executable code.
2. Replay scope attribution does not depend on Python hooks executing at replay.
3. CPU mock/source-contract tests and existing DSA focused checks; no claims
   of GPU measurement or native-runtime qualification from these tests.
4. Exact output schema, coverage mapping, native references, launch requirements,
   and a bounded consumer-admission proposal, without implementing admission.
5. Commit explicit owned files after checks; report remaining GPU-only blockers.
