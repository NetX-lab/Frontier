## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Started independent diff-driven quality cleanup. |

# Progress

## P0 — in-progress

- Read all 817 lines of AGENTS.md, prior requirements/plan/progress/current status, and relevant validation records. No nested AGENTS.md found in source/test/data/docs search.
- Pinned `main=0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`, `candidate=c288a19f59bec09529ee18d782fa57218da2c781`, identical merge-base/main. Initial tracked worktree clean; 266 changed files including substantial historical evidence.
- Local clock confirms 2026-09-17 Asia/Hong_Kong. No new remote access or publication performed.
- Planning-with-files applies using this task directory; code-review supplies independent Standards/Spec reviews; codebase-design informs invariants/ownership, subordinate to user scope.
- Read-only Standards worker owns profiling/training/config review; Spec worker owns prior behavior/verification boundary check. Main owns runtime inspection and all implementation.
- Previous task reported 18 known main-equivalent CPU failures and explicit native hardware skips. These are leads for fresh verification, not current PASS evidence.

## Pending work

1. Build ranked module inventory and preserve executable pre-refactor reference.
2. Establish fresh baseline evidence and inspect runtime changed hunks.
3. Execute P1–P5 in dependency order.

New unresolved issues: none established.

## P1a — completed: construction and finalized snapshot contracts

- Preserved pre-refactor candidate in `../quality-baseline-c288a19f`; pinned main reference remains `../pr33-r12-baseline-20260915`.
- Default Python 3.12.3 lacks pytest/dependencies. Consulted environment/package-mirror handbooks, then installed isolated CPU/test requirements with uv in `/data/ycfeng/tmp/quality-review-env`. No shared environment was modified.
- Fresh full-unit collection failed on eight missing-torch imports and one missing-matplotlib import. Public CPU Torch index timed out after three retries; logs: `/data/ycfeng/tmp/quality-pre-unit-env.log`, `/data/ycfeng/tmp/quality-env-torch.log`. Matplotlib installation through the internal mirror succeeded. Torch resolution remains pending.
- Motivation: Simulator duplicated predictor construction; ExecutionTime treated constructor-owned components as absent; StageExecutionTime repeated filtering after rejecting missing IDs.
- Method: one predictor construction loop, direct required component access with isolated operator-map copies, uniqueness check on validated IDs. Ordinary monolithic manager/path remain None; disaggregated/hybrid sharing and physical-layer identity remain unchanged.
- Checked CC backend lazy construction: backend factory reads cluster configuration/runtime topology; manager owns training/cache state, not backend configuration. Shared initialization remains before predictor creation.
- Verification: frozen candidate eight non-dummy cases passed (65.34 s). Cleaned candidate 78 focused/non-dummy tests passed (71.00 s). Existing comparator found 90/90 stable artifacts equal across eight cases. Bidirectional artifact inventory and acceptance/stage-summary comparison follow the independent Spec review recommendation.
- Editing command `apply_patch` was absent from PATH; no patch was applied on the first attempt. Resolved by invoking the installed Codex binary in its apply_patch mode, without changing system configuration.
