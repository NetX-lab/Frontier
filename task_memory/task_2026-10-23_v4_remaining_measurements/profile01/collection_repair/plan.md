# W128-PROFILE03-COLLECTION plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded the bounded implementation and verification dependency graph. |

## Dependency and ownership

`read approved design/source -> baseline and RED controls -> two-file worker repair -> GREEN controls -> {worker suite, manifest suite, profile-driver suites in separate processes} -> unchanged-domain matrix binding -> source/records commit -> clean worktree and stop`

The owner performs this slice without further delegation. Root reviews and integrates independently. Only the two allocated source files may change; all owned records stay here. No package or native runtime change.

## Implementation and acceptance

1. Reuse `node_plan`'s existing absolute local Torch profiler window; validate the task-local role mapping like existing role block overrides. Copy the selected profiler configuration and preserve the original input, role/lane path and unrelated fields. No production iteration/record literals.
2. Derive drain paths from the already-known peer plan. Use the same paths for quorum and retained membership. Preserve the existing failed-peer/native checks and polling loop.
3. Change only the headless post-stop-marker budget to `peer_wait_seconds`. Record local monotonic start/end times for quorum wait, head marker wait and native stop in a separate diagnostic field; preserve frozen shutdown semantics and native stop behavior.
4. Extend the existing `exercise` seam with deterministic time/filesystem controls. Verify delayed markers for natural and active headless processes, soft late/missing markers, native errors, delayed enumeration, partial/never-complete peer quorum and role configuration copying.
5. Run affected existing suites separately with the approved CPU environment. Compare exact outputs, not just exit status; classify any failures before changes. Explicitly bind the existing77-case matrix through unchanged domain trees and retained comparison hashes.
6. Re-read requirements; record original failures, commands/results, budgets and hashes. Commit verified code and records. Confirm the worktree is clean and stop.

## Budgets

- Production: `native_pd_worker.py`, additions plus deletions <=80.
- Tests: `test_native_pd_worker.py`, additions plus deletions <=180.
- Total source <=260; records separately accounted.
