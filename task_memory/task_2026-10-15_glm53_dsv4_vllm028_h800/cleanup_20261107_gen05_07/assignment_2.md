# Task: execute the prepared generation05-07 scratch cleanup (correction to request 890bf5f2)

## Goal
Request 890bf5f2-a7eb-473d-ac18-51c83fd2de58 classified the scope and wrote
`<TASK>/cleanup_20261107_gen05_07/deletion_manifest.json` (449 candidates: 87 archived duplicates, 360 Node
compile-cache files, 2 superseded prompts; 32 uncovered files kept), then stopped on two prerequisites. The lead has
resolved both (below). Revalidate each candidate against current content, delete exactly those 449 files plus the
directories left empty by them, and write the final receipt and report.

Acceptance:
- Each deleted file still has the manifest's sha256 at deletion time; archived duplicates still match their covering
  archive file's current sha256.
- The 32 uncovered files, all non-scratch label records and both generation08 prompt pairs remain unchanged.
- Readable protected files hash-identical before and after, with the known exceptions below listed, not treated as
  failures.
- `deletion_receipt_final.json`, an updated `report.md`, counts and bytes (logical and allocated) per category and label.

## Lead decisions on the two prerequisites
1. Host process check. The lead accepts this evidence instead of a host-namespace scan, which the sandbox cannot do:
   every label in scope has its Pi `exit` file; the newest file in each label was written between 05:39Z and 08:00Z
   today (more than 80 minutes ago); the three launchers have `launcher_exit.json` in
   `<TASK>/h200_mainline/p2_64_formal_retry0{5,6,7}/package_01/`; the three monitors have `final.json`. Recheck these
   facts immediately before deleting and stop if any candidate's label received a write after 08:05Z.
2. Protected-input completeness. These exceptions are unrelated to the deletion scope and are accepted as known:
   - the five unreadable paths under `/data/ycfeng/tmp/glm53_dsv4/` (owned by another uid, from container tests);
   - `<TASK>/native_track_20261020/g0d_launch/launcher.log`, which an old launcher outside this task step keeps
     appending to (known since an earlier session, progress.md "finding (not this task's job)"). Do not touch it.
   Protection is shown by the readable-file hash comparison plus the exact deletion list; do not re-hash the 4 GB log.

## Scope and context
- Working root: /data/ycfeng. `<TASK>` = /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800.
- Previous report: `<TASK>/codex-implement-890bf5f2-a7eb-473d-ac18-51c83fd2de58/implementation.md`; reuse its scripts
  and records in the record directory.
- Same writable paths and protections as the first assignment (`assignment.md`): deletions only inside the 13 labels'
  `scratch/` trees and the two named prompt files; records only in the record directory. No `mv`, no Git, no other
  cleanup, no platform or network writes.

## Expected return
- Report language: English. Deleted and kept counts/bytes, checks with expected and actual results, the exact receipt
  and report paths, and anything left undone.
