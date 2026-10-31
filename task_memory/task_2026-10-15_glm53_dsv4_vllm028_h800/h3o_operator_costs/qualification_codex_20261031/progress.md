# P2 H200 qualification investigation progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Completed the continued evidence verification and scoped Git delivery checks; retained the original task date and source receipts. |
| 2026-10-31 | Created the authorized read-only P2 qualification investigation record. |
| 2026-10-31 | Resumed exact artifact reads; reconciled packed descriptors, exact startup memory counters, F4 terms and original admission. |

## Current milestone

- completed: B30b is already committed locally as dbd3ff6f and integrated by the lead as 5d1754d0; no implementation work is reopened.
- completed: the new sidecar started from a clean worktree; read the main current route and ticket checkpoints.
- completed: exact packed-payload, non-KV/capacity and original admission/promotion evidence reconciliation.
- in-progress: final report verification, check-progress checkpoint and scoped report commit.

## Evidence milestone — 2026-10-31 continuation

- completed (step 1): current main authority remains `5d1754d0`; the first authority check was unchanged at initialization; the lead later changed main plan/progress while retaining the same current-route authority. The sidecar remains based on `dbd3ff6f`. No other process was found with the sidecar as its working directory.
- completed (step 2): clean reference logs confirm one packed region and 1,002,240 B/block. Source-selected T6144 counts [24, 2, 2, 2, 16] imply 46,103,040 descriptor bytes/lane versus 39,521,664 component bytes/lane; the 6,581,376 B gap is reported without repair. Actual per-request wire selection remains unobserved.
- completed (step 3): located the existing `stage4/f_variants/f4/p2_{32,64}_init/planner_blocks.json` files from the main command/report pointers. Read only startup memory/configuration fields in the 32+64 retained rank inventories and 12 named server logs. Exact native requested memory is 127,603,313,869 B; F4 requested memory is 128,687,957,606 B. Counts are 116,034/120,623 versus native 109,934/112,914. Workspace inventory is 1.5/3 GiB per rank; independent stage peaks remain explicitly unverified.
- completed (step 4): both original R4 results are PASS under their preserved 2026-10-25 authorization; P2-64 explicitly includes clean reference and basic validation. Neither original manifest binds a promotion receipt. Retained B29 numeric results are reused; no A6/A10 execution or gate run is repeated.
- in-progress (step 5): retain selected evidence with source hashes, write the English criterion table, verify arithmetic and references, run the task-local check-progress checkpoint and create the scoped report commit.

Path discovery misses were limited to stale guessed F4/script filenames. Existing report pointers resolved the planner JSONs; these misses are not source defects. The user reports separate lead-owned running work; this audit does not inspect those jobs, sessions or outputs.

## Verification milestone — 2026-10-31

- completed: collected the already running capture (session 51299), exit0: 160 sources, 96 rank inventory rows, 12 node results, 12 startup logs and 24 retained B29 metric rows.
- completed: first retained-byte verification at 15:49:34Z, exit0 PASS: 160 hashes, 73 source excerpts/JSON pointers and 384 rank pointers. Exact command: `env TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/qualification_codex_20261031/evidence/audit.py --verify`.
- completed: an initial capture failed because its initial-authority-hash assertion encountered the lead's new plan bytes. Preserved `initial_sources.json`; current plan SHA-256 is `939f953ba278cf62413e7ae00f208460374d3ffeba2dda38c38ca31f55887e5e`, current progress SHA-256 is `cfe15f6d87f593047142f474d02d9425550932cc6eacadb981b06e74d9ff5c75`. Current bytes and changed-since-initial flags replace the inappropriate frozen-authority assumption. Native artifacts were not changed.
- completed: resume snapshot confirms sidecar tracked/index diff is empty at `dbd3ff6f`; no independent writer with this worktree cwd was observed. Existing ignored local task files were preserved.
- completed: English report contains 14 criteria, exact byte/capacity terms and all retained B29 repeat errors. Small task-local checks were added for all packed-registration rows, graph-memory log rows, report references and existing arithmetic; no project source or test was edited.
- completed: final capture and report verification after the Ticket09 excerpt correction: 160 hashes, 73 selectors, 384 rank pointers, 96 packed rows, 96 graph-memory rows, 37 source keys, 14 criteria and 24 metric rows PASS at 15:58:36Z. Three native source files also match their original pinned git objects byte-for-byte.
- in-progress: final task checkpoint, scoped Git staging and commit.

A one-off display command failed with `TypeError: unhashable type: 'list'` when treating workspace inventory lists as scalar set members. The evidence collector was unaffected; JSON list display then confirmed workspace lists `[1610612736]` and `[3221225472]`. This is a display failure, not a native memory or qualification failure.

## Final evidence checkpoint — 2026-11-01 Asia/Hong_Kong

- completed: the final reproducible `audit.py --verify` command passed at 2026-10-31T16:02:56Z (2026-11-01 00:02:56 Asia/Hong_Kong), including the three original native git objects. All previous evidence counts and limits remain unchanged.
- completed: check-progress found no D1-D7 drift or S1-S6 scope/acceptance defect in this sidecar. Steps 1-4 PASS; step 5 needs only the scoped commit and completion record.
- pending: stage and commit the verified report/evidence, then record the exact commit and changed paths. The lead-owned formal prerequisites are findings, not blockers of this investigation delivery.
