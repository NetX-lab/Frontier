# D30 isolated implementation plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Eight-file603-line repair integrated at4ec5b23 after review, affected/unit-delta and strict77 checks. |
| 2026-10-22 | Record completed implementation and affected checks; retain parent-only acceptance steps. |
| 2026-10-22 | Merge independent D30-SCOPE-7-750 approval and start implementation; no D29 dependency. |
| 2026-10-22 | Add missing verdict consumer and six fixture instances; record seven-file numerical proposal pending approval. |
| 2026-10-22 | Record dependency chain and focused acceptance. |

Completed candidate chain:
`approval/source inspection/design` → `focused RED` → `scheduler + observer`
→ `focused GREEN + legacy checks` → code `cf1a53f` + records `defe596`.
Historical candidate was unmerged and not merge-ready: adjacent fixtures failed
and the missing verdict consumer was outside the128-pass adjacent selection.

Current chain:
`reviewer findings + read-only inspection` → `complete scope records`
→ **YC independent D30-SCOPE-7-750 approval (b8be287, completed)**
→ {`diagnostic metadata consumer + emitter-to-loader regression`,
   `six fixture instances through real all-lane event driver`}
→ `focused affected verification + phase review` → parent integration/strict77
→ parent fresh native scalar observation.
The two implementation arms are now authorized independently of D29;
they touch separate existing files. No new subagents are authorized here.

Current status: source/CPU implementation and integration complete at4ec5b23.
Final D30-SCOPE-8-750 amendment3d6744f extends only the obsolete canonical
source-count assertion: eight files/530insertions+73deletions=603 against b8be287.
Independent review No findings; parent420 checks,65 canonical checks, baseline
unit failure/error-set equality after the affected rerun, strict77 and902
artifact hashes pass. Merged source/tests exactly match the accepted tree.
Fresh native counter qualification remains open under the new-generation work.

1. Read approval/source and existing owners; record design and human receipt.
2. Add focused tests: step1, 16, reset, active peer, dummy, legacy32 and DP1.
3. Apply family-scoped scheduler policy and existing-install scalar observer.
4. Verify boundaries and fault isolation, code size, diff scope and whitespace.
5. Commit verified source/tests separately from records; report exact paths,
    commands, logs, commits and unverified live-native evidence to the parent.

## Current follow-up scope and acceptance

Implementation authorization is recorded at b8be287 and the approved receipt
in `repairs/human_review_receipt_D30-SCOPE-7-750_2026-10-22.json`. Merge the
non-D30 parent baseline, preserve d581625, implement only the seven listed paths,
run affected checks, then commit code and records separately. No profiling,
GPU/platform action, D29-native design inspection or root overwrite.
Original Step1/16 semantics, legacy32/DP1 behavior, native original
call/return/errors,2400s CLI cap and CPU-overhead-last remain unchanged.

The COMPLETE seven-file source/test proposal and per-file budgets are recorded
in `issues.md` (the earlier six-file/<450-line estimate is superseded):
`4 existing + 3 additional = 7 files`; `327 existing + 400 additional = 727`
budgeted changed lines; approved cap **7 files / <=750 changed lines** versus
b8be287, including cf1a53f. All insertions and deletions count. No new verdict-test file or shared
driver module is omitted from the count: regression reuses already changed
`test_step5_finish_sync.py`, and event fixture support lives in the already
proposed two PD test modules. If these budgets cannot fit complete semantics,
stop and present measured expansion before further edits.

Acceptance of the two remaining arms is:

1. Metadata consumer: emit real observer JSONL for both P/D and both counter
   events; load/judge without Unknown event or request/engine/TP validation;
   retain exact raw rows with location separately from request records. Valid
   request verdicts remain unchanged, metadata-only evidence remains UNVERIFIED,
   unknown/error events retain rejection, and no D30/native PASS is introduced.
2. Six instances in five PD functions: drive real replica/stage arrivals,
   schedule, dummy completion and consensus release events; preserve output,
   context, request ownership, P response/source retention, capacity retry and
   READ/readiness/cost assertions. No direct `_end_iteration`, barrier override
   or omitted returned event. Include the decode loop revealed by parent
   d581625's byte correction, and keep its byte/block expectation unchanged.
3. Required focused verification includes the unchanged existing verdict test
   module (omitted previously), emitter/consumer regression and all six fixture
    instances in their actual supported configurations. Use frontier-py310 for
    simulator tests and asyncopd-py312 for aiohttp R2 checks.
4. Parent phase review and strict77 remain dependent on these fixes. Native
   source/package/image/GPU work stays parent-owned; raw counters alone do not
   establish native scheduling or end-to-end acceptance.

Acceptance is the complete focused matrix, not merely a successful test exit.
No full CLI or CPU-overhead suite. Parent owns strict77 and native package/pins.
Coordinate shared-file merge order with D29 later, not as an approval dependency.
