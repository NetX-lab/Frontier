# D29 run-fixed collective input completion archive

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Archive accepted source component db581ff6; whole-workflow gates remain open. |

## Overview and deliverables

Implemented YC's fixed-per-run collective input decision. P/D constructors bind their backends to the existing shared prediction manager's `CollectiveProfileInputs`; standalone backends use the same loader. A successful UTF-8/JSON read publishes immutable bytes by normalized configured path. File updates become visible to a new run, while current numerical, point, count, context, buffer and actual-layer proof checks remain live.

Source commit: `db581ff6ad43e1ce57361ed00fc4be0256e85a45` in `/data/ycfeng/Frontier/.worktrees/step5-d29-input-snapshot`. Six source modules and four existing test files are listed, hashed and accounted in `evidence/parent_component_receipt.json`. The new module is `frontier/cc_backend/collective_profile.py`; no model identity, native numerical cache key or output reduction was introduced.

Local task artifacts:
- `design.md`: ownership, interface, failure and verification contracts.
- `test_report_2026-10-24_run_snapshot.md`: exact RED/GREEN commands and observed results.
- `review.md`: independent Standards0/Spec0 findings on the complete byte-plus-snapshot diff.
- `wave/test_report_2026-10-24_run_snapshot_wave.md` and `wave/evidence/`: actual208-event replay, source binding and exact commands; records commit7c73fdd1.
- `evidence/parent_component_receipt.json`: source hashes,212/340 gross(73source/139test), net86 and component accounting.

## Observed validation

Two behavioral baseline REDs became GREEN. Four affected files passed82 tests; the separate two-trainer selection passed2 tests; five configured-path checks passed. Counts remain separate. Independent reviews returned No findings. Real wave208-event values/order, whole JSONL/ledger, snapshot isolation,64 wrong-key negatives and3902-byte error recovery all matched. P/D binding performed one file read; cold/warm waves performed none. Warm median0.340867→0.162715s is executor evidence;288 numerical lookups/144 fresh proofs/64 collective prices per wave are unchanged. Parent verified28 manifest,41 source,6 script and6 raw artifact hashes plus complete JSONL/ledger equality.

Component reconciliation: CORE4173/4404, PRICING3940/3956, native1350/1582;9463 source/9477 global. This is not final114-path union accounting.

## Open work and limits

Accepted code/records are integrated at64c83ac4 with source/test/config bytes identical to db581ff6. Perform final source-union accounting and frozen regressions after the separately allocated key successor. Complete6144/1024 CLI under2400s/mode, T2 sealing, native/GPU/G3/E2E qualification remain unfinished. Cached-manager wave reconstruction is not full Simulator execution. Config-scan and private-encoder candidates remain unallocated; the public `to_dict` conversion candidate is rejected. Existing root-owned cache/pyc files remain untouched.
