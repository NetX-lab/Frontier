# D29 performance final regressions archive

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Verify19compactarchivefiles and5scriptbindings,with no stagedfiles/agentcommits. |
| 2026-10-23 | Complete required fresh79 regressions with8018raw unitPASS,new3PASS,and52PASS governance supplement. |
| 2026-10-23 | Record completed short/strict gates and explicit remaining full-unit work. |

TaskID: D29-PERFORMANCE-FINAL-REGRESSIONS. Fresh immutable source
79f1a3fd35d42b751cd1520c764ddac717b923d9, tree
8c6db078e852d3b885a6f14bdf5e8007d9ecddb7, submodule
3dac53c544e546574cb6e8078cd14053d0916113. Source worktree:
`/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-06`.

## Independent results

| Selection | Current recorded result | Status |
| --- | --- | --- |
| strict77 final05 | 77/77 identical,zero differences;run/compare/audit exit0 | Complete PASS |
| same-source supplement101 | 101PASS/0FAIL/ERROR/SKIP,exit0 | Complete PASS |
| original metadata196 file selection | Actual197PASS/0FAIL/ERROR/SKIP,exit0;new parser ID passes | Complete PASS |
| catalogue53 | 53PASS/0FAIL/ERROR/SKIP,exit0 | Complete PASS |
| full-unit parent05 | Actual8018PASS/279FAIL/32ERROR/281SKIP,rawexit1 | RawFAIL;exact regression gatePASS |
| conditional governance52 | 52PASS/0FAIL/ERROR/SKIP,exit0,sameparent52IDs | CompletePASS |
| final source/hash acceptance | 2288pre-launch files unchanged;HEAD/tree/submoduleclean | CompletePASS |

No selection totals are summed. Strict77 uses unchanged original713e6a8 harness,
definitions and retained baseline;each side451 artifact hashes and519 cache files
verified,complete clean-cache/provenance/scenario/dependency equality. All three
short selections retain previous IDs;metadata adds one passing parser case.
Actual metadata CSV bytes remain equal e41 and legacy baseline where applicable.
All267FAIL/31ERROR locked historical IDs remain. Failure/error sets exactly equal
e41;three new case IDs allPASS,zero case loss/PASS-to-SKIP or prior status changes.
Fresh101PASS supplement covers12 identical missing-aiohttp IDs and16 native cases
behind1 missing-Torch collectionERROR. Fresh52PASS governance supplement covers
five existing missing-Torch failures only after full-unit showed them unsettled.
All parent81/1oldSKIP,2,128,29IDs are preserved and priorPASS remainPASS. Metadata
197/catalogue53 also pass as full-unit subsets. No duplicate affected union run.

Final verdict: `PASS_REQUIRED_REGRESSION_GATES`. Rawunit remains unsuccessful;
no new code failures after explicit environment comparison. All required subtasks
completed,zero failed required gates,zero unfinished required gates.

## Deliverables and exact paths

Owned records:
`/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/performance_regressions/`.
Files include requirements/plan/progress, independent dated short/strict/unit
reports, checkpoints and compact source-stamped `evidence/` JSON receipts.

Strict raw output:
`/data/ycfeng/tmp/d29_workflow_fidelity_20261023_final05/`.
Candidate manifest/results SHA256 f597643a.../11247b36...;comparisonaae7a53b...,
full hashes in `evidence/strict77/final05_receipt.json`.

Unit raw targets:
`/data/ycfeng/tmp/d29_unit_parent05.xml`, `.log` and basetemp without suffix.
New runner/audits/short XML/logs/isolated cache:
`/data/ycfeng/tmp/d29_performance_regressions_final05/`.
Exact executed commands and environments retained in each execution.json;
runner SHA25662ce1aa3... in `evidence/command_bindings.json`.
Unit XML SHA256 `cb64bf1bfbba142b5ad7b3fbad90343d1f9134e6a627c35af5017bfc3654a4d8`.
Final comparison SHA256 `55a787e63413241c4f285698f990ab3043d67545453ce11d987209efa6e53f58`.
`evidence/final_acceptance.json` binds all six raw suite results,logs/XML/execution
hashes and seven changed-path affected coverage. `evidence/source_final_receipt.json`
binds the full pre-launch inventory and clean final source. Independent dated
governance report records exact approvedextras environment and module IDs.
Final record verification:19compactarchivefiles hash/sizePASS,5scriptbindings
unchanged,11Markdown/26JSON checksPASS;no ownedfilesstaged or agentcommits.
Exact receipt: `evidence/record_verification.json`.

## Remaining acceptance and limits

No required regression remains unfinished. Primary archives and accepts these
uncommitted records;this agent did not commit. Raw historical unitFAIL/ERROR/SKIP
remain by design,including the unchangedTorch group tensor-interface skip.

No source edit, checkout, commit, dependency installation, full CLI/GPU/network
run or further delegation occurred. Historical raw unit failures/errors/skips
must remain visible;regression acceptance is not all-green unit qualification.
Primary main08,native/GPU/G3/E2E acceptance remain separate. All old artifacts
are preserved and no stale e41 run label is reused.
