# D29 final budget audit on integrated-06

TaskID: D29-FINAL-BUDGET-AUDIT
Audited tree: `/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-06` at `79f1a3fd35d42b751cd1520c764ddac717b923d9`
Baseline: `1aee0af11cd58691e64603b99eca0ac459a5b350` (merge-base identical)
Status: audit complete; no per-path or owner violation; no reallocation required

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | First audit of frozen integrated-06 (`79f1a3fd`) against the 111-path effective ledger and nine ordered overlays. |
| 2026-10-23 | Correction: step-receipt tables labelled a cap split as measured. Model-group measured source/test is 17/54 (cap 45/60) and artifact parse is 8/35 (cap 15/35). Registry 331/360 and fixtures 72/80 are marked checkpoint scopes. No measured total changed. |

## 1. Result

Every owner and every path is within its effective ceiling, so the ledger needs no adjustment and no
zero-sum transfer is required. The measured figures match the expected values exactly.

| Owner | Allocated paths | Measured gross | Cap | Free |
| --- | --- | --- | --- | --- |
| native | 19 | 1350 | 1582 | 232 |
| core_source | 46 | 3970 | 4404 | 434 |
| pricing | 46 | 3920 | 3956 | 36 |
| source/test/config | 111 | 9240 | 9942 | 702 |
| + gitlink 2 + module-size note 12 | | 14 | | |
| global | | 9254 | 10000 | 746 |

Composition: 9240 = 8808 superproject
(8525 added / 285 deleted including the gitlink pair)
+ 333 submodule + 99 native-image.

## 2. Ledger replay

`scope.json` holds 138 rows over 107 unique paths totalling 9942. The nine ordered overlays replay
without a single `before` mismatch or bad `before + delta`:

| # | Overlay | Rows applied |
| --- | --- | --- |
| 1 | registry | 6 |
| 2 | core_review | 18 |
| 3 | legacy_output | 3 |
| 4 | prepare_metadata | 2 |
| 5 | final_scope_reconciliation | 46 |
| 6 | cli_placement | 2 |
| 7 | model_group_reuse | 6 |
| 8 | init_validation | 3 |
| 9 | collective_artifact_parse | 6 |

Result: 111 unique allocated paths, ceiling total 9942, quotas 1582 / 4404 / 3956 unchanged.
110 paths actually changed; `tests/unit/test_step5_contract_reuse.py`
is allocated 5 lines with no diff. Composition: 46 core_source, 19 native, 46 pricing; by source
101 superproject (98 Python plus one shell, one Dockerfile, one JSON), five submodule, four native-image.

## 3. Owner mapping

`frontier/execution_time_predictor/moe_operator_times.py` (H04, 79/79) and
`frontier/profiling/moe/fp8_block_experts.py` (H17, 106/106) belong to pricing by the explicit
sole-writer rule. With that assignment the owner totals reproduce 1582 / 4404 / 3956 exactly.

## 4. New work in this tree

Seven code paths differ from `e41abc07`, and no record path is counted:
step gross 152 (145/7),
retained net 138.

| Group | Step gross | Retained net | Path ceilings | Measured |
| --- | --- | --- | --- | --- |
| model-group reuse `3763e5b8` | 71 | 59 | 175 / 128 / 260 | 143 / 111 / 254 |
| loader family `ee7e66ee` | 38 | 38 | 17 / 237 | 13 / 231 |
| artifact parse `51e1f720` | 43 | 41 | 93 / 233 | 84 / 233 |

The measured source/test split inside each step, for cross-checking against the cap split:
model-group 17 source and 54 test;
loader 4 source and 34 test;
parser 8 source (7 added, 1 deleted) and 35 test.

All 16 paths touched by those commits are byte-identical to their owner commit (16 of 16 hashes
match). The seven new paths give identical numstat under `git diff -w`, and all seven parse with
`ast.parse`, so no whitespace-only edit is counted as production work.

## 5. Step receipts

Cap split is source / test (allowance); measured columns are gross added-plus-deleted lines.

| Receipt | Cap total | Cap split | Measured source | Measured test | Measured total | Source |
| --- | --- | --- | --- | --- | --- | --- |
| CLI placement | 65 | not split | 0 | 40 | 40 | `ae1fb721` |
| Model-group reuse | 105 | 45 / 60 | 17 | 54 | 71 | `3763e5b8` |
| Loader family | 45 | 6 / 39 | 4 | 34 | 38 | `ee7e66ee` |
| Artifact parse | 50 | 15 / 35 | 8 | 35 | 43 | `51e1f720` |
| CPU plan consumer | 120 | not split | 19 | 92 | 111 | `78a475be` |
| Native T2 | 100 | not split | 0 | 85 | 85 step, 81 retained | `892b6a1` |
| Legacy output | 80 | not split | 12 | 25 | 37 step, 35 retained | `b60dfddd` |
| Added test paths | 60 | not split | 0 | 16 | 16 | `047f7313` |
| Prepare metadata | 40 | not split | 11 | 20 | 31 step, 23 retained | `3d51b9b2` |
| Registry (checkpoint scope) | 360 | not split | not split | not split | 331 | `a5189e19` |
| Affected fixtures (checkpoint scope) | 80 | not split | not split | not split | 72 | `f03a1157` |

Two receipts cannot be re-derived from a baseline in this tree. Core review 230/228 is carried as
recorded: `e49b26f8..HEAD` is 0 gross for its seven paths because that commit already contains the
repair, and baseline-relative totals are dominated by pre-review implementation. Registry 390 and
fixtures 106 exceed their caps only as cumulative figures across later separately authorized
repairs (+59 model-group, +34 loader) on the same paths; both were inside cap at their own
checkpoints. These two receipts are therefore checkpoint scopes, not current cumulative
measurements: 331/360 and
72/80 stand as the historical record, and the
later repairs are accounted by their own allocations. No reallocation.

## 6. Extras

- gitlink 1/1, `d28fe917` to `3dac53c544e546574cb6e8078cd14053d0916113`, submodule worktree clean.
- submodule patch 5 files 331/2 = 333,
  sha256 `c96cace541b72f7627b5f429d55615a17be4cd6c86d1827ef53969d05d766a1d`.
- native-image patch sha256 `66608153204012927e8ef678378d95ca1ef93dab9e2f504a82a5318b9828bbab`, 4 files,
  99 gross, four derived hashes unchanged, no image rebuild.
- module-size note: 12 added lines, 259 total, 14 table rows,
  sha256 `b9897e3be8f2f04048dcb3769c5d277ef46f5938582bcdf496259b56718429e5`, and its inventory sentence (98 / 5 / 4 Python paths) matches this tree.

## 7. Verification limits

Read-only: no tests, no checkout or merge, no Docker, no network, no GPU; no native package or
filesystem source rebuild. Submodule figures come from the pinned clean `3dac53c` checkout inside
this tree. Allocated-but-unchanged paths are reported, not counted as work. No maintained ledger,
overlay, document or source file was modified; the only writes are this report and its receipt.

## 8. Files

- `d29_final_budget_audit_receipt06.json` — 111 rows with before/measured/kind/source plus all facts
- `report_2026-10-23_d29_final_budget_audit06.md` — this report
