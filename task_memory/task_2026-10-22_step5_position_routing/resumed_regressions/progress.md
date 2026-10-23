# D29-E41 final regressions progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Full-unit parent04 exact-ID/environment gate closes; final2611-file hash check clean. |
| 2026-10-23 | strict77 final04 and full evidence audit PASS77/77; full-unit remains running. |
| 2026-10-23 | Supplement101, metadata196 and catalogue53 independently pass; strict77/full-unit continue. |
| 2026-10-23 | Recover commands/baselines, verify frozen source and establish writable scratch isolation. |

- completed: root requirements/plan/progress and final_parent reports read.
  Frozen source HEAD e41abc073500d61f224eda4fca1123e74e40544d; submodule
  3dac53c544e546574cb6e8078cd14053d0916113; status empty before execution.
- completed: comparator help confirms `--baseline-label` and `--candidate-label`.
  Baseline `/data/ycfeng/tmp/d30_workflow_fidelity_20261022/candidate`, source713e6a8.
  Locked unit IDs: repairs/evidence/d29_regression_baseline01.json (267FAIL/31ERROR).
  Actual baseline XML and overlay paths recovered from previous parent reports.
- completed: existing asyncopd-py312 has Python3.12.13, Torch2.9.1, aiohttp3.13.3;
  no dependency installation or ambient path addition is required.
- completed: actual baseline XML reparsed with unique IDs: raw8093 cases,
  7617PASS/267FAIL/31ERROR/178SKIP; latest candidate plus65-ID overlay8124 cases,
  7647PASS/267FAIL/31ERROR/179SKIP. Both exact failure/error sets equal locked JSON.
  Evidence: scratch baseline_unit_id_check.json. Candidate comparison will report
  missing IDs and PASS-to-SKIP against both actual baseline views.
- completed: prepare scratch-backed read-only source namespace and independent
  runs. Harness source inspected: run_matrix.py:336-338 cleans repo/cache, so
  direct execution would violate source read-only ownership. Bubblewrap supported
  by the host; each pre-existing source entry will be read-only while generated
  cache remains under unique scratch. Original repo absolute path is unchanged.
- completed: Python3.12 supplement101:101PASS/0FAIL/ERROR/SKIP, exit0,
  JUnit8.598s; metadata196:196PASS/0FAIL/ERROR/SKIP, exit0, JUnit8.208s;
  catalogue53:53PASS/0FAIL/ERROR/SKIP, exit0, JUnit2.581s. No suite counts summed.
  All source-before/source-after receipts remain e41abc07/sub3dac53c, clean.
- completed: metadata CSV checks: two legacy files byte-equal retained baseline;
  selected Step prepare FP8/profiling/empty factor; genuine mismatch
  BF16/approximation/2.0; grouped-GEMM approximation0.5 retained.
- completed: strict77 final04 run exit0/474.553s; compare exit0/2.770s;77/77
  identical,zero differences. Retained auditor exit0; each side451 artifact hashes
  and519 cache files checked,clean capture and source provenance valid. Package
  versions equal baseline/candidate/actual. Original auditor's hard-coded historical
  task ID is preserved in raw output and mapped explicitly in final04_receipt.json.
- completed: full-unit parent04 raw8015PASS/279FAIL/32ERROR/281SKIP,exit1,
  JUnit1265.252s/outer1269.361s. Exact267FAIL/31ERROR locked baseline retained;
  no removals,missing IDs,PASS-to-SKIP or baseline PASS-to-FAIL/ERROR.
  Added12 missing-aiohttp failures and1 missing-Torch collection error all covered
  by same-source101PASS supplement (12 exact HTTP IDs and16 native module cases).
  Raw suite stays unsuccessful; comparison is NO_NEW_CODE_FAILURES_WITH_EXPLICIT_ENVIRONMENT_SUPPLEMENT.
  All196/53 affected IDs pass in full-unit and equal old focused ID sets.
  Prior final03 to fresh final04:zero new/missing IDs or status changes.
  Scratch root `/data/ycfeng/tmp/d29-e41-final-regressions-20261023-6dah8l8s`;
  exact expanded commands/environment/version/mounts in each launch.json.
  Runner: run_suite.py. Required source/evidence audits are prepared, not run early.
- completed: final source/submodule remain e41abc07/sub3dac53c clean;2611 tracked
  files rehashed against during-run inventory,zero differences. No source writes.
- completed: independent suite reports,raw XML/log/expanded commands,strict audit,
  source inventories and exact-ID comparison archived in evidence/. Final
  check-progress finds no route drift;raw unit FAIL and regression gate PASS remain
  explicitly separate. All required regression runs are complete,not broadened.
- pending: record-only commit after staged-path/hash verification.
- outside ownership: root main06 CLI; GPU/native/G3/E2E remain separate.

## Unfinished work and next actions

1. Verify archived hash inventory and stage only this new record directory; commit
   final records. No required regression subtask remains unfinished.
2. Return concise per-suite results,source/submodule HEADs,artifact paths and raw
   failure/environment limits to root. Root owns main06 and wider workflow acceptance.
   No source fix,exclusions or new suite is authorized.
