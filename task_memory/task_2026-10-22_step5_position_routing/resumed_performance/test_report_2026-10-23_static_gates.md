# Parent final-source static and short-gate acceptance

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Accept full-unit exact-ID comparison and supplements; record both main08 timeouts and clean final source. |
| 2026-10-23 | Verify source79 strict77/short receipts and T2 static binding;preserve pending full gates. |

## Code consolidation

`git merge --no-ff 79f1a3fd35d42b751cd1520c764ddac717b923d9` lands at
feature commitcf894fa41b13715312db909cee5d338fca667fac. Parent
`git diff 79f1a3fd HEAD -- frontier tests data Dockerfile docs/general/module-size-exceptions.md`
is empty. Active CLI/regression worktree remains frozen at79,not the docs/merge tip.

## Short gates

Original commands/environments are retained in performance_regressions reports
and per-suite execution.json. Parent independently reparses the original three
JUnit XMLs,verifies source-before/after79 and clean status,zero error/failure/skip,
all exits0,and rehashes every item in evidence/short_archive_manifest.json.
Observed PASS:supplement101 in9.085s,metadata197 in6.346s,catalogue53 in3.258s.
These are overlapping independent selections and are never summed. The historical
metadata196 label now contains the additional parser regression.

Parent reparses strict77 final05_receipt.json and the77 records on each side;
all comparison counts are zero and environments match. It rehashes both
manifest.json files,both results.jsonl files and final05/comparison.json against
the source-bound receipt. PASS77/77;the regression worker's full audit separately
verified451 artifact hashes and519 cache filenames per side. Parent's first
checker assumed results.json and raised FileNotFoundError;actual directory
inspection establishes results.jsonl,and the corrected checker passes without
changing any result. Machine evidence:short_gate_acceptance.json.

## T2 source binding

Python3.10.6 parent check,under timeout60s,executes:

```bash
sha256sum -c hashes.sha256
```

Cwd:/data/ycfeng/Frontier/.worktrees/step5-d29-t2-performance/task_memory/task_2026-10-22_step5_d29_native_balanced/t2_preparation.
Expected63 successful checks;observed63/63. Parent independently compares all
26 files from the previous full_repository_files map against their SHA256,
source79 Git objects,integrated06 disk and records-worktree disk. All26 pass.
`git diff 79f1a3fd a886466c -- frontier tests data Dockerfile` is empty;whole
changed-path list has only11 existing records and the records worktree is clean.

Parent verifies resource_proposal.json's source/mount pointers,16H800/1800s,
covered H800 authority and explicit false values for H200,readiness,capacity,
submission and this docs-only task's launch permission. The controlled environment
still unsets FRONTIER_STEP5_PROFILE_MANIFEST. New index SHA
571fb0c6b33ad0f498e986716568f43c52ab9dfbfd51e246bd7ef3df287a6b62.
T2/helper hashes remain7d8422d4/63baf647 in the exact machine receipt.
Owner2907/12/53/native4 verification is retained and its indexed receipts are
checksummed;parent does not claim a new installed inventory or GPU execution.

## Full-unit completion

The exact original run and comparison commands, environments and baseline paths
are in performance_regressions/test_report_2026-10-23_unit_parent05.md, archived
atf3a3147e. The parent independently reads the original XMLs with Python3.10.6
and xml.etree.ElementTree, builds unique classname/name IDs, checks every e41
status and the three added IDs, and compares the full failure/error sets.
It rehashes the execution receipts, logs, XMLs and every archive-manifest entry.

Expected: all8607 prior statuses preserved,three new tests pass,no added code
failures or missing cases. Observed PASS:8018PASS/279FAIL/32ERROR/281SKIP;
all8607 prior statuses unchanged and the three added tests pass. Raw exit1 and
JUnit1203.307s remain visible. Supplement101 covers the12 added missing-aiohttp
failures and the uncollectable native module;governance52 passes all52 cases,
including five retained Torch-absence failures. Totals are not summed.
Exact source/count/hash evidence:unit_gate_acceptance.json.

## Main08 completion

The actual command is preserved in test_report_2026-10-23_integrated_point.md.
The wrapper exits1. Offline and online both FAIL by the original2400s limit:
2400.175252s and2400.152131s respectively. Each scheduler records four emitted
decode tokens and schedules decode iteration4;the request is incomplete.
Operator traces occupy1,447,809,841 and1,517,626,194 bytes. These sizes are
observations,not a diagnosis. Source79 and submodule3dac are clean after the run,
verified with git rev-parse HEAD and git status --porcelain in both checkouts.
Mode receipts and validation files remain under the original main08 output root;
compact outcome:main08_execution.json.

Full workflow acceptance still fails. T2 target staging,sealing,actual capacity
and runtime qualification follow workflow acceptance. CORE is building a new
seconds-scale execution-path loop from the retained run;no new repair or full
CLI rerun is assigned. Static binding is not runtime admission.
