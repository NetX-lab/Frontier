# Fresh79 full-unit parent05 and exact-ID acceptance

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Close exact-ID/environment regression gate after fresh52PASS governance supplement;raw unitFAIL remains. |
| 2026-10-23 | Record actual8610-case raw result,three newPASS IDs and conditional governance failure diagnosis. |
| 2026-10-23 | Record exact active command, actual locked baseline and pending-result acceptance rules. |

TaskID D29-PERFORMANCE-FINAL-REGRESSIONS. Immutable source
79f1a3fd35d42b751cd1520c764ddac717b923d9/tree
8c6db078e852d3b885a6f14bdf5e8007d9ecddb7, submodule
3dac53c544e546574cb6e8078cd14053d0916113; clean before launch.
Cwd `/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-06`.

## Exact execution and environment

```bash
python3 /data/ycfeng/tmp/d29_performance_regressions_final05/run_suite.py unit
```

Executed payload, without case exclusions:
```bash
timeout 1800s /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit -q -p no:cacheprovider --continue-on-collection-errors --basetemp=/data/ycfeng/tmp/d29_unit_parent05 --junitxml=/data/ycfeng/tmp/d29_unit_parent05.xml
```

Python3.10.6 frontier-py310,pytest9.1.1,numpy2.2.6,pandas2.3.3,scipy1.15.3,
sklearn1.7.2,plotly7.1.0,fasteners0.20,ddsketch3.0.1. Torch/aiohttp absent.
PYTHONPATH exactly repo:repo/tests/unit, PYTHONDONTWRITEBYTECODE=1,
TMPDIR/FRONTIER_TMP_ROOT=/data/ycfeng/tmp, OMP/OPENBLAS/MKL threads1;
FRONTIER_TEST_CPU_WRITER_FILE/SHA256 unset. Original repo absolute path retained;
all existing source entries read-only through unchanged bubblewrap isolation,
generated cache/outputs live under unique scratch. No dependency installation,
ambient import-path addition, gate modification or source write.

Complete expanded launch command/environment/packages:
`/data/ycfeng/tmp/d29_performance_regressions_final05/unit/launch.json`.
Raw unit output destinations were absent before execution:
`/data/ycfeng/tmp/d29_unit_parent05.xml`,
`/data/ycfeng/tmp/d29_unit_parent05.log`,
`/data/ycfeng/tmp/d29_unit_parent05`.

## Actual baseline, not inferred expectations

Reparsed original `/data/ycfeng/tmp/d30_unit_baseline01.xml`:
8093 unique cases,7617PASS/267FAIL/31ERROR/178SKIP, SHA256
`4c1629eaf42bde40b033f43e480f77aa5c81cba8ca4dca00398f43cc9eeda25d`.
Latest `/data/ycfeng/tmp/d30_unit_candidate01.xml` plus65-ID
`/data/ycfeng/tmp/d30_canonical_count_green01.xml` overlay:
8124 unique cases,7647PASS/267FAIL/31ERROR/179SKIP.
Both actual FAIL/ERROR sets equal locked
`../repairs/evidence/d29_regression_baseline01.json`, not counts alone.

Retained accepted e41 source result:
`../resumed_regressions/evidence/unit/results.xml`,SHA256
`2a14e4db9990282224e77807cb043f64415ac9399503446aa0b32fb3ed6c61fb`;
8607 unique cases,8015PASS/279FAIL/32ERROR/281SKIP,raw exit1. The12 added
missing-aiohttp failures and1 missing-Torch module collection error require a
fresh79 supplement, not reuse of old e41 PASS evidence. Actual source79 counts
will be read only from its completed XML and are not prefilled from8015.

## Required comparisons

Compare all three baseline views for case loss, new IDs, PASS-to-SKIP and
PASS-to-FAIL/ERROR; compare exact locked267FAIL/31ERROR membership and added/removed
failure/error IDs against e41. Every newly added test must pass:

- `test_collective_artifact_reuses_only_identical_current_text`.
- `test_moe_loader_derives_family_once_without_skipping_unselected_rows`.
- `test_fixed_predictor_groups_reused_by_queries_and_binding_selection`.

All seven changed paths map to fresh full-unit affected modules and parent
selection IDs. No duplicate affected union is planned. Required metadata file
selection actual197PASS and catalogue53PASS will also be joined as passing full-unit
subsets. Group parent81/1oldSKIP,trainer2,parser128 and loader29 IDs must preserve
all prior PASS outcomes;unchanged old Torch skip remains explicitly reported.

Governance52 previously has47PASS/5 missing-Torch FAIL inside e41 full-unit and
does not overlap supplement101. A fresh52 module run using explicitly approved
existing `/data/ycfeng/tmp/d29_py312_cpu_extras01/site` is conditional on actual
new-unit coverage remaining unsettled. This is not an ambient workaround or
authority to run wider suites.

## Raw result and exact comparison

Fresh raw result: **8018PASS/279FAIL/32ERROR/281SKIP**,8610 unique cases;
JUnit1203.307s,outer1206.805s,raw exit1,not timeout. This raw full-unit result is **FAIL** and
is not changed by environment reconciliation. All three added tests above pass;
e41-to79 shows three new IDs,zero missing IDs and zero previous status changes.
Exact comparison command after automatic completion:
```bash
python3 /data/ycfeng/tmp/d29_performance_regressions_final05/compare_results.py
```

| Comparison | Actual result |
| --- | --- |
| Locked historical failure/error membership | All267/31 retained;zero removals |
| Failure/error additions relative to locked IDs | Same12 aiohttpFAIL/1 Torch collectionERROR as e41 |
| Failure/error additions/removals relative to e41 | 0/0 |
| Case loss original/latest/e41 | 0/0/0 |
| PASS-to-SKIP original/latest/e41 | 0/0/0 |
| PASS-to-FAIL/ERROR original/latest/e41 | 0/0/0 |
| Added case IDs original/latest/e41 | 517/486/3 |
| New79 regression cases | 3/3PASS |
| Metadata197/catalogue53 passing full-unit subsets | All197/53PASS,zero case loss |
| Group parent selection | 81PASS/1unchanged TorchSKIP;all oldIDs present |
| Actual trainer parent selection | 2PASS;all oldIDs present |
| Parser parent selection | 128PASS;all oldIDs present |
| Loader MoE/routing parent selection | 29PASS;all oldIDs present |
| Governance parent52 selection | 47PASS/5retained missing-TorchFAIL;all oldIDs present |

Fresh source79 supplement101 covers all12 identical HTTP IDs and all16 cases
in the otherwise uncollectable native-position module. Each full-unit traceback
explicitly identifies missing aiohttp or Torch;complete IDs/tracebacks joined in
`unit_final05_comparison.json`. The supplement is101PASS/0FAIL/ERROR/SKIP,exit0,
not a replacement all-green full-unit run. All279 failure/32 error IDs exactly
match accepted e41 raw sets,including the locked267/31 historical IDs.

The initial comparison was incomplete solely because governance52 contains five
old Torch-absence failures that do not overlap the required101 supplement. Exact
failure IDs and original tracebacks are retained in
`/data/ycfeng/tmp/d29_performance_regressions_final05/governance_failure_diagnosis.json`.
The module's current actual exceptions,not an expected label,justify the conditional
fresh py312 run using existing explicitly approved CPU extras. No code fix is
needed or authorized. The initial comparison JSON remains preserved as
`unit_final05_before_governance.json` before final comparison updates.

Fresh source79 governance module now passes **52PASS/0FAIL/0ERROR/0SKIP**,exit0;
all52 IDs exactly equal original parent52 and all five previously unresolved IDs
pass. Final comparison exit0:
`NO_NEW_CODE_FAILURES_WITH_EXPLICIT_ENVIRONMENT_SUPPLEMENT`. Required regression
gate is **PASS**;raw unit remains8018/279/32/281,exit1. No duplicate affected union
was run. All seven changed paths are covered by actual full-unit modules and the
one necessary governance environment supplement. The old group tensor-interface
Torch skip is unchanged,not a new exclusion or an inferred PASS.

Raw unit XML SHA256:
`cb64bf1bfbba142b5ad7b3fbad90343d1f9134e6a627c35af5017bfc3654a4d8`.
Final exact-ID comparison SHA256:
`55a787e63413241c4f285698f990ab3043d67545453ce11d987209efa6e53f58`.
Compact comparison/execution receipts are retained in `evidence/`;large raw
XML/logs stay at the original new scratch paths with full hashes/byte counts in
`evidence/final_acceptance.json`. Final source79/tree8c6/sub3dac remains clean;
all2288 pre-launch tracked file hashes remain unchanged after all executions.

## Limits

Raw exit1 must remain visible if historical failures remain. A passing regression
comparison is not an all-green full-unit result. No gate can pass with unexplained
new code failure,baseline case loss or PASS-to-SKIP. main08, native/GPU/G3/E2E
qualification remain primary-owned and separate.
