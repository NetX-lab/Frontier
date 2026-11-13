# Post-key frozen regression primary checks

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Accept completed46fe regressions after2736 parent hash checks and raw JUnit comparison; retain the failed old-ID predicate explicitly. |
| 2026-10-24 | Accept completed strict77 and three short suites; retain full-unit/final guards as pending. |

Source `46fe83ac55876c10f6dd8ff792705e87efd1905f`, tree
`bbbca98802261662e02d32a59c1a4d6ab20a12e5`, submodule `3dac53c`.
Owner task `D29-POST-KEY-FROZEN-REGRESSIONS`, session
`ses_e98925887ffecetw6nTB6nD8f0`; records are in the isolated
`d29-post-key-regressions` worktree under `post_key_regressions/`.

## Observed results

- **PASS, strict77:** 77 compared and identical; all mismatch/failure/missing
  counters zero. Owner auditor rehashed 451 artifacts and checked 519 cache
  filenames per side. Run 460.640 s and compare 15.700 s, both exit 0.
- **PASS, separate short selections:** supplement101=101, metadata196=197,
  catalogue53=53; no failures/errors/skips. Counts are not added together.
- Metadata has one removed old passing ID and one new passing ID. The former
  `test_collective_artifact_reuses_only_identical_current_text` was replaced by
  `test_collective_artifact_is_fixed_and_failed_initial_reads_are_not_published`
  under the approved run-fixed input contract. The old ID was not executed or
  aliased. Other retained IDs/statuses agree with the prior79 receipts.
- **Primary PASS:** 27 hashes checked: 16 source/retained receipt copies, five
  raw fidelity manifests/results/comparison files, and six old/new JUnit files.
  Primary independently parsed the six JUnit files and verified exact ID sets,
  the disclosed replacement and all fresh passing counts. The 451 artifact
  rehash per side remains owner-auditor evidence, not a repeated parent scan.

## Command and environment

Primary command, cwd `/data/ycfeng/Frontier/.worktrees/step5-preview`:

```bash
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_post_key_parent_checks.py
```

Python 3.10.6; executed as `brainpp`, 30G memory scope, 60s cap. A retained copy
of the checker is `post_key_parent_checks.py`. Owner exact commands, sandbox
mounts, package versions and raw exits are in `post_key_regressions/evidence/`
execution receipts. The primary did not run additional tests or read the active
full-unit log. An initial inline check assumed `results.json`; path discovery
identified `results.jsonl`, and the corrected check passed before writing any
acceptance receipt. This was a checker-path error, not a product failure.

The paragraphs above record the initial partial check. The completed acceptance
below supersedes its pending full-unit status without changing any raw result.

## Completed parent acceptance

Owner records `c6ccae82` and resumed reconciliation `0c6f8d99` are archived as
`e85547fa` and `36af3394`. No missing/interrupted run remains. Primary read the
full-unit report and independently parsed the retained79 and current46fe raw
JUnit, all five current suite XMLs and their exact source/receipt bindings.

| Independent suite/check | Observed result | Parent verdict |
| --- | --- | --- |
| Canonical unit, unchanged no-binary environment | 8020PASS/279FAIL/32ERROR/281SKIP,8612 unique IDs, rawexit1 | No new code failure; approved ID replacement retained |
| Retained79 comparison | Old8018PASS/279FAIL/32ERROR/281SKIP,8610 IDs; all present IDs keep status | PASS status comparison |
| FAIL/ERROR/SKIP ID sets | Exact equality | PASS |
| Added source IDs | Three, allPASS | PASS |
| Missing old IDs | One approved hot-update test | Raw presence predicate remainsFAIL |
| Supplement/metadata/catalogue | 101/197/53PASS separately | PASS |
| Governance supplement | 52PASS; same five raw Torch-failure IDs pass | PASS |
| Strict77 | 77/77 equal | PASS, existing completed evidence |
| Hash/source checks | 2736 checks, including2629 frozen source files,13 guarded inputs,8 scripts and39 records | PASS |

The single missing ID is
`tests.unit.test_collective_sim_cc_backend::test_collective_artifact_reuses_only_identical_current_text`.
Its new run-fixed counterpart passes, alongside the new dual-role constructor
and prepared-point replacement tests. D29-RUN-INPUT-SNAPSHOT already approved
the input-lifetime change and replacement coverage. Primary admits this
declared source/test change; it does not pretend the old ID ran or rewrite the
raw comparator. `FAIL_RETAINED_ID_LOSS_OR_STATUS_CHANGE`, both old-presence
failures and rawexit1 remain visible. This is not all-green unit acceptance.

The first8036/279/32/267 observation remains non-canonical: it exposed an htsim
binary absent during retained79. The owner reran the same selection/cap in the
original no-binary namespace; canonical duration1124.468s within1800s. Original
missing-binary skips were restored by the namespace, not injected skip markers.
Primary verifies both XML hashes and preserves that environment distinction.

Exact final primary command (same cwd and Python3.10.6):

```bash
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/resumed_performance/post_key_parent_checks.py > /data/ycfeng/tmp/d29_post_key_parent_final01.log 2>&1
```

PASS on the first final-check execution. No suite/probe was rerun by primary.
Frozen46fe HEAD/tree/3dac and tracked source remain clean. The machine receipt
is `post_key_regression_acceptance.json`, SHA256
`099352c24939aec28c5f64f8bf58c173a8b00ff4888981bff4619431ba23d91c`, status
`ACCEPTED_NO_NEW_CODE_FAILURE_WITH_APPROVED_TEST_ID_REPLACEMENT`.
Named-model-query successor is not part of46fe. Full CLI/native/G3/E2E acceptance
remains open and cannot be inferred from these regressions.
