# Parent verification of post-named-query frozen regressions

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Verify four completed short suites with27 hashes and independent JUnit comparisons; strict/full-unit remain running. |

## Bound source and acceptance boundary

Source `dbe19f531ff68d9edafc480418e8927c40eb1c60`, tree
`6c43090097e92c6fc0c33aa7358e69c2cf092705`, frozen at
`/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-08` with clean
submodule `3dac53c544e546574cb6e8078cd14053d0916113`.

Parent result is **PASS_SHORT_SUITES_ONLY**. No test was rerun. The parent read
only completed suite artifacts, compared raw/retained execution and comparison
receipts, verified27 file hashes and re-parsed every current/baseline JUnit ID
and status. The live source HEAD/tree/clean state matches each before/after
execution receipt. Source-wide final hash guards belong to the unfinished
regression pass and are not claimed complete here.

| Independent suite | Actual result | Python/environment | Against46fe |
| --- | --- | --- | --- |
| supplement101 | 101PASS,exit0 | 3.12.13,asyncopd-py312 | All IDs/statuses equal |
| metadata196 selection | 197PASS,exit0 | 3.10.6,frontier-py310 | All IDs/statuses equal |
| catalogue53 | 53PASS,exit0 | 3.10.6,frontier-py310 | All IDs/statuses equal |
| governance52 | 52PASS,exit0 | 3.12.13,asyncopd-py312 | All IDs/statuses equal |

These counts are not summed. Metadata's legacy directory label196 does not
change the observed197 count. Against79, the approved old test
`test_collective_artifact_reuses_only_identical_current_text` is missing and
`test_collective_artifact_is_fixed_and_failed_initial_reads_are_not_published`
is new and passes. All common statuses match. The raw old-ID presence result
remains failure; no alias or waiver is introduced. Catalogue also matches79;
governance also matches the accepted52-test parent receipt.

## Commands, artifacts and iterations

The owner ran the existing `run_suite.py SUITE` mechanism in
`/data/ycfeng/tmp/d29_post_named_query_regressions_20261024/`. Full exact commands,
package versions, source/scratch mounts, single-thread environment and raw exits
are retained in each suite's `execution.json` under
`post_named_query_regressions/evidence/`. Runs use30G cgroups and read-only source
namespaces, with no binary projection or dependency installation.

Parent command, run from the primary worktree:

```sh
timeout 60s sudo -n systemd-run --scope -p MemoryMax=1G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/resumed_performance/verify_post_named_query_shorts.py > /data/ycfeng/tmp/d29_named_query_parent01/frozen_shorts_parent.log 2>&1
```

PASS first verifier execution. Receipt:
`post_named_query_regression_acceptance.json`, SHA256
`74673c1f3244cb2f831fb3f8e6d4fceccc8db1dc74c184772b6a5dd83ddca58e`.
Its checks retain exact artifact paths/hashes and all baseline differences.
An earlier schema-discovery command raised `KeyError: 'baselines'`: supplement101
uses the single-baseline `prior46fe_xml` field. Discovery was corrected before
validation; no artifact or test result was altered.

## Unfinished work

Strict77, canonical full-unit, final comparisons/guards and owner records commit
are still pending. Whole6144/1024/2400s CLI and native/GPU/G3/E2E remain separate
unfinished gates. This receipt does not qualify any of them.
