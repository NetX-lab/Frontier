# Post-key frozen regression primary checks

## Modification History
| Date | Change |
| --- | --- |
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

Machine receipt: `post_key_regression_acceptance.json`, currently
`PARTIAL_ACCEPTED_STRICT77_AND_SHORT_ONLY`. Full-unit, its exact comparison,
necessary environment supplements and final source guards remain pending.
Full CLI/native/G3/E2E acceptance is not established.
