# D24-G3-STATE-VALUES completion archive

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Archived bounded first-tail value capture, passing CPU verification and parent-only staging handoff. |
| 2026-10-22 | Recorded verified implementation commit 34a72b8. |

## Outcome

The collector now retains the already copied raw/mean bytes for at most two valid logical pages at `first_local_tail_complete`: the computed-token page and its predecessor. Both the actual local-tail and independent full-local request paths use this seam at every TP rank. The existing two full mean wire rows also retain bytes at this boundary only.

The complete hash/geometry census remains intact, later checkpoints remain hash-only, and the independent-state numerical policy is unchanged: mismatches stay **UNVERIFIED**. No BF16/FP8 tolerance, FP8 decoding, native kernel change, extra GPU copy, synchronization, reservation, flag or alternate capture path was added.

## Deliverables

Worktree `/data/ycfeng/Frontier/.worktrees/step5-reference-values`, branch `feat/step5-reference-values`, exact base `0c9d1ef299e27d0e9bddbe4eb888527f966ba718`.

Verified implementation and task records: commit `34a72b8` (`feat(step5): retain bounded first-tail reference values`). The subsequent documentation-only commit records this completed receipt; no code changed after the 550-test final run.

- `tests/integration/step5_pd/r2_reference.py`: keep raw_hex only for selected first-tail logical pages; complete original census unchanged.
- `tests/integration/step5_pd/r2_observation.py`: keep existing mean wire raw_hex only at first-tail; no other edits.
- `tests/unit/test_step5_native_reference.py`: 48 all-rank/caller cases, 21 page/region/boundary cases, exact counts/hashes, unchanged consumer verdict, inactive predecessor, cap isolation, stage pin checks.
- `tests/unit/test_step5_native_pd_r2_observation.py`: update existing boundary-specific mean retention expectation.
- Task records: `task_memory/task_2026-10-22_step5_reference_values/{requirements.md,plan.md,progress.md,summary.md,test_report_2026-10-22_bounded_reference_values.md}`.

## Observed validation

Final superset: **550 passed in 41.21 s**, Python 3.12.13, Torch 2.9.1+cu128, pytest 9.0.2; 60 s timeout with 30 GiB memory scope. Log `/data/ycfeng/tmp/d24_reference_values_final04.log`, JUnit `/data/ycfeng/tmp/d24_reference_values_final04.xml`; exact command and earlier 83/235/539 PASS iterations are in the test report. `git diff --check` PASS. No test failure occurred; injected cap errors are expected isolated witnesses.

Verified conditions include two original diagnostic synchronizations, one reservation/host copy per existing sample, unchanged full logical census, correct BF16/uint8 storage bytes and hashes, no later-checkpoint raw_hex, stale predecessor excluded, non-identical state UNVERIFIED and unchanged after stripping hex. The existing synthetic stage verifier remains not G3-qualified/not clean-timing-eligible. No native package was staged or restaged.

## Budget and handoff

- Three normal pairs: **182,951,936 B** checkpoint hex + **2,260,992 B** mean wire hex = **185,212,928 B** extra payload (176.63 MiB), excluding small raw_hex key/quote overhead.
- With the unchanged cancel/reuse controls both reaching first-tail: **259,072,000 B** upper bound (247.07 MiB). The prior approximate 183 MB counted only three normal checkpoint pairs.
- Current 1 GiB per-file/per-request, 600 s post-return diagnostic and workload 7168/8192/128 limits are unchanged.
- No staging change is needed: existing builder copies the collector, refreshes `reference_helper_sha256` and freezes tool/observer/config/package hashes. Branch source SHA256 values are in the test report; parent must recompute after combining HTTP/census/input corrections and build one new generation.

## Completed, excluded and open work

- Completed: requested CPU capture capability, tests, task records and verified commit.
- Excluded by scope: GPU/platform/auth/cloud/SSH/push/delegation; native staging; connection/census/graph/workload/pressure/core/input edits; old reports/verdicts/packages. Main tree and finished D23 report were not edited.
- Parent-owned next steps: integrate the branch, prepare one fresh diagnostic within 90 min/48 GPUh, capture actual bytes, perform source-backed numerical analysis, then seek separate numerical-rule approval. No numerical rule is proposed or approved by this slice.

Task/session `D24-G3-STATE-VALUES` / `ses_ea074b7bfffe5UImN2k3jg0Uax`.
