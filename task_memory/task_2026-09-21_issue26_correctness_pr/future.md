# Issue 26 Correctness PR — Deferred work

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-22 | Created. Records the `tests/debug/` pointer defect found during Step 8 and the companion-PR gitlink follow-up. |

## 1. `tests/debug/` is referenced but absent from the published repository

**Found:** Step 8 §14.1, while running the two PP2 entry points `AGENTS.md`
names under "Tests".

**Evidence**

```
bash: tests/debug/e2e-level/monolith_mode/scripts/test_dense_tp2_pp2_dummy.sh: No such file or directory
bash: tests/debug/e2e-level/monolith_mode/scripts/test_moe_tp2_ep2_pp2_dummy.sh: No such file or directory
```

`tests/debug/` exists neither on `fix/issue26-correctness-pr` nor on
`origin/main` (`1f694f7`). The published `tests/` tree is `analysis/`,
`comparison/`, `e2e/`, `fixtures/`, `integration/`, `performance/`, `unit/`.
The same section also names `comm_backend_tests/`, which does not exist either;
the CC-backend tests live in `tests/unit/test_cc_backend_*.py`.

Three places still point into the removed tree:

| Location | Reference |
| --- | --- |
| `AGENTS.md` §Tests | `comm_backend_tests/` and `debug/` directory entries, plus two `bash tests/debug/e2e-level/monolith_mode/scripts/*.sh` commands under "Start with:" |
| `tests/unit/test_colocation_release_review_contracts.py:13,14,89,90,139,140,182,193,203` | Resolves `tests/debug/e2e-level/monolith_mode/scripts/` and two scripts under it |
| `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py:16` | Docstring pointer to `tests/debug/flow-level/admission_control_dev_guide_en.md` |

**Impact:** 10 of the 84 baseline unit failures are
`test_colocation_release_review_contracts.py` failing on the missing paths, and
the documented "start here" commands do not run.

**Why it is deferred:** this is one pre-existing defect class inherited from the
release scrub that removed `tests/debug/`, and it is unrelated to Issue 26.
A correct repair is a decision about the published test surface — restore the
tree, or retarget the contract test and the documentation at entry points that
ship — not a documentation tweak. Making the doc read correctly while the
contract test still fails on the same paths would hide the real gap.

**Suggested resolution, for whoever owns the release scrub**

1. Decide whether the co-location review contracts should assert on shipped
   scripts (`examples/architecture/co-location/**`) or on a restored
   `tests/debug/` tree.
2. Retarget `test_colocation_release_review_contracts.py` accordingly.
3. Update `AGENTS.md` §Tests and the scheduler docstring to match.

Equivalent PP2 coverage for this PR was obtained through the example scripts;
see `test_report_2026-09-22_w8_combined_regression.md` §5.

## 2. Re-point the collective-sim gitlink at `main`

`frontier/cc_backend/backends/collective-sim` currently points at `eb7bc4f` on
the companion branch `fix/zero-payload-input-handling` of
`fwyc0573/frontier-htsim`, because companion PR 1 is still draft. Once that PR
merges, bump the gitlink to the resulting commit on `main` and re-run
`tests/unit/test_collective_sim_zero_payload.py` plus the clean-checkout
validation described in `test_report_2026-09-22_w7_collective_sim_zero_payload.md`.
