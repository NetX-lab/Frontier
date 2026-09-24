# Issue 26 Correctness PR — Deferred work

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-24 | §2 resolved: companion PR 1 merged (`d28fe917`), gitlink re-pointed in `1b11eff`. |
| 2026-09-24 | §2: gitlink now `ff11ee6`. §3: S43 and S42 transferred to their own tasks by the user's decision; S44 added as a proposal for the S43 task; the S43 admission anchor corrected. |
| 2026-09-23 | §3 added: candidate fidelity findings S43 and S42 from the Step 9 G4 ground truth. |
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

## 2. Re-point the collective-sim gitlink at `main` (resolved 2026-09-24)

Resolved: companion PR 1 merged into `main` as `d28fe917`, whose tree equals
`e922c77`. The gitlink moved in `1b11eff`. Companion tests 16 passed, the Frontier
module 3 passed, and the clean-checkout validation passed (`/data/ycfeng/tmp/issue26-correctness-pr/q9_gitlink_20260924/` (`companion.txt`, `frontier.txt`, `clean_checkout.sh`, `clean_checkout.txt`)).
The history below is kept.

`frontier/cc_backend/backends/collective-sim` currently points at `ff11ee6` on
the companion branch `fix/zero-payload-input-handling` of
`fwyc0573/frontier-htsim`, because companion PR 1 is still draft. Once that PR
merges, bump the gitlink to the resulting commit on `main` and re-run
`tests/unit/test_collective_sim_zero_payload.py` plus the clean-checkout
validation described in `test_report_2026-09-22_w7_collective_sim_zero_payload.md`.

## 3. PP>1 engine-loop findings from the Step 9 ground truth (S43, S42)

**Found:** the G4 native PP2 run `dpp-g4-20260923a` and the G5 workflow-gap
analysis (`calibration/dp_pp_case_001/analysis/workflow_gap_table.csv` rows WG05
and WG03, plan §18.21). Both findings are outside the Step 9 placement scope.
Neither is authorized as a repair. Under the calibration contract each needs the
user's review decision before a scoped `workflow-repair`.

- **S43, admission after an empty schedule.** At PP>1, vLLM 0.10.2's
  `step_with_batch_queue` appends an empty schedule and then blocks on the
  oldest in-flight batch (`vllm/v1/engine/core.py:385-424`). Requests that
  arrive meanwhile wait in the input queue. Frontier admits whenever a stage
  slot is free (`frontier/scheduler/replica_scheduler/base_replica_scheduler.py:1052-1063`
  for MONOLITHIC and PREFILL, report hook `:1061`; the unified DECODE loop is
  `:896-924`. Corrected 2026-09-24 from `:906`, which is the DECODE loop only).
  In G4 the later burst requests were admitted 21.3–47.4 ms after the burst's
  first route in bursts a–c, and 129.5–311.8 ms after it in burst d. Frontier
  admits them on arrival. The effect is on PP>1 batch composition and TTFT for
  any cluster scheduler, not only `vllm_load_balancing`.
- **S42, DP wave-start and idle dummy forwards.** An idle DP engine runs
  `execute_dummy_batch()` in EP lockstep, and the dummy pass advances the step
  counter (`core.py:1170-1216`). This is not modeled. In G4 burst d, engine 1
  has no step-0 record, which is consistent with this reading. The reading is
  inferred, because the dummy pass writes no record.

**Next step if approved:** a new calibration case scoped to one finding. It
would record a reference-loop expectation first, change one Frontier component,
rerun in isolation, and run a fresh analysis (contract "Analysis Before Code
Change").

**Decided 2026-09-24:** "S43 和 S42 各开一个单独的校准和修复任务". They continue in
`task_memory/task_2026-09-24_s43_pp_empty_schedule_admission/` and
`task_memory/task_2026-09-24_s42_dp_wave_idle_forward/`; this section is their
source record.

- **S44, published counts include undrained input-queue requests (proposal).**
  vLLM publishes a DP engine's counts after the step, and a request that
  arrived during the step is still in `input_queue`, so it is absent from them
  (`core.py:1130-1144`, `:1170-1216`). Frontier's report includes every request
  routed to the lane since it last scheduled. Native G4: 1 of 92 receipts with
  `waiting > 0`; Frontier G5: 24 of 106. No measured placement benefit yet
  (`test_report_2026-09-24_fix_review.md` §6). Proposed as a second row of the
  S43 task, because both concern the same input queue; awaiting the user's
  decision.
