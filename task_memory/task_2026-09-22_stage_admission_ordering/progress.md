# Stage admission ordering under pipeline parallelism — Progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-23 | R-10 executed: commits `1661bf1`, `a8e8d8a`, `e35242f` and records; all checks pass; R2-06 recorded as P6. |
| 2026-09-23 | R-10: round-2 remediation started (plan §7). |
| 2026-09-23 | Round-2 code review posted to PR 36 (15 inline comments, `review.md`); fixes deferred by the owner. |
| 2026-09-23 | P4 completed: branch pushed at `4bcd616`, PR 36 body updated (still draft), W9-01 resolution recorded in the parent task (`4c2d573`). |
| 2026-09-23 | R-8 / D-9 adopted; both comparisons rerun and pass (`aeeca93`); P4 in progress. |
| 2026-09-23 | P0–P3 and P5 executed. Rule committed (`dac4e69`). Two plan stop conditions reached (C3 witness metric, C7 V5 dense); P4 push held for the user. |
| 2026-09-23 | R-6 received: execution started; P5 (vLLM comparison) added to the plan. |
| 2026-09-23 | Round-1 plan review verified against source and applied to `design.md`, `plan.md` and `requirements.md`; `review.md` created; resume prompt updated for round 2. Not executed. |
| 2026-09-23 | Decisions adopted; records pushed for remote review. |
| 2026-09-22 | Created. Worktree and branch created; defect reproduced on `origin/main`; plan and design written for review. No source change. |

## State

| Item | State | Evidence |
| --- | --- | --- |
| Worktree `.worktrees/stage-admission-ordering` on `fix/stage-admission-ordering` @ `1f694f7` | completed | `git worktree list` |
| Reproduction on `origin/main`, 15 author-run shapes | completed; P0 republishes them as group R0 from published inputs | `design.md` shape table; logs under `/data/ycfeng/tmp/w10_repro/case_*.log` |
| Drain state dump (FIFO, active owners, lane heaps, sync room) | completed | `design.md` "Observed state at the drain" |
| Root-cause diagnosis and option analysis | completed | `design.md` |
| Plan for review | completed; round-1 review applied | `plan.md`, `review.md` |
| Records published for remote review (`.gitignore` exception, docs commit, push, draft PR) | completed 2026-09-23 | commit and PR recorded below |
| Round-1 plan review (10 findings) verified and applied | completed 2026-09-23 | `review.md`; plan D-6, D-7 |
| P0 evidence and baseline | completed 2026-09-23 | 98 cases classified as designed; rerun hashes stable; see "Execution" |
| P1 rule | completed | `dac4e69` (with P2 tests) |
| P2 tests and base negative controls | completed | `evidence/base_negative_controls.log`; all base outcomes as planned |
| P3 rerun and comparison | completed; the first comparison stopped on the C3 witness rule, passes under D-9 | `test_report_2026-09-23_stage_admission_ordering.md` §4 |
| P5 vLLM comparison | completed; the first analysis stopped on dense V5, passes under D-9 | case `calibration/stage_admission_case_001/`, report §5 |
| P4 records, commit, push | completed 2026-09-23 | "Execution" P4 rows |
| Round-2 code review (R2-01..R2-15) | posted; fixes pending the owner's decision | `review.md` Round 2; https://github.com/NetX-lab/Frontier/pull/36#pullrequestreview-5286523149 |

## Commands run (2026-09-22)

```bash
git -C /data/ycfeng/Frontier worktree add -b fix/stage-admission-ordering \
  /data/ycfeng/Frontier/.worktrees/stage-admission-ordering origin/main

# fresh process per shape; scripts in the session scratchpad w10/
python repro_main.py <out> {moe|dense} <attn_dp> <moe_ep> <pp> <num_requests>
python drain_state.py <out> 4      # context FIFO / active owners
python drain_lanes.py  <out>       # lane heaps and sync room
```

Interpreter `/data/ycfeng/envs/frontier-py310/bin/python`, `PYTHONPATH` = the
worktree, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`.

## Publication (2026-09-23)

| Item | Value |
| --- | --- |
| Branch | `fix/stage-admission-ordering` on `NetX-lab/Frontier`, base `main` `1f694f7` |
| Records commit | `62d25b9` (plan, design, requirements, progress, `.gitignore` exception) |
| Draft PR | https://github.com/NetX-lab/Frontier/pull/36 |
| Reviewer resume prompt | `review_prompt.md` in this directory |
| Round-1 review applied | the commit that adds `review.md` (`git log -- task_memory/task_2026-09-22_stage_admission_ordering/review.md`) |

## Execution (2026-09-23)

| Step | Command / action | Evidence | Result |
| --- | --- | --- | --- |
| P0 base set | `python -m tests.e2e.stage_admission_matrix run --set base --jobs 8` at `a054d87` | `/data/ycfeng/tmp/stage_admission_ordering/base/` | G1 30 success; G3a 10 deadlock + 8 success; G3b 6 + 6; G4 12 success; G5 6 success; G7 MoE 2 deadlock, dense 2 success; R0 as `design.md` |
| P0 rerun stability | set `base-rerun`, 4 cases | same | hashes identical; no file excluded |
| P0 pytest | unit and integration with `--junitxml` | `/data/ycfeng/tmp/stage_admission_ordering/base-pytest/` | unit 84 failed / 3644 passed / 49 skipped / 10 errors; integration 11 / 21 skipped / 5 errors |
| P1 | `try_acquire` rule and docstrings | `dac4e69` | 34 passed on the two contract files, no assertion change |
| P2 | tests (a), (a′), (b), (c) | `dac4e69` | all pass after P1 |
| P2 base controls | new tests in a `git archive 799ccb4` export | `evidence/base_negative_controls.log` | 7 failed, 2 passed, each at the planned assertion |
| P3 after set | `run --set after --jobs 8` at `dac4e69` | `/data/ycfeng/tmp/stage_admission_ordering/after/` | 97 success, 1 configuration_rejection (R0 dp2-pp3) |
| P3 compare | `compare --before base --after after` | `compare_base_after.json` | U 50 PASS; L 18 PASS; T 6 PASS, 6 EXPLAIN, 2 STOP (dp4 witnesses) |
| P3 explain | `evidence/explain_t_path.py` | `evidence/p3_t_path_explanation.json` | all 8 differing T cases: same batches and component durations, start times only |
| P3 G2 | unit and integration after-runs | `evidence/g2_*_compare.json` | no regression, no new failure, skips and errors unchanged |
| C6 probe | `evidence/step9_probe/probe_completion.py` | `evidence/step9_probe/` | MoE dp2-ep2-pp2 6/6 (base drains); pp3 W9-02 rejection |
| P5b run a | RJob `exp-0923-022226-151935` | `runs/vllm-instrumented/sa-pp-20260923a/` | dense complete; MoE failed on `_moe_C::topk_softmax` 5 vs 4 args |
| Decision | user: "topk_softmax 统一修复为4 个参数的版本" | `requirements.md` R-7 | recorded overlay patch, checkout unchanged |
| Overlay patch support | `vllm_burst_driver.py overlay --patch`, worker `OVERLAY_PATCH` | `a1b9819`; CPU dry run against an `upstream-v0.10.2` export | accepted; `_custom_ops.py` equals upstream after patch; second application fails loudly |
| P5b run b | RJob `exp-0923-024146-345158` | `runs/vllm-instrumented/sa-pp-20260923b/` | MoE and dense complete, status 0 |
| P5c | `compare_lanes --vllm-run …/sa-pp-20260923b` | `calibration/stage_admission_case_001/analysis/` | 50/52 MATCH; V5 dense n8/n16 MISMATCH (vLLM 0.706/0.865 vs 1.0) |
| D-9 rules | witness by co-execution fraction; V5 gated on MoE only | `aeeca93` | — |
| P3 compare rerun | `compare --before base --after after --output …/compare_base_after_d9.json` | scratch root | U 50 PASS; L 18 PASS; T 6 PASS, 8 EXPLAIN; no STOP |
| P5c rerun | `compare_lanes --vllm-run …/sa-pp-20260923b` | `analysis/` | status PASS: 50 MATCH, 0 MISMATCH, 2 INFORMATIONAL |
| Checks after D-9 | P5a synthetic check; P2(c) integration test | — | synthetic planted round still caught by V3/V4; 3 passed |
| P4 push | `git push origin fix/stage-admission-ordering` | remote head `4bcd616` | records `df7868e`, `fc34341`; `4bcd616` force-adds `evidence/base_negative_controls.log`, which the repository-wide `*.log` rule had kept out of the tree |
| P4 PR body | REST `PATCH repos/NetX-lab/Frontier/pulls/36` (`gh pr edit` fails on the retired Projects classic query) | PR 36 | body carries the rule, commits, C1–C4/C7 and C3 tables, R-7/D-9 and open items; body read back identical; still draft |
| P4 parent note | parent `issues.md` W9-01 Resolution, `progress.md`, case manifest decision `W9-01-scope`; `summary.md` and the test report copied to `w9_01_stage_admission_ordering/` (D-5) | `fix/issue26-correctness-pr` `4c2d573`, pushed | PR 35 still draft |

## Round-2 remediation (R-10, plan §7)

| Step | Command / action | Evidence | Result |
| --- | --- | --- | --- |
| Records | `requirements.md` R-10, `plan.md` §7 | — | recorded |
| R2-01/R2-11/R2-13 | `try_acquire` one branch per scope; active ticket refused; docstring | `1661bf1` | 181 passed on the three context unit files; the new test's scenario: base `False`, `dac4e69` `ValueError`, now `False` |
| R2-03/R2-07/R2-08 | matrix: `sys_arch`, `simulation_mode`, Poisson rate, recipe env; groups G8–G11; cluster-keyed drain report; `--case-timeout`; set lock | `a8e8d8a` | probe set `r2-probe`: PDD first rejected for missing role replica counts, fixed by one Replica per role; online Poisson cells found on lane 0 only (PR 35 W2), burst cells added; lock and 2 s timeout checked |
| Base for G8–G11 | rule file swapped to `1f694f7` at `a8e8d8a`, `run --set base --group G8 … G11 --jobs 16`, then `git checkout` of the file | scratch `base/` | 12 `admission_deadlock` (G8 4, G9 burst 4, G10 MoE burst 4), 38 success |
| After set | `run --set after-r2 --jobs 16` at `a8e8d8a`, clean | scratch `after-r2/` | 147 success, 1 configuration rejection (R0 dp2-pp3, W9-02) |
| Identity | `after` vs `after-r2` | scratch `identity_after_after-r2.json` | 98/98 identical |
| Compare | `compare --before base --after after-r2` | scratch `compare_base_after-r2.json` | 120 PASS, 12 EXPLAIN, 16 informational, 0 STOP |
| Explain | `explain_t_path.py <root> after-r2 …` | `evidence/r2_t_path_explanation.json` | 12 EXPLAIN: same batches and durations, start times only |
| Tools | compare_lanes N1/N4 rows, placement; driver overlay and patch; tool unit tests; `synthetic_check.py` removed; decomposition and probe scripts | `e35242f` | 9 passed; 7 fail on the `ecff89a` tools |
| Decomposition | `decompose_co_execution.py` on runs a and b | `analysis/co_execution_decomposition_*.json` | no disjoint pair; aligned M5 dense 0.66–0.98, MoE 0.988–0.994 |
| C7 rerun | `compare_lanes --after after-r2` on run b | `analysis/` | PASS, 56 rows, controls hold |
| C6 probe | `probe_completion.py` with `PYTHONPATH` only | scratch `r2/c6_probe/` | 6/6 for both shapes |
| G2 | unit and integration, `--junitxml` | `evidence/r2_g2_*_compare.json` | 0 regressions, 0 new failures, 0 skip changes |
| Records | test report §8, `summary.md`, `design.md`, `review.md`, manifest, workflow-gap summary, plan §7 amendment and D-9 (b) | this commit | — |
