# Stage admission ordering under pipeline parallelism — Progress

## Modification History

| Date | Change |
| --- | --- |
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
| P4 records, commit, push | in-progress | see below |

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
