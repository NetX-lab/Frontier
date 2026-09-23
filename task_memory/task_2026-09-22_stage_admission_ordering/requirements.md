# Stage admission ordering under pipeline parallelism — Requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-23 | R-10: round-2 fixes confirmed; R2-02 per the recommendation; R2-03 extended to PDD, online and PD-AF. |
| 2026-09-23 | R-9: code review of PR 36 posted as inline comments; fixes deferred. |
| 2026-09-23 | R-8: D-9 adopted for the C3 witness rule and dense V5; P4 authorized to continue. |
| 2026-09-23 | R-7: ground-truth `topk_softmax` fixed to the four-argument version for the MoE retry. |
| 2026-09-23 | R-6: execute P0–P4 and validate the fix against vLLM on a GPU worker (package P5). |
| 2026-09-23 | R-5: round-1 plan review verified and applied to the records; execution deferred. |
| 2026-09-23 | R-4: D-1..D-5 adopted; push and draft PR authorized. |
| 2026-09-22 | Created from the Step 9 finding W9-01 in `task_2026-09-21_issue26_correctness_pr`; recorded the user's scope decision and the request for a reviewable plan. |

## Origin

Found on 2026-09-22 while probing Frontier boundaries for Step 9 of the Issue 26
correctness PR (package P1(b)). Recorded there as W9-01
(`task_memory/task_2026-09-21_issue26_correctness_pr/issues.md`). The defect
predates both stacked PRs; the three files involved are byte-identical to
`origin/main` at `1f694f7`.

## Requests

`[Original Request]` (2026-09-22, after the W9-01 report with three options and
a recommendation to fix it as a separate correctness item):

> 采纳你的推荐，继续

`[Original Request]` (2026-09-22, interrupting the first source read):

> 先给出"共享 admission 排序问题"的修复计划,将具体的计划落地到文档，由我审阅

`[Original Request]` (2026-09-23, after the first external review of PR 36 at
`a6ec6a6`; the pasted review is recorded finding by finding in `review.md`):

> 以下是最新review结果，请你核实每个comments，采纳高价值和必要决策，修复完善docs，暂不执行。

`[Original Request]` (2026-09-23, after round 1 was applied and pushed):

> 按照已有plan执行上述修复（该修复需要和在gpu worker上运行的vllm进行合理的对比验证，确保修改的有效性）

`[Original Request]` (2026-09-23, during P5, after the first GPU run failed on the
MoE `topk_softmax` ABI):

> 我先提前决策，避免中断任务：topk_softmax 统一修复为4 个参数的版本

`[Original Request]` (2026-09-23, answering the two stops of the test report §6):

> 采纳你的推荐，继续

Quality gates the user repeated for every core-module change in this line of
work, carried over verbatim:

> 对frontier 核心模块的代码的修改和实现上，确保可读性和可维护，任何引入的修改和实现都应该是高价值的（要么对fidelity有收益，要么与模拟功能直接相关，不可替代），禁止hard-coding，禁止临时补丁，禁止过度防御，禁止冗余性设计和实现，禁止使用ai味命名函数和变量。

## Decisions

| Id | Decision | Source |
| --- | --- | --- |
| R-1 | The defect is fixed as a separate correctness item, not inside the Issue 26 feature branch. Step 9's PP>1 packages stay paused until it lands. | user, 2026-09-22 |
| R-2 | Branch `fix/stage-admission-ordering` from `origin/main` `1f694f7`, worktree `/data/ycfeng/Frontier/.worktrees/stage-admission-ordering`. | agent, under R-1 |
| R-3 | No source change before the user reviews `plan.md` and `design.md`. | user, 2026-09-22 |
| R-4 | Plan decisions D-1..D-5 adopted as recommended. Push the branch and open a draft PR so the review happens on the remote; the reviewer resumes from a prepared prompt. | user, 2026-09-23 |
| R-5 | Verify every review finding against the source, adopt the high-value and necessary corrections into the records (dispositions in `review.md`, new decisions D-6 and D-7 in `plan.md`), and do not execute: no P0 run, no source change. The docs commit is pushed to the draft PR under R-4. | user, 2026-09-23 |
| R-6 | Execute P0–P4 as planned. The fix must also be validated against vLLM running on a GPU worker, in a comparison designed to show whether the change is effective (package P5 in `plan.md`). The request authorizes the GPU job within the standing GPU rules below. | user, 2026-09-23 |
| R-7 | The vLLM ground truth uses the four-argument `topk_softmax` (wrapper and call). Applied as the recorded overlay patch `calibration/stage_admission_case_001/inputs/groundtruth_overlay.patch` on the one retry job; the vLLM-BS checkout is unchanged. | user, 2026-09-23 |
| R-8 | Adopt both recommendations (plan D-9): contention witnesses pass on a strictly larger co-execution fraction; V5 gates MoE only and reports dense. Continue to P4. | user, 2026-09-23 |
| R-10 | Fix the recommended round-2 findings (R2-01, R2-04, R2-05, R2-07 to R2-11, R2-14, R2-15), with R2-12 and R2-13 as documentation. R2-02: restate the D-9 rationale with both sources and quantify the start-offset part without a new GPU job. R2-03: add PDD and online cells, plus PD-AF online if needed. R2-06 had no answer: record the pre-merge step only (plan §7). | user, 2026-09-23 |

## Constraints carried from the parent task

- `rm` is authorized; `mv`, destructive overwrites, history rewrites, force
  pushes and branch or worktree deletion are not.
- Pushing this branch and opening a draft pull request were authorized on
  2026-09-23 (R-4). Merge, marking ready, force-push and history rewrites remain
  unauthorized.
- No `Co-Authored-By: Claude` on commits or pull requests.
- Temporary files under `/data/ycfeng/tmp`; the simulator interpreter is
  `/data/ycfeng/envs/frontier-py310/bin/python`.
- Never `cd` into the original repository root; use `git -C` and absolute paths.
- GPU work (R-6): StepMind Python `RJobBackend` only, `i-fengyicheng` personal
  auth, `charged_group="codesign"` only (`steptron_ci` paused until the user
  allows it again), `positive_tags=["H800"]`, submitted from this machine with
  local NFS mounts, launcher kept alive, no resubmission while queued. Cloud
  volume access is confined to `/mnt/codesign-exp/ycfeng`. Credential values
  stay in restricted files and process environments; never print or record
  them, and keep shell tracing off.

`[Original Request]` R-9 (2026-09-23, after P4):

> review pr36，将review comments提交到该remote repo的pr36上，暂不执行修复。

Outcome: round-2 review recorded in `review.md` and posted to PR 36 as one
`COMMENT` review with 15 inline comments. No source or test change.

`[Original Request]` R-10 (2026-09-23, after the round-2 review). The question
listed the recommended fixes, two options for R2-02 (recommended: restate D-9
with both sources and measure the post-synchronization start without a new
GPU job), and asked for the scope of R2-03 and R2-06:

> 确认，执行上上述修复； R2-02 采纳你的推荐；R2-03需要补充  PDD+online（如果你认为pd-af+online有必要，请一并补充）
