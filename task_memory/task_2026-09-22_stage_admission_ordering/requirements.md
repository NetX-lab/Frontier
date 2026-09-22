# Stage admission ordering under pipeline parallelism — Requirements

## Modification History

| Date | Change |
| --- | --- |
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
