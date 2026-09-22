# Stage admission ordering under pipeline parallelism — Progress

## Modification History

| Date | Change |
| --- | --- |
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
| P0–P4 | pending | P0 is defined (`plan.md` P0, §4); it waits for the owner's start signal (R-5: "暂不执行") |

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
