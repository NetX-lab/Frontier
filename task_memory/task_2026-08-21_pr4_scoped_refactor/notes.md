## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-26 | Created the operational notes for the post-PR17 narrow A' implementation. |

# Notes

## Current Worktree

- Worktree: `.worktrees/pr20-post-pr17-merge-20260824`
- Branch: `integration/pr20-post-pr17-merge-20260824`
- Remote operations, pushes, merges, and main-checkout changes remain out of
  scope for this local implementation pass.
- `task_memory/` is ignored by default; task-document commits use explicit
  force-add when a documentation checkpoint is complete.

## Operational Boundaries

- Preserve existing user changes in unrelated files.
- Use `apply_patch` for manual edits and keep temporary logs under
  `/data/ycfeng/tmp`.
- Run focused tests before broad suites. Record exact numeric token widths,
  assignment counts, lane participants, model-call counts, and stage/KV
  identities in the final report.
- Treat raw `per_expert_tokens` as an output projection only. Physical predictor
  and communication inputs must resolve an `EPLaneWorkload`.
