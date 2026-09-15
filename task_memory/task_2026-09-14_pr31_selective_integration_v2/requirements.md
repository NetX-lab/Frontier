## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-15 | Recorded authorization to commit the current task documents and push them to the existing PR. |
| 2026-09-15 | Recorded the explicit user decision that the branch must not be merged without a later separate authorization. |
| 2026-09-14 | Recorded the original selective PR #31 integration request and acceptance constraints. |

# Requirements

## [Original Request]

Implement the read-only plan at `/data/ycfeng/stepfun-performance-optimization/Frontier/.draft-plan/frontier_pr31_selective_integration_plan_v2_en.md` in the current worktree branch. Do not modify that plan. Convert its staged work into durable local task documents under `task_memory/`, execute the planned increments one by one, and verify each increment.

The deliverable is a new pull request from this branch. Preserve the original PR author's contribution history and source credit wherever the extracted implementation materially derives from PR #31, so that a later merge can credit the contributor.

## Constraints and acceptance criteria

- Start from the current main baseline and selectively extract PR #31; do not merge or cherry-pick the PR wholesale.
- Work in the current worktree branch and keep independently reviewable commits.
- CPU master validation is the primary executable acceptance path because no AMD/MI355X host is available.
- Do not claim real AMD profiling, ROCm runtime, or benchmark/ground-truth parity. Mark unavailable AMD checks as `SKIP: AMD/MI355X hardware unavailable` and preserve runnable tests.
- For refactors affecting execution-time or simulator behavior, run complete CPU E2E and preserve existing numeric/discrete behavior within the repository's comparison tolerances.
- Keep standard ROCm data paths separate from experimental SGLang artifacts and keep unsupported GDN features fail-fast.
- Keep generated caches, profiling outputs, and temporary logs outside versioned source.
- Before completion, record source/candidate evidence, tests, known failures, attribution, and unresolved hardware checks; then prepare the branch for publication as a new PR.

## [Follow-up Decision]

- On 2026-09-15 the user explicitly stated that the current branch must not be merged without separate permission. This remains an active integration boundary: local edits, verification, commit, push, and PR maintenance are authorized; merge, rebase, force-push, history rewrite, and branch deletion are not.

## [Original Request] Documentation publication follow-up — 2026-09-15

> 将当前worktree branch未提交的docs change提交并push到pr上

Publish the 25 Markdown documents in this task directory, including both independent review reports, through the existing `feature-amd-sglang-gdn` branch and PR #33. The directory is ignored by the repository's default rule; explicitly stage these task documents without changing that rule. Preserve both reviews as separate files. This documentation publication does not authorize implementation fixes, test execution, or merging the PR.
