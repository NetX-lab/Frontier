## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-16 | Recorded explicit D02 scoped residual acceptance; D01 uniform timing and D03 supported lifecycle decisions remain in force. |
| 2026-09-16 | Recorded authorization to publish local commits and the latest independent completion audit to PR #33 for manual review; merge remains unauthorized. |
| 2026-09-16 | Recorded the independent v1.1 completion audit and audit-only boundary. |
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

## [Original Request] Independent completion audit — 2026-09-16

Audit every accepted issue and remediation in `Frontier_PR33_review_revision_plan_2026-09-15_v1.1_en.md` against the current worktree, exact Git/PR state, production code, tests, and reproducible artifacts. Prior completion claims are not ground truth. Classify every work package, preserve dependencies and acceptance conditions, independently check R01–R14 and D01–D03, and specifically verify real attention routing, automatic GDN memory/slots, phase semantics, layer/stage ownership, metrics, integration, CUDA/DEVICE_EVENT evidence, clean-main fidelity, and the historical 9.42x run-phase regression.

Deliver an executive conclusion, complete item matrix, dependency/severity-ordered actionable remaining TODOs, necessary user decisions with RCA/options, and explicit verification gaps. Do not implement remaining fixes, make semantic decisions, publish changes, or merge. Stop after the audit and authoritative TODO list. Preserve original reviews and pre-existing dirty documentation.

## [Follow-up Decision] Audit publication — 2026-09-16

> 将本地已有的commit 和 最新审阅docs都push到remote pr，我需要在remote端人工审阅。

Publish the local commits that are ahead of the remote PR branch together with the
latest independent completion-audit documentation through a normal push to
`feature-amd-sglang-gdn`. This authorizes the documentation-only commit and the
15 already-existing local commits to become visible on PR #33 for manual review.
It does not authorize implementation of the audit TODOs, merge, rebase,
force-push, history rewrite, or branch deletion.

## [Original Request] Runtime-first remediation — 2026-09-16

> strictly follow the doc plan to implement and fix bugs and questions in current worktree branch: task_memory/task_2026-09-14_pr31_selective_integration_v2/Frontier_PR33_New_Execution_Plan_2026-09-16_EN.md

The named W00–W11 plan is the active implementation specification and supersedes the historical audit-only boundary. Its in-scope runtime/interface migrations, tests, cleanup, and local commits are authorized. Preserve all D01–D03 decisions and hardware boundaries. Prior publication requests concern already-existing commits/documents; no new publication is inferred for this remediation.

## [Follow-up Decision] D03 supported lifecycle scope — 2026-09-16

> 保留当前 lifecycle 范围（推荐）

Retain the existing admission, continuation, completion and failure-cleanup scope. No request-level cancellation API exists; StageExecutionContext.cancel(ticket) is stage-ticket cancellation only. A new request-cancellation API belongs to a separate task. D03 is resolved; no new API is authorized or required for this task.

## [Follow-up Decision] D01 uniform timing semantics — 2026-09-16

> 接受已定位的语义修正（推荐）：保留 uniform layer/stage contract，记录 baseline 差异并以独立 oracle 验收。

Accept the two isolated corrections in w03_divergence_decision.md: PD-AF dummy PREFILL must retain physical one-layer values rather than divide them by 32; co-location MoE stage ledger must include all 32 executed layers rather than report the first layer. Preserve the uniform layer/stage contract and record the measured baseline differences. Validate using independent arithmetic oracles; comparator tolerances remain unchanged. This does not authorize blanket acceptance of other discrepancies.

## [Execution clarification] D01 acceptance scope and verified related defects

The user accepted uniform physical-layer/stage ownership, not blanket cross-version numerical equality or a tolerance change. During W10, the same historical dummy `/resident_layers` defect was confirmed in PDD PREFILL and DECODE. The active plan §6 explicitly allows established implementation defects to be fixed directly, and preserving this multiplier would violate the accepted uniform contract. The same correction is therefore applied with independent physical-layer arithmetic. Required W08 reporting repairs also remove repeated layer multipliers and add actual EP lane artifacts. These are implementation outcomes under the authorized plan, not new user quotations or a new D01 waiver.

The 58-case comparator's raw PASS/FAIL results remain visible. Every affected stage/operator and arrival-dependent batch difference is classified in `test_report_2026-09-16_w10_fidelity.md`; common synthetic-profile request/system metrics retain the existing tolerances. Any unrelated or unexplained discrepancy remains a blocker. D02 performance acceptance is still a separate required decision.

## [Decision proposal, superseded by reply below] D02 measured residual — 2026-09-16

The final source `c9f8f904` retains dense run-phase residuals after removal of repeated snapshots and homogeneous block calculations. The agent presented the two concrete options required by active plan §6: accept the measured scoped cost with the current uniform contract, or keep acceptance open and continue a shared layer-view representation investigation. Full evidence and design consequences are in `performance_rca.md`. D02 remains pending; no reply, universal budget or new implementation direction is inferred.

## [Follow-up Decision] D02 scoped performance acceptance — 2026-09-16

> 接受本次测得的残余成本，保留当前 contract，完成本轮交付（推荐）

Accept the measured residual in the final three-pair campaign at `c9f8f904`: small/long dense median run durations14.149→25.225ms and29.309→49.268ms, paired median ratios1.775045x/1.679023x; representative MoE0.978816x. Retain actual per-layer identity, finalized isolation and the uniform layer/stage contract. D02 is resolved for this delivery. This does not establish a universal performance budget, authorize another representation change, imply native hardware verification, or authorize remote publication/merge. Complete the local evidence handoff.
