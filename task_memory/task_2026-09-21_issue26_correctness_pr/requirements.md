# Issue 26 Correctness PR — Requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-21 | Recorded the original request, the specification hand-off, and the decisions from the planning interview. |

## [Original Request] 2026-09-21

"this is a new task and current is plan mode, we need to discuss and land key docs first. plz read and analysis draft `.local-draft/Frontier_Issue26_Correctness_PR_Execution_Spec_2026-09-21.md` and land doc. if there is something need to consult me, use grill-me skill."

The draft specification is landed verbatim in `plan.md` together with an Amendments table.

## Decisions from the planning interview (2026-09-21)

| Question | Decision |
| --- | --- |
| Q1/Q7 Task document location | `task_memory/` inside the PR worktree; narrow `.gitignore` exception so the records are published with the PR. No `docs/development/` tree. |
| Q2 Step 7 collective-sim zero-payload | Decide at Step 7 (facts: candidate gitlink `e564935d…` unreachable on the configured remote; no local clone). |
| Q3 GPU path | StepMind runbook exists; GPU workers available. Simulator runs on local CPU; GPU only for Step 6 native check and additional profiling CSVs. |
| Q4/Q8 2,000-line gate | Full cleanup + split for the four touched oversized modules only. |
| Q9/Q10 PR organization | Stacked: `refactor/oversized-module-split` first, `fix/issue26-correctness-pr` based on it; both draft PRs open, correctness base = refactor branch, retarget after merge. |
| Q11 Refactor acceptance | ≥50 scenarios; per scenario `request_metrics.csv` value-identical and `system_metrics.json` identical after removing timestamps/run ids; any difference is FAIL unless an explicitly approved fidelity fix. |
| Q12 Task directories | Two: `task_2026-09-21_oversized_module_split` and `task_2026-09-21_issue26_correctness_pr`. |
| Q5 Publication | Authorized: push both branches to `origin`, create/update draft PRs. Not authorized: merge, force-push, history rewrite, closing Issue 26. |
| Q6 Stop point | Stop after the Step 0 push for user review. |
| vLLM reference | Clone into `.real-engine/` (local exclude), pinned to `ea95f57`. |
