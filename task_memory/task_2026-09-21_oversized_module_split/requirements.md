# Oversized Module Split — Requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-21 | Created from the Issue 26 planning interview. |

## [Original Request] 2026-09-21

This task was created as the prerequisite of the Issue 26 correctness PR (`task_memory/task_2026-09-21_issue26_correctness_pr/`). During the planning interview the user decided:

- Q4=c: apply the `AGENTS.md` 2,000-line gate as a full cleanup-first pass and functional split, not as a documentation-only analysis.
- Q8=a: scope is limited to the four modules the correctness PR must touch: `frontier/config/config.py` (5720 lines), `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` (5138), `frontier/execution_time_predictor/shared_prediction_model_manager.py` (4614), `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` (3539). The other five modules above 2,000 lines are out of scope.
- Q9=b, Q10=b: the refactor is a separate PR from branch `refactor/oversized-module-split` (base `origin/main` `1f694f7c549aa3aeeb7c5bbae04e119c09167a77`). The correctness branch is based on it. Both draft PRs are opened; the correctness PR's base is this branch until this PR merges.
- Q11: acceptance is a fidelity matrix of at least 50 scenarios. For each scenario the refactor branch and main must produce value-identical `request_metrics.csv` and identical `system_metrics.json` after removing timestamps and run ids. Any difference is FAIL unless it is an explicitly approved fidelity fix. Coverage: dense/MoE, offline/online, co-location / sequential PDD / sequential PD-AF, dummy predictor and checked-in profiling CSVs, varied request lengths, counts, and QPS.
- Q5: pushing this branch to `origin` and creating/updating its draft PR are authorized. Merge, force-push, history rewrite are not.
- Q7=b: task records are published with the PR through a narrow `.gitignore` exception for this directory.

## Constraints carried from `AGENTS.md`

- Cleanup first: remove redundant, dead, or overly defensive code before splitting; document the reason for anything that remains above 2,000 lines.
- Record proposed boundaries and sequencing before implementing a split.
- Plain ML-system names for new modules; no behavior change, no numeric change, no public import surface break (`frontier/config/__init__.py` star re-export must keep working).
