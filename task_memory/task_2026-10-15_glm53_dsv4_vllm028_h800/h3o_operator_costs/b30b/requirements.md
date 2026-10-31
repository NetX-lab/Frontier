# B30b continuation requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-31 | Recorded the authorized sidecar continuation and protected ownership boundaries. |

## [Original Request] 2026-10-31

Continue the authorized B30b F1–F3 small-microbenchmark implementation in the existing worktree /data/ycfeng/Frontier/.worktrees/b30b-ex-situ-link-probes. Main integration owner is working separately on B29 gate/census, F1a CPU replay, and main task record corrections. Your immediate sidecar output: verify the existing B30b patch and current stage JSONs, run appropriate existing tests/dry-runs, fix concrete in-scope defects if needed, and produce a reviewable scoped commit plus current budget/case counts. Do not submit or control GPU jobs; main owner will do that. Read applicable AGENTS.md, main requirements decision 46 (F1-F4 diagnostic comparison, measured CC remains formal; ingress A approved), current spec/tickets 05/06/09/19 and B30b assignment/README before work. Persistent task directory: task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b30b/. Initial base b0e568f5 has uncommitted collective profiler modifications, new cuda_launch_depth.py, rank_placement.py, tests/analysis/v4_dummy_kernel_count.py, tests/unit/test_b30b_ex_situ_probes.py, and g5h/launch/runtime_check.py modification. Preserve other edits. Confirm no active writer and snapshot existing diff before changes. Write only B30b implementation/test files and B30b-local evidence/report; do not edit main requirements/spec/plan/progress/tickets, other worktrees, or README.md. Core changes are limited to this approved profiling work (no simulator behavior/backend interfaces/config-schema changes). If current patch exceeds 5 frontier files/500 changed lines report the actual counts to main before further expansion. Run tests/unit/test_b30b_ex_situ_probes.py, affected collective profiling tests, G5h launch tests, exact fresh B30b dry-run; verify old G5h/K5 generated plans are byte-identical through existing preservation check. Current stage may have A=132 per node, B=142 per mode, C=80 (old README/dryrun claims 496 B /29.65 GPUh are stale). Verify actual generated jobs/cases/budget from files, do not trust summary. CPU Python /data/ycfeng/envs/frontier-py310/bin/python; no unqualified python command. Use /data/ycfeng/tmp for logs/caches. Record exact commands/environment, expected/actual values PASS/FAIL and limits in test_report_2026-10-31_b30b_ex_situ_probes.md in the local task area. Commit verified code+durable local task artifacts scoped to B30b; list changed paths, commit, tests, fresh plan and budget paths in your final answer. Do not delete/move files, change gates, apply B30a to the main branch, or rewrite existing published measurements. Code/comments formal English. Reasoning effort ultra requested previously; proceed with that depth.

## Decisions and acceptance

- This request overrides the assignment's earlier instruction to leave committing to the lead. Deliver one verified scoped commit in this existing branch.
- Decision 46 controls diagnostic use: measured CC remains formal; F1b/F2/F3 generate diagnostic inputs only. B29, F1a, F4 and all main records remain with the integration owner.
- Read-only reference: `../codex_assignment_B30b_ex_situ_link_probes.md`, local `README.md`, main `requirements.md` decision 46, current Spec v2 at `/data/ycfeng/Frontier/.local-draft/new-models/Frontier_GLM53_DeepSeekV4Flash_vLLM028_H80064_Spec_v2.md` and tickets 05/06/09/19.
- Required validation: focused B30b, affected profiling and G5h tests; fresh generation of both stages; preservation of recorded G5h/K5 listings; explicit case/budget arithmetic; defect evidence and regression checks if a defect is found.
- Keep the original diff snapshot and all earlier published measurements/plans intact. New dry-run and preservation evidence use new paths.
- CPU-only evidence does not establish real H200 queue capacity, physical/NIC placement, measured link parameters, or any formal accuracy gate.

## [Original Request] Continuation in Codex CLI, 2026-10-31

Continuing the main task in Codex CLI. Please finish your existing B30b bounded probe implementation and validation in /data/ycfeng/Frontier/.worktrees/b30b-ex-situ-link-probes. Do not duplicate or expand the slice. Return fresh generated plan paths, case counts, per-job wall/GPU-hour estimates, validation commands/results, changed file paths, and any scoped commit already created. Preserve old G5h/K5 plans. Keep all GPU submission/control with the lead; do not modify main task docs or README. Main lead will record the Codex migration and B29/A10 evidence while you finish.

## [Original Request] Final steering, 2026-10-31

boss YC has resumed in Codex CLI; mainline at 058e4ba0. Your worktree still shows the B30b profiler/runtime-check edits plus four new files. Please complete your existing bounded probe/fit preparation scope (do not launch GPUs): fresh dry-run plans/counts, kernel-depth evidence, exact two-job resource/GPU-hour bounds, preservation of old G5h/K5 plans, focused validation, task-local records, then a scoped commit. Return changed paths, commit(s), exact launch entry points/plans and any concrete blocker. Lead will integrate and directly submit/monitor through personal Python RJobBackend. Please send a concise current status while finishing; do not duplicate mainline F1a replay or B29/F4 comparisons.
