# D23-ROOT-METRICS Requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Record delegated scope and acceptance criteria. |
| 2026-10-22 | Add blocking reviewer P1 owner-projection follow-up and public-result acceptance. |

## Original request
[Original Request]
> You are a subagent spawned by another session.
> Task D23-ROOT-METRICS. Goal: fix demonstrated full CLI blocker in existing measured root metrics/trace projection, preserving inclusive root exactly once and truthful OperatorCost metadata/counts. Create worktree /data/ycfeng/Frontier/.worktrees/step5-root-metrics branch feat/step5-root-metrics from current parent HEAD61cbac9 (verify); no edits to parent or other worktrees. Own only relevant existing metrics/attention projection code, focused regression tests and task_memory/task_2026-10-22_step5_root_metrics/ (requirements/plan/progress/test_report/summary). Root AGENTS/global instructions apply; no further subagents, GPU, native runs, push, README edits or broad refactors. Parent owns full CLI, merge and legacy77. Read /data/ycfeng/Frontier/.worktrees/step5-mrv2-main/task_memory/task_2026-10-22_step5_mrv2_main/core_blocker.md; parent already ran its minimal repro and got exact KeyError step5_layer_e2e at metrics_store.py4504 in /data/ycfeng/tmp/d23_ledger_parent_red01.log. Fixture root .6244ms/count23; actual source c9e9388f offline/online exit1 after1032s, no D events. Current core32fb934 fixes only graph-capacity invariant. Metrics get_attention_trace_op_times currently separately rejects root because family root predictor_target=True/e2e_trace_target=False/execution_time_attr=None. Need support true scalar-free measured inclusive root at existing projection seam; merely .get is insufficient, changing synthetic inputs/scalar fields, dropping root, disabling ledger, or inventing child costs invalid. Check all analogous operation metric iterators, stage ledger sums, kernel counts/gaps, per-layer and Stage ownership and actual trace metadata. Reuse family/role/OperatorCost registry classification; avoid hardcoded model names and new paths. Giant metrics_store should gain minimal wiring only, use its existing module-size exception and update justification if needed. Diagnosis workflow: run provided fast repro RED before modifying behavior. Parent ranked hypotheses: 1 scalar dictionary falsely assumed total; 2 trace enumerates children but omits supported inclusive root; 3 aggregation might double count root/children. Falsify with exact totals/count/evidence tests. Add meaningful regression through actual metrics/trace call sites, RED then GREEN, covering mixed root/stage-owned fields, diagnostic children not billed twice, malformed/unknown scope still rejected and legacy dense/inclusive paths. Use /data/ycfeng/envs/frontier-py310/bin/python with PYTHONPATH repo:repo/tests/unit TMPDIR=/data/ycfeng/tmp. Read GLOSSARY/docs/adr and relevant code; no scaling/default/fallback to suppress errors. Commit verified scoped code then records. Return compact task ID/session, commits, changed files/lines, exact results/logs and unresolved checks. Don't claim full CLI/native/admission passes.

## Acceptance
- The original `.6244 ms` / `23 kernels` root reaches the existing component ledger and trace projection without error, once only.
- Mixed layer and stage-owned fields retain their authoritative owners; component sums equal runtime stage totals.
- Diagnostic children are not additional physical charges or fake kernel costs.
- OperatorCost duration, measurement type, kernel count, hashes, evidence kind, exact lookup, and kernel-gap ownership survive actual trace publication.
- Unknown/malformed scopes and overlapping physical charges remain rejected.
- Focused real call-site regressions go RED before the fix, GREEN after; dense/MLA/legacy inclusive behavior remains covered.
- Minimal wiring in the oversized metrics store uses its existing module-size exception; unrelated cleanup is excluded.

## Evidence limits
CPU fixtures are structural regression evidence only. Full CLI completion, native accuracy, serving qualification, admission, and the parent's legacy77 matrix are not this task's PASS claims.

## P1 follow-up

[Original Request]
> D23-ROOT-METRICS follow-up, blocking reviewer P1. Continue same worktree/branch, own same scoped files/task records; no further delegation or platform. Parent independently verified161PASS1deselected and inherited cache-order same failure/error on both trees; merged your source at f237362. FullCLI and new77 have NOT started yet because review arrived. Reviewer: metrics_store.py4370-72 recursively computes stage_execution_time owner ledger with newly moved4498-4506 attention projection. Real measured predictor at sklearn_moe_execution_time_predictor.py1469-1488 applies first input_cast/last final_residual via _replace_operator_time_values; execution_time.py620-622 retains root duration but drops OperatorCost there. Authoritative stage layers still have proofs; non-authoritative stage-only owner does not. New projection thus raises Inclusive attention root step5_layer_e2e requires OperatorCost on reachable first/last paths. Existing positive owner fixture supplies fake-valid proof and misses bug. Required: reproduce through existing public predictor first/last -> Stage ledger before source edits; this is shortest real call-site RED. Owner ledger must project ONLY stage-owned fields, ignoring its irrelevant attention payload; keep strict root validation for authoritative layers. Use smallest existing projection mechanism, no .get/default bypass on actual layers, don't manufacture owner proofs, don't change predictor cost or _replace semantics to mask ownership. Preserve canonical/scalar field naming, once-only boundaries, kernel-gap owner semantics and legacy stage totals. Add first/last public-result regressions through metrics and mixed owner proof absence/invalid-per-layer fields ignored. Rerun affected suite, record P1 cause/RED/GREEN, commit scoped fix and return. Parent has fixed source worktree for CLI staged at f237362; don't touch it. Reviewer session ses_e9f8b6cffffeSZiBfBvZkmXBQa. Exact review text summarized above; parent owns rereview/merge/fullCLI/77. Return compact commits/counts/logs and code delta gain.

Acceptance additions:
- Public non-dummy predictor first/last results with proof-free stage owners reach the Stage component ledger and actual metrics/trace publication.
- Only authoritative layer roots require valid OperatorCost; malformed owner layer payload and owner layer kernel gap do not affect billing or acceptance.
- Existing stage-only canonical/scalar field names and once-only counts remain unchanged; predictor and replacement semantics are untouched.
- No rereview, full CLI, legacy77, platform, parent worktree edits, or further delegation in this follow-up.
