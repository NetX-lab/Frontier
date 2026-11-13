# D29-MODEL-GROUP-REUSE requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record the primary's measured fixed-model group-reuse allocation. |

## [Original Request / Decision]

Implement D29-MODEL-GROUP-REUSE on an isolated e41-based worktree as CORE sole
writer. Authority: D29-REPAIR-PATH-AUTHORITY, primary scope/design74517c54;
diagnosisd8a08722 archivedd10314a9. Primary independently reproduced actual
P6128+idle1 pricing3.424/3.369s and1056 descriptions versus0.397/0.349s and0 under
one-variable reuse, identical cost/proof/plan SHA256d2c141f….

Authoritative allocation (read only; do not copy a stale overlay over primary):
`/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/repairs/{model_group_reuse.md,model_group_reuse_scope_amendment.json}`.

Exactly three editable paths:
- `frontier/execution_time_predictor/moe_routing_workload.py` (retained ceiling175).
- `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` (128).
- `tests/unit/test_prediction_measurement_bindings.py` (260).

Relative to e41: source<=45,test<=60,total<=105 gross. CORE4404,111paths/global10000
remain unchanged. Donors and earlier repair steps remain primary records only.

Reuse actual resolved groups once per fixed-model predictor through one existing
consumer-local selection method. Keep independent predictors/models isolated;
layer3/88, family, source/context/native measurement binding, selected estimator,
point admission, positive exact duration/count proofs and plan checks remain live.
Do not change the generic resolver/profiler, PRICING step5_cost, registry, flags,
inputs, numerical rules or gates. No global cache, unsupported invalidation or
independent public mode.

Acceptance: RED before fix at the actual dual-consumer/binding seam; affected
consumer/binding tests; original main07 P6128 pricing loop against committed
candidate without `--reuse-groups`, cold/warm description counts and timings,
identical full cost/proof/plan hash; residual profile and exact budget/diff.
Probe changes may only rebind source directory/revision, with exact diff retained.

No full CLI/strict77/full-unit/platform/GPU/network runs or additional agents.
Primary owns review, final freeze and original2400s offline/online acceptance.
