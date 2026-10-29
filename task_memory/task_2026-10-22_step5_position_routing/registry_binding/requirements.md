# D29 registry binding repair requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record delegated D29-REGISTRY-SEAM-IMPLEMENTATION scope and acceptance. |

## [Original Request]

TaskID: D29-REGISTRY-SEAM-IMPLEMENTATION. Work only in the
`step5-registry-bindings` worktree, branch `feat/step5-registry-bindings`,
based on `d96b3279`. Preserve the root-owned submodule gitlink and all peer work.
No agents, platform/GPU/network/dependency changes or peer imports.

Retain/query actual native binding, source and selected group metadata in the
existing registry. Keep global typed serialization and existing cache hashes
unchanged. Ambiguous native queries fail; family views and registered bundles
retain all native variants. Connect `_step5_ffn_model` selection and
`_step5_position_queries` physical-row enumeration, using existing routed groups.

Only four code/test paths are authorized: `prediction_model_registry.py`
(120 gross against `1aee0af`), `test_prediction_measurement_bindings.py`
(180 lines), `sklearn_moe_execution_time_predictor.py` (138 gross; baseline98),
and `moe_routing_workload.py` (145 gross; baseline124). Whole repair <=360 gross;
caller delta <=60. Effective overlay and decision are the root's
`repairs/registry_scope_amendment.json` and `repairs/registry_binding_decision.md`.

Required: actual dual-binder RED/GREEN, FP32 gates at layers3/88, source/group
selection and ambiguity/missing source, same-precision variants, bundle snapshots,
legacy/default/cache controls and existing real shared/independent trainer/binder
checks in frontier-py310. No README edits. Commit verified steps and give exact
interface, commands, receipts, budget and remaining root-owned global gates.
