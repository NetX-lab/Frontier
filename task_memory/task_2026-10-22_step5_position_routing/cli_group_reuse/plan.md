# D29 model-group reuse plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record scoped repair dependencies and fixed-model lifetime evidence. |

Dependency: isolate e41/3dac + inspect supported lifetime -> RED at real dual
consumers -> local reuse -> affected tests -> code commit -> actual pinned
pricing cold/warm and residual profile -> final budget/handoff.

| Step | Completion condition | State |
| --- | --- | --- |
| Source/lifetime | e41-based worktree and3dac pin; no supported model mutation | completed |
| Real regression RED | Both consumers plus actual layer3/88 binding/query/proof checks; multiple predictors | completed |
| Implementation | One predictor-local full-group tuple; two consumers select current layer each time | completed |
| Affected checks | Consumer/binding/trainer/source suites, original negative coverage preserved | completed |
| Pinned actual pricing | Original point/setup unchanged, no reuse flag, same full result hash, cold/warm counts | completed |
| Handoff | Exact source/test/per-path caps, immutable commits and residual limits | completed |

## Supported lifetime

`BaseExecutionTimePredictor.__init__`, base_execution_time_predictor.py:35–37,
binds `_replica_config` and `_model_config` once. Production predictor modules
have no reassignment to `_model_config` and no mutation of its attributes during
execution. Simulator:164–182 creates a separate predictor per role from that
role's replica config and shares only the model manager. MoE constructor:611–620
passes that fixed config through the superclass. Supported Step P/D excludes
speculation/MTP; no model hot-swap route exists in the approved lifetime.

`step5_layer_metadata` in PRICING-owned step5_cost.py:28–41 caches runtime geometry
by layer/parallel configuration. It is not the complete routed grouping used by
selected measurement identity. This repair stores only the immutable full
RoutedLayerGroup tuple on the same fixed-model predictor. It selects layer3/88
anew at each consumer call; estimator/context/query/cost/proof selection is not
stored. No global sharing or mutation invalidation is introduced.
